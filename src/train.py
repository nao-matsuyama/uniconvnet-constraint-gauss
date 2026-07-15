import argparse
import json
import os
import subprocess
import sys
from datetime import timedelta, timezone
from pathlib import Path

# 💡 Docker内での NCCL 通信エラーを回避するための環境変数
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"  # InfiniBandを無効化
os.environ["NCCL_SOCKET_IFNAME"] = "lo,eth0"  # ループバックと標準ネットワークを指定

# 🔥【ここを修正】共有メモリ(shm)とP2Pメモリを無効化し、強制的にSocket通信にする
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

import datetime
from zoneinfo import ZoneInfo  # 👈 日本時間取得のためにこれを追加

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from augment_scinti import AugmentedScintiDataset
from dataset_scinti import ScintiMultiClassDataset
from model_uniconvnet_unet import UniConvNet_UNet_13CH
from models.erf_regularization import erf_reg_loss
from models.gaussian_derivative_dw import load_dense_into_gaussian_derivative
from models.separable_dw import load_dense_into_separable
from models.spectral_dw import SpectralDW
from models.spectral_gaussian_dw import SpectralGaussianDW


def _resolve_path(path_str):
    path = Path(path_str)
    if path.exists():
        return path
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / path
        if candidate.exists():
            return candidate
    return path


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_ema", "ema", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _strip_prefix(state_dict, prefix):
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def load_pretrained_backbone(model, checkpoint_path):
    if not checkpoint_path:
        return

    resolved = _resolve_path(checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    print(f"📦 Pretrained checkpoint を読み込みます: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "backbone.")

    backbone = (
        model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    )
    backbone_state = backbone.state_dict()
    filtered_state = {
        k: v
        for k, v in state_dict.items()
        if k in backbone_state and backbone_state[k].shape == v.shape
    }

    if not filtered_state:
        print(
            f"⚠️ 互換する backbone 重みが見つかりませんでした: {resolved}。"
            " そのままランダム初期化で学習を続行します。"
        )
        return

    missing = sorted(set(backbone_state) - set(filtered_state))
    unexpected = sorted(set(state_dict) - set(backbone_state))
    backbone.load_state_dict(filtered_state, strict=False)
    print(
        f"✅ Backbone に {len(filtered_state)}/{len(backbone_state)} 個の重みを読み込みました"
    )
    if missing:
        print(f"ℹ️ 未初期化の backbone パラメータ数: {len(missing)}")
    if unexpected:
        print(f"ℹ️ checkpoint 側で未使用のキー数: {len(unexpected)}")

    # 機構A: separable の weight_h/weight_v は密の a*.2.weight とキーが違い上の filter で
    # 転移されない。密カーネル(unexpected 扱い)を rank-1 SVD で分離初期化して転移する。
    load_dense_into_separable(backbone, state_dict, verbose=True)
    # ガウス微分基底も密カーネルとキーが違う(coeff/log_sigma)。密 a*.2.weight を
    # ガウス微分基底へ最小二乗射影して係数を初期化する(local 枝/gamma を持たないので、
    # 制約を最初から効かせつつ「最良のガウス微分近似」から fine-tune を始める)。
    load_dense_into_gaussian_derivative(backbone, state_dict, verbose=True)


def build_optimizer(model, args):
    """optimizer を構築。--spectral-sigma-lr>0 なら log_sigma(σ) を高lr・WD0 の別 group に。

    σ は base lr(1e-4) では実質凍結する(実測 |Δlog|~0.005 << σ2倍に必要な 0.69)。σ 専用に
    高 lr を与えると実際に σ が動く。σ に weight_decay を掛けると σ→1 に引っ張られるので、
    σ group は weight_decay=0 にする。sigma_lr=0 なら従来通り全パラメータ base lr。
    """
    sigma_lr = float(args.spectral_sigma_lr)
    sigma_params, base_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if sigma_lr > 0 and name.endswith("log_sigma"):
            sigma_params.append(p)
        else:
            base_params.append(p)
    groups = [{"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    if sigma_params:
        groups.append({"params": sigma_params, "lr": sigma_lr, "weight_decay": 0.0})
        print(
            f"🔧 σ専用 param group: log_sigma {len(sigma_params)} 個を "
            f"lr={sigma_lr} (weight_decay=0) で最適化 → visualize_sigma.py で移動量を確認"
        )
    return optim.AdamW(groups)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train UniConvNet U-Net for scinti segmentation"
    )
    parser.add_argument("--data-dir", default="/workspace/scinti_segmentation")
    parser.add_argument(
        "--view",
        choices=["both", "anterior", "posterior"],
        default="both",
        help="読み込むビュー。anterior=正面(_A)/posterior=背面(_P)/both=両方(既定)。"
        "フィルタは train/val 分割の前に掛かるので、anterior 指定で val も正面のみになる。"
        "胸骨(c12)等の前面固有クラスの取り違えを避けたいときに anterior を使う。",
    )
    parser.add_argument("--pretrained", default="uniconvnet_t_1k_224_ema.pth")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="モデル初期化・学習の乱数シード。対照実験で初期値を揃える",
    )
    parser.add_argument(
        "--freeze-dilation",
        action="store_true",
        help="log_dilation を学習させない (dilation を初期値1に固定)",
    )
    # コンテンツ適応 depthwise (画素ごと dilation ゲート)
    parser.add_argument(
        "--adaptive-dw",
        action="store_true",
        help="ConvMod の a1/a2/a3 を ContentAdaptiveDW に差し替える "
        "(画素ごとに dilation をゲートで選択)",
    )
    parser.add_argument(
        "--adaptive-dilations",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="adaptive_dw 時の dilation 枝 (例: --adaptive-dilations 1 2 4 8)",
    )
    # 周波数ガウス depthwise (per-channel 学習可能 σ, FFT で一定コスト)
    parser.add_argument(
        "--spectral-dw",
        action="store_true",
        help="ConvMod の a1/a2/a3 を SpectralGaussianDW に差し替える "
        "(周波数領域ガウスで RF を制御、σ は --erf-reg-weight/--erf-target-spread で駆動)",
    )
    parser.add_argument(
        "--spectral-init-sigma",
        type=float,
        nargs="+",
        default=[1.0],
        help="spectral σ の初期値(fmap px)。1値で全stage共通、4値で stage0..3 別。"
        "lr 1e-4 では σ-reg で σ を動かしにくいので、狙いの RF に直接初期化する運用を推奨。",
    )
    parser.add_argument(
        "--spectral-init-gamma",
        type=float,
        default=0.0,
        help="spectral 枝のゲート初期値。0 で事前学習の conv と厳密一致(転移温存)",
    )
    parser.add_argument(
        "--spectral-max-sigma",
        type=float,
        nargs="+",
        default=[32.0, 24.0, 12.0, 6.0],
        help="stage 別 σ 上限(fmap px)。深層ほど特徴マップが小さいので小さく。"
        "1値で全stage共通、4値で stage0..3 別。",
    )
    parser.add_argument(
        "--spectral-use-local-branch",
        action="store_true",
        help="SpectralGaussianDW の spatial local branch も併用する。"
        "未指定なら FFT による Gaussian branch のみを使う。",
    )
    parser.add_argument(
        "--spectral-sigma-warmup-epochs",
        type=int,
        default=0,
        help="σ-warmup: 実効σの上限を start から init_sigma(target) へ漸増する epoch 数。"
        "0 で無効。大きい init_sigma を崩壊させず使うための coarse-to-fine 駆動。"
        "使うときは --spectral-init-sigma を狙いの広い値(例 8 6 4 2)にする。",
    )
    parser.add_argument(
        "--spectral-sigma-warmup-start",
        type=float,
        default=1.0,
        help="σ-warmup の開始 σ (小さく。既定 1.0)。epoch1 でこの値から始め "
        "warmup-epochs で init_sigma に到達、その後 max_sigma まで開放。",
    )
    # ERF 正則化 (学習可能 dilation / spectral σ を目標 RF 広がりへ寄せる)
    parser.add_argument(
        "--erf-reg-weight",
        type=float,
        default=0.0,
        help="ERF 正則化損失の係数 lambda。0 で無効 (spectral では σ の駆動力)",
    )
    parser.add_argument(
        "--erf-target-spread",
        type=float,
        nargs="+",
        default=[4.0],
        help="目標 RF 広がり(特徴マップ画素単位)。"
        "1値で全stage共通、4値で stage0..3 別 (低層=小/高層=大)。"
        "例: --erf-target-spread 2 3 4 6",
    )
    # 統一 depthwise セレクタ (dense/separable/spectral/spectral_mix)。機構A/B/C の新モジュール。
    parser.add_argument(
        "--dw-mode",
        choices=[
            "dense",
            "separable",
            "spectral",
            "spectral_mix",
            "gauss_deriv",
            "gauss_pyramid",
        ],
        default="dense",
        help="RFA a1/a2/a3 の depthwise 機構。dense(既定)/separable(機構A 1×K・K×1 分離)/"
        "spectral(機構B 周波数+動的スペクトル切り出し)/spectral_mix(機構C 多ガウス混合)/"
        "gauss_deriv(ガウス微分基底で RF を σ のみに縛る)/"
        "gauss_pyramid(多スケール純ガウス+pointwise DoG)。separable/gauss_deriv は"
        "事前学習の密カーネルを分離初期化/基底射影して転移する。",
    )
    parser.add_argument(
        "--gauss-pyramid-growth",
        type=float,
        default=1.6,
        help="gauss_pyramid の枝間 σ 成長率。a1/a2/a3 の σ = init_sigma·growth^{0,1,2}。"
        "1.0 でカスケード(ガウス半群)のみの増大、>1 で明示的にピラミッドを広げる。",
    )
    parser.add_argument(
        "--freeze-scale",
        action="store_true",
        help="gauss_pyramid の σ を固定(純スケール空間)。未指定なら学習可能"
        "(--spectral-sigma-lr>0 で駆動)。",
    )
    parser.add_argument(
        "--gauss-deriv-order",
        type=int,
        default=2,
        help="gauss_deriv の次数 N(基底数 M=N+1)。制約↔表現力の唯一のノブ: "
        "N=0=純ガウス(最強制約)/N=2(既定)=エッジ・リッジ表現可/N大=無制約へ漸近。"
        "σ(RF スケール)は --spectral-init-sigma/--spectral-max-sigma で指定し、"
        "--spectral-sigma-lr>0 で学習可能にする(log_sigma 専用 param group)。",
    )
    parser.add_argument(
        "--spectral-num-gaussians",
        type=int,
        default=3,
        help="機構C spectral_mix の周波数ガウス混合の成分数 K。K=1 で機構B(単一ガウス)に一致。"
        "既定 3 は erf_gmm_fit の N=3(RFA が集約する a1/a2/a3 の3スケール)に合わせる。",
    )
    parser.add_argument(
        "--separable-rank",
        type=int,
        default=1,
        help="機構A separable の分離 rank R。R 本の 1×K・K×1 の和で近似 (タップ 2RK)。"
        "R=1(既定)は従来の rank-1 分離、R=2 で事前学習カーネルの非分離構造(境界)を回収。"
        "R が大きいほど密に近づき精度↑・コスト↑ (R=K で密に厳密一致)。",
    )
    # データ拡張 (train のみ, 既定オフ=従来と厳密一致)。過学習対策=worst 群汎化の底上げ。
    parser.add_argument(
        "--aug",
        action="store_true",
        help="train データ拡張を有効化 (左右swap付き水平反転+小アフィン+強度ジッタ)。"
        "骨シンチは RF 無関係=精度改善は拡張など RF 以外の軸で狙う。val は不変(seed42固定)。",
    )
    parser.add_argument(
        "--aug-hflip-prob",
        type=float,
        default=0.5,
        help="左右スワップ付き水平反転の確率 (8:R-Arm↔9:L-Arm, 10:R-Leg↔11:L-Leg を入替)。0で無効。",
    )
    parser.add_argument(
        "--aug-rotate",
        type=float,
        default=10.0,
        help="ランダム回転の上限(度)。0で無効。",
    )
    parser.add_argument(
        "--aug-scale",
        type=float,
        default=0.1,
        help="ランダム等方スケールの上限(1±s)。0で無効。",
    )
    parser.add_argument(
        "--aug-translate",
        type=float,
        default=0.05,
        help="ランダム平行移動の上限(正規化grid座標, ±t)。0で無効。",
    )
    parser.add_argument(
        "--aug-gamma",
        type=float,
        default=0.1,
        help="強度 gamma ジッタの強さ(img のみ)。0で無効。",
    )
    parser.add_argument(
        "--aug-brightness",
        type=float,
        default=0.05,
        help="輝度シフトの強さ(img のみ)。0で無効。",
    )
    parser.add_argument(
        "--aug-noise",
        type=float,
        default=0.0,
        help="ガウスノイズの標準偏差(img のみ)。既定0=無効。",
    )
    parser.add_argument(
        "--spectral-alpha",
        type=float,
        default=2.0,
        help="機構B spectral の切り出し比 η=α/σ のスケール。大きいほど帯域を広く残す。"
        "α=2 で実質無損失(AGD厳密, cost∝(2/σ)²)、α=1 で更に安いが僅かにリンギング。",
    )
    parser.add_argument(
        "--spectral-sigma-lr",
        type=float,
        default=0.0,
        help="log_sigma(σ) 専用の学習率。0 で無効(base lr と同じ=従来通り)。"
        "lr1e-4 では σ が凍結する(実測 |Δlog|~0.005)ため、σ を実際に動かすには "
        "1e-2 程度を推奨。この group は weight_decay=0 (σ を 1 に引っ張らない)。"
        "visualize_sigma.py で σ が動いたか確認できる。",
    )
    parser.add_argument(
        "--spectral-pad-factor",
        type=float,
        default=0.0,
        help="機構B spectral: rfft2 前の reflect パディング率(各軸±round(f·辺長))。"
        "0で無効(循環畳み込み)。0.25 程度で境界の巻き込み破綻を抑える(FFTは大きくなる)。",
    )
    parser.add_argument(
        "--spectral-crop-quantile",
        type=float,
        default=0.0,
        help="機構B spectral: 切り出し σ_ref を層内 min でなく分位点にする(0=min)。"
        "σ が per-ch 分化して 1ch でも極小σになると truncation が死ぬのを防ぐ(例 0.1)。",
    )
    parser.add_argument(
        "--tag", default="", help="実験フォルダ名に付ける識別子 (例: erf0.1_t3.6-8)"
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="完了/失敗の Webhook 通知を無効化 (既定は scripts/.notify_webhook があれば送る)。",
    )
    return parser


def _git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def write_run_config(save_dir, args):
    """実験条件を run_config.json に保存 (機械可読)。"""
    cfg = {
        "timestamp": _now_tokyo().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_count": torch.cuda.device_count(),
        "args": vars(args),
    }
    with open(os.path.join(save_dir, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


def _run_notify(status, title, body):
    """scripts/notify.py をベストエフォートで叩く(通知失敗で学習を落とさない)。

    ラッパ notify_run.sh 経由の場合は NOTIFY_WRAPPED=1 が立つのでここでは送らない
    (ラッパ側が生成物込みで通知する → 二重送信を避ける)。--no-notify でも抑止。
    """
    if os.environ.get("NOTIFY_WRAPPED") or "--no-notify" in sys.argv:
        return
    notify_py = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "notify.py"
    )
    if not os.path.exists(notify_py):
        return
    try:
        subprocess.run(
            [sys.executable, notify_py, "--status", status, "--title", title, body],
            check=False,
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001  通知失敗は握り潰す
        print(f"[notify] スキップ: {e}")


def write_run_info(save_dir, args, result):
    """実験条件と結果を run_info.txt に保存 (人間可読)。"""
    a = vars(args)
    lines = [
        "=" * 50,
        " 実験サマリ — UniConvNet-T U-Net",
        "=" * 50,
        f" 実験フォルダ : {os.path.basename(save_dir)}",
        f" 日時         : {_now_tokyo().strftime('%Y-%m-%d %H:%M:%S')}",
        f" git commit   : {_git_commit()}",
        "",
        "[ 学習条件 ]",
        f"  pretrained        : {a['pretrained']}",
        f"  seed              : {a['seed']}",
        f"  batch_size        : {a['batch_size']}",
        f"  max_epochs        : {a['max_epochs']}  (patience {a['patience']})",
        f"  lr                : {a['lr']}",
        f"  weight_decay      : {a['weight_decay']}",
        f"  freeze_backbone   : {a['freeze_backbone']}",
        f"  freeze_dilation   : {a['freeze_dilation']}",
        f"  augment           : {a.get('aug', False)}"
        + (
            f"  hflip={a.get('aug_hflip_prob')} rotate=±{a.get('aug_rotate')}"
            f" scale=±{a.get('aug_scale')} translate=±{a.get('aug_translate')}"
            f" gamma=±{a.get('aug_gamma')} bright=±{a.get('aug_brightness')}"
            f" noise={a.get('aug_noise')}"
            if a.get("aug")
            else ""
        ),
        f"  dw_mode           : {a.get('dw_mode', 'dense')}"
        + (
            f"  init_sigma={a.get('spectral_init_sigma')}"
            f" init_gamma={a.get('spectral_init_gamma')}"
            f" alpha={a.get('spectral_alpha')}"
            f" sigma_lr={a.get('spectral_sigma_lr')}"
            f" use_local={a.get('spectral_use_local_branch')}"
            f" pad={a.get('spectral_pad_factor')}"
            f" cropq={a.get('spectral_crop_quantile')}"
            f" max_sigma={a.get('spectral_max_sigma')}"
            + (
                f" num_gaussians={a.get('spectral_num_gaussians')}"
                if a.get("dw_mode") == "spectral_mix"
                else ""
            )
            if a.get("dw_mode") in ("spectral", "spectral_mix")
            else (
                f"  rank={a.get('separable_rank')}"
                if a.get("dw_mode") == "separable"
                else (
                    f"  order={a.get('gauss_deriv_order')}"
                    f" init_sigma={a.get('spectral_init_sigma')}"
                    f" max_sigma={a.get('spectral_max_sigma')}"
                    f" sigma_lr={a.get('spectral_sigma_lr')}"
                    if a.get("dw_mode") == "gauss_deriv"
                    else (
                        f"  base_sigma={a.get('spectral_init_sigma')}"
                        f" growth={a.get('gauss_pyramid_growth')}"
                        f" max_sigma={a.get('spectral_max_sigma')}"
                        f" freeze_scale={a.get('freeze_scale')}"
                        f" sigma_lr={a.get('spectral_sigma_lr')}"
                        if a.get("dw_mode") == "gauss_pyramid"
                        else ""
                    )
                )
            )
        ),
        f"  adaptive_dw       : {a.get('adaptive_dw', False)}"
        + (
            f"  dilations={a.get('adaptive_dilations')}" if a.get("adaptive_dw") else ""
        ),
        f"  spectral_dw       : {a.get('spectral_dw', False)}"
        + (
            f"  init_sigma={a.get('spectral_init_sigma')}"
            f" init_gamma={a.get('spectral_init_gamma')}"
            f" max_sigma={a.get('spectral_max_sigma')}"
            if a.get("spectral_dw")
            else ""
        ),
        "",
        "[ ERF 正則化 ]",
        f"  erf_reg_weight(λ) : {a['erf_reg_weight']}",
        f"  erf_target_spread : {a['erf_target_spread']}",
        "",
        "[ 結果 ]",
        f"  best_bone_dice    : {result['best_bone_dice']:.4f}  (epoch {result['best_epoch']})",
        f"  epochs_run        : {result['epochs_run']}",
        f"  early_stopped     : {result['early_stopped']}",
        f"  final_mean_dilation: {result['final_mean_dilation']}",
        f"  final_mean_spread  : {result['final_mean_spread']}",
        f"  final_per_stage_spread: {result['final_per_stage_spread']}",
        "=" * 50,
        "",
    ]
    with open(os.path.join(save_dir, "run_info.txt"), "w") as f:
        f.write("\n".join(lines))


def _now_tokyo():
    try:
        return datetime.datetime.now(ZoneInfo("Asia/Tokyo"))
    except Exception:
        return datetime.datetime.now(timezone(timedelta(hours=9)))


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=13, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1)
        # targets (B, H, W) を One-hot (B, C, H, W) に変換
        targets_one_hot = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        )
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        # 💡 背景(0)を除いた、骨の12部位(1~12)の平均DiceLossを返す
        return 1.0 - dice[1:].mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(num_classes=num_classes)

    def forward(self, outputs, targets):
        return self.ce(outputs, targets) + self.dice(outputs, targets)


def train_net(args=None):
    if args is None:
        args = build_parser().parse_args()

    # 0. 乱数シード固定 (対照実験で初期値・学習軌道を揃えるため)
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"🎲 seed = {args.seed}")

    # 1. すべてのCUDAデバイスを認識させる
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("experiments", exist_ok=True)

    # 🔥 タイムゾーンを強制的に「Asia/Tokyo（日本時間）」に指定して時刻を取得
    now_str = _now_tokyo().strftime("%Y%m%d_%H%M%S")
    if args.save_dir:
        save_dir = args.save_dir
    else:
        tag = args.tag.strip().replace(" ", "-").replace("/", "-")
        save_dir = f"experiments/run_{now_str}" + (f"_{tag}" if tag else "")
    os.makedirs(save_dir, exist_ok=True)
    write_run_config(save_dir, args)
    print(f"📁 今回の実験データは {save_dir} に保存されます (run_config.json 出力済み)")

    # データセットの読み込みと分割 (view で正面/背面を絞れる。分割前フィルタなので val も同ビュー)
    full_dataset = ScintiMultiClassDataset(data_dir=args.data_dir, view=args.view)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # データ拡張 (train のみ)。val split は seed42 固定=不変なので比較基盤と整合。
    # 既定オフ (未指定なら従来と厳密一致)。骨シンチは左右クラス(8/9 arm,10/11 leg)が
    # あるので水平反転は左右ラベルを入れ替える (augment_scinti 側で対応)。
    if args.aug:
        train_dataset = AugmentedScintiDataset(
            train_dataset,
            hflip_prob=args.aug_hflip_prob,
            rotate_deg=args.aug_rotate,
            scale=args.aug_scale,
            translate=args.aug_translate,
            gamma=args.aug_gamma,
            brightness=args.aug_brightness,
            noise=args.aug_noise,
            seed=args.seed,
        )
        print(
            f"🔀 データ拡張 ON | hflip(左右swap)={args.aug_hflip_prob} "
            f"rotate=±{args.aug_rotate}° scale=±{args.aug_scale} translate=±{args.aug_translate} "
            f"gamma=±{args.aug_gamma} bright=±{args.aug_brightness} noise={args.aug_noise}"
        )

    # 💡 4枚のGPUにデータを配るため、バッチサイズを4倍（16 × 4 = 64）に引き上げます！
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2. nn.DataParallel を使ってモデルをマルチGPU化
    if args.adaptive_dw:
        print(
            f"🧭 ContentAdaptiveDW 有効 | dilations={args.adaptive_dilations} "
            "(ConvMod a1/a2/a3 を画素ごと dilation ゲートに差し替え)"
        )
    if args.spectral_dw:
        print(
            f"🌊 SpectralGaussianDW 有効 | init_sigma={args.spectral_init_sigma} "
            f"init_gamma={args.spectral_init_gamma} max_sigma(stage別)={args.spectral_max_sigma} "
            "(周波数ガウスで RF 制御。σ は --erf-reg-weight/--erf-target-spread で駆動)"
        )
        if args.erf_reg_weight == 0 and args.spectral_sigma_warmup_epochs == 0:
            print(
                "  ⚠️ --erf-reg-weight=0 かつ σ-warmup 無しだと σ は init_sigma で固定。"
                "RF を育てたいなら --spectral-sigma-warmup-epochs で σ を漸増するか λ>0。"
            )
    if args.dw_mode == "separable":
        print(
            f"🔪 SeparableDWConv 有効 (機構A) | rank={args.separable_rank} | "
            f"RFA a1/a2/a3 を R 本の 1×K・K×1 分離の和 (タップ 2RK) に差替。"
            f"事前学習の密カーネルは rank-{args.separable_rank} SVD で分離初期化して転移。"
        )
    if args.dw_mode == "spectral":
        print(
            f"🌈 SpectralDW 有効 (機構B) | init_sigma={args.spectral_init_sigma} "
            f"init_gamma={args.spectral_init_gamma} alpha={args.spectral_alpha} "
            f"sigma_lr={args.spectral_sigma_lr} use_local={args.spectral_use_local_branch} "
            f"max_sigma(stage別)={args.spectral_max_sigma} "
            "(周波数+動的スペクトル切り出し。σ大で帯域を η=α/σ に切り詰め)"
        )
        if args.spectral_sigma_lr == 0:
            print(
                "  ⚠️ --spectral-sigma-lr=0: σ は base lr で凍結する見込み。"
                "σ を動かすなら 1e-2 程度を指定 (visualize_sigma.py で確認)。"
            )
    if args.dw_mode == "spectral_mix":
        print(
            f"🎛️ SpectralMixtureDW 有効 (機構C) | num_gaussians={args.spectral_num_gaussians} "
            f"init_sigma={args.spectral_init_sigma} init_gamma={args.spectral_init_gamma} "
            f"alpha={args.spectral_alpha} sigma_lr={args.spectral_sigma_lr} "
            f"use_local={args.spectral_use_local_branch} max_sigma(stage別)={args.spectral_max_sigma} "
            "(周波数包絡を K ガウス混合にし ERF を集約ガウス=AGD へ。分離rank-K+振幅考慮切り出し)"
        )
        if args.spectral_sigma_lr == 0:
            print(
                "  ⚠️ --spectral-sigma-lr=0: σ は base lr で凍結する見込み。"
                "σ を動かすなら 1e-2 程度を指定 (visualize_sigma.py で確認)。"
            )
    if args.dw_mode == "gauss_deriv":
        print(
            f"📐 GaussianDerivativeDW 有効 (本命) | order N={args.gauss_deriv_order} "
            f"(基底 M={args.gauss_deriv_order + 1}) | init_sigma={args.spectral_init_sigma} "
            f"max_sigma(stage別)={args.spectral_max_sigma} sigma_lr={args.spectral_sigma_lr} | "
            "カーネルを W=H(σ)·A·H(σ)ᵀ (ガウス微分基底) に構造的に閉じ込め、RF スケールを "
            "σ のみに縛る。local 枝/gamma なし = ガウス性がハード制約。"
        )
        if args.spectral_sigma_lr == 0:
            print(
                "  ⚠️ --spectral-sigma-lr=0: σ は base lr で凍結する見込み(RF は init 固定)。"
                "σ(RF)を学習させるなら 1e-2 程度を指定 (visualize_sigma.py で確認)。"
            )
    if args.dw_mode == "gauss_pyramid":
        print(
            f"🔭 GaussianPyramidDW 有効 | 多スケール純ガウス | init_sigma(base/stage別)="
            f"{args.spectral_init_sigma} growth={args.gauss_pyramid_growth} "
            f"(a1/a2/a3 σ=base·growth^0/1/2) max_sigma={args.spectral_max_sigma} "
            f"freeze_scale={args.freeze_scale} sigma_lr={args.spectral_sigma_lr} | "
            "純ガウス低域通過(local枝/gamma なし)。境界は pointwise の DoG で復元。"
            "カスケードのガウス半群で実効σ増大。"
        )
        if not args.freeze_scale and args.spectral_sigma_lr == 0:
            print(
                "  ⚠️ σ 学習可(freeze_scaleなし)だが --spectral-sigma-lr=0: σ は実質凍結。"
                "σ を動かすなら 1e-2 程度を指定、または --freeze-scale で明示的に固定。"
            )
    model = UniConvNet_UNet_13CH(
        num_classes=13,
        adaptive_dw=args.adaptive_dw,
        adaptive_dilations=tuple(args.adaptive_dilations),
        spectral_dw=args.spectral_dw,
        spectral_init_sigma=args.spectral_init_sigma,
        spectral_init_gamma=args.spectral_init_gamma,
        spectral_max_sigma=tuple(args.spectral_max_sigma),
        spectral_use_local_branch=args.spectral_use_local_branch,
        spectral_alpha=args.spectral_alpha,
        spectral_pad_factor=args.spectral_pad_factor,
        spectral_crop_quantile=args.spectral_crop_quantile,
        spectral_num_gaussians=args.spectral_num_gaussians,
        separable_rank=args.separable_rank,
        gauss_deriv_order=args.gauss_deriv_order,
        gauss_pyramid_growth=args.gauss_pyramid_growth,
        gauss_freeze_scale=args.freeze_scale,
        dw_mode=args.dw_mode,
    )
    # FFT を使う機構 (旧 spectral_gaussian / 新 spectral / 多ガウス混合 / gauss_pyramid) は
    # DataParallel で崩壊する既知バグ。gauss_deriv は空間 separable conv (FFT 非使用) なので DataParallel 可。
    uses_fft = args.spectral_dw or args.dw_mode in (
        "spectral",
        "spectral_mix",
        "gauss_pyramid",
    )
    if torch.cuda.device_count() > 1 and uses_fft:
        # 単一GPU に落とす。複数GPUを使いたい場合も CUDA_VISIBLE_DEVICES で1枚に絞ること。
        print(
            "⚠️ spectral 系(FFT) は DataParallel で崩壊するため単一GPU(cuda:0)で学習します。"
            "（複数GPUが見えていても DataParallel は使いません）"
        )
    elif torch.cuda.device_count() > 1:
        print(
            f"🚀 {torch.cuda.device_count()}基のGPUを束ねて、爆速並列学習を開始します！"
        )
        model = nn.DataParallel(model)
    else:
        print("🚀 シングルGPUモードで学習を開始します")
    model.to(device)

    load_pretrained_backbone(model, args.pretrained)
    if args.freeze_backbone:
        backbone = (
            model.module.backbone
            if isinstance(model, nn.DataParallel)
            else model.backbone
        )
        for param in backbone.parameters():
            param.requires_grad = False
        # 学習可能 dilation は固定しない (RF を ERF にフィットさせるため)
        for m in backbone.modules():
            if hasattr(m, "log_dilation"):
                m.log_dilation.requires_grad = True
        print("🔒 Backbone を固定して学習します (dilation は学習可能のまま)")

    # 対照実験用: dilation を学習させず初期値1に固定 (ERF拡大を起こさない)
    if args.freeze_dilation:
        m_ = model.module if isinstance(model, nn.DataParallel) else model
        n_frozen = 0
        for mod in m_.modules():
            if hasattr(mod, "log_dilation"):
                mod.log_dilation.requires_grad = False
                n_frozen += 1
        print(
            f"🧊 dilation を固定 (log_dilation {n_frozen} 個を学習対象外, dilation=1)"
        )

    if args.erf_reg_weight > 0:
        print(
            f"🎯 ERF 正則化 ON | lambda={args.erf_reg_weight} "
            f"target_spread={args.erf_target_spread}"
        )

    criterion = CombinedLoss(num_classes=13)
    optimizer = build_optimizer(model, args)

    # 🔥 アーリーストップ用の設定
    best_bone_dice = 0.0
    best_epoch = 0
    patience_counter = 0  # 改善しなかった連続回数をカウント
    max_epochs = args.max_epochs  # 最大エポック数
    early_stopped = False
    epochs_run = 0

    # σ-warmup: SpectralGaussianDW の実効σ上限を epoch ごとに start→target(init_sigma)
    # へ漸増し、warmup 後は max_sigma まで開放。大 init_sigma を崩壊させず使うための駆動。
    warmup_epochs = args.spectral_sigma_warmup_epochs
    spectral_mods = []
    if args.spectral_dw and warmup_epochs > 0:
        base = model.module if isinstance(model, nn.DataParallel) else model
        spectral_mods = [m for m in base.modules() if isinstance(m, SpectralGaussianDW)]
        print(
            f"🐌 σ-warmup ON | start={args.spectral_sigma_warmup_start} → "
            f"target(init_sigma) を {warmup_epochs}ep で到達後 max_sigma へ開放 "
            f"({len(spectral_mods)} 個の SpectralGaussianDW)"
        )

    last_diag = None
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        model.train()
        # σ-warmup: 当該 epoch の実効σ上限を各 spectral 枝に配信。
        if spectral_mods:
            ws = args.spectral_sigma_warmup_start
            for m in spectral_mods:
                if epoch <= warmup_epochs:
                    cap = ws + (m.target_sigma - ws) * (epoch / warmup_epochs)
                else:
                    cap = m.max_sigma
                m.set_sigma_cap(cap)
            if epoch <= warmup_epochs or epoch == warmup_epochs + 1:
                caps = [round(float(m.sigma_cap), 2) for m in spectral_mods[:4]]
                print(f"   σ-cap @ep{epoch}: {caps} ...")
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # ERF 正則化: 学習可能 dilation を目標 RF 広がりへ寄せる
            if args.erf_reg_weight > 0:
                reg, last_diag = erf_reg_loss(model, args.erf_target_spread)
                loss = loss + args.erf_reg_weight * reg.to(loss.device)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Validation (骨の12部位の平均Diceを算出)
        model.eval()
        val_dices = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)  # (B, H, W)

                for c in range(1, 13):  # クラス1〜12
                    pred_c = (preds == c).float()
                    mask_c = (masks == c).float()
                    inter = (pred_c * mask_c).sum(dim=(1, 2))
                    uni = pred_c.sum(dim=(1, 2)) + mask_c.sum(dim=(1, 2))
                    dice = (2.0 * inter + 1e-5) / (uni + 1e-5)
                    val_dices.extend(dice.cpu().numpy())

        mean_bone_dice = np.mean(val_dices)
        log_line = (
            f"Epoch {epoch:02d} | Train Loss: {train_loss/len(train_dataset):.4f} "
            f"| Val Mean Bone Dice: {mean_bone_dice:.4f}"
        )
        if last_diag is not None:
            log_line += (
                f" | mean_dilation: {last_diag['mean_dilation']:.3f} "
                f"mean_spread: {last_diag['mean_spread']:.3f}"
            )
        print(log_line)

        # 🔥 アーリーストップの判定判定ロジック
        if mean_bone_dice > best_bone_dice:
            best_bone_dice = mean_bone_dice
            best_epoch = epoch
            patience_counter = 0  # 最高精度が出たらカウンターをリセット！
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            save_path = os.path.join(save_dir, "best_uniconvnet_unet.pth")
            torch.save(state_dict, save_path)
            print(
                f"✨ 最高精度更新! モデルを保存しました (Bone Dice: {best_bone_dice:.4f}) -> {save_path}"
            )
        else:
            patience_counter += 1
            print(
                f"⚠️ 精度向上ならず... (我慢カウンター: {patience_counter}/{args.patience})"
            )

        # 限界まで我慢したらループを抜ける（アーリーストップ発動）
        if patience_counter >= args.patience:
            print(
                f"\n🛑 アーリーストップ発動！ {args.patience}エポック連続で精度向上が見られなかったため、過学習を防ぐために学習を終了します。"
            )
            print(f"🏆 最終的な最高 Bone Dice: {best_bone_dice:.4f}")
            early_stopped = True
            break

    # 実験条件＋結果を run_info.txt に記録
    result = {
        "best_bone_dice": best_bone_dice,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "final_mean_dilation": (
            round(last_diag["mean_dilation"], 4) if last_diag else None
        ),
        "final_mean_spread": round(last_diag["mean_spread"], 4) if last_diag else None,
        "final_per_stage_spread": last_diag["per_stage_spread"] if last_diag else None,
    }
    write_run_info(save_dir, args, result)
    print(
        f"📝 実験条件と結果を {os.path.join(save_dir, 'run_info.txt')} に保存しました"
    )

    # 完了通知(ベストエフォート)。tag/機構/best dice/保存先を Discord などへ。
    tag = getattr(args, "tag", "") or "(no tag)"
    body = (
        f"tag: {tag} | dw_mode: {getattr(args, 'dw_mode', 'dense')}\n"
        f"best_bone_dice: {result['best_bone_dice']:.4f} (epoch {result['best_epoch']})\n"
        f"epochs_run: {result['epochs_run']} | early_stopped: {result['early_stopped']}\n"
        f"dir: {os.path.basename(save_dir)}"
    )
    _run_notify("ok", f"学習完了: {tag}", body)


if __name__ == "__main__":
    try:
        train_net()
    except Exception:
        # 例外で落ちた場合も通知(ラッパ経由なら NOTIFY_WRAPPED でラッパが担当)。
        import traceback

        tb = traceback.format_exc()
        _run_notify(
            "fail",
            "学習が例外で停止",
            f"argv: {' '.join(sys.argv[1:])}\n\n```\n{tb[-1500:]}\n```",
        )
        raise
