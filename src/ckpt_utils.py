# coding:utf-8
"""
チェックポイントの RF 機構を自動判別してモデルを構築する共有ヘルパ。

UniConvNet_UNet_13CH は ConvMod の depthwise を複数通りに切替できる:
  - 通常 (DilatedDWConv)
  - ContentAdaptiveDW   … state_dict に gate.2.weight を持つ
  - SpectralGaussianDW  … state_dict に log_sigma(1D) を持つ
  - SpectralMixtureDW   … state_dict に log_sigma(2D=(C,K)) を持つ(機構C 多ガウス混合)
評価系スクリプト（evaluate / compare_models / erf_sigma_table / benchmark）はどの
チェックポイントでも正しく load できるよう、ここで判別とモデル構築を一元化する。
"""

import json
import os

import torch

from model_uniconvnet_unet import UniConvNet_UNet_13CH


def read_run_arg(weights, key, default=None):
    """checkpoint と同じフォルダの run_config.json から学習時の引数を読む。

    use_local_branch のように state_dict に残らない構築フラグを評価時に再現するため。
    run_config.json が無い/キーが無い旧 run では default を返す。
    """
    cfg = os.path.join(os.path.dirname(os.path.abspath(weights)), "run_config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                args = json.load(f).get("args", {})
            if key in args:
                return args[key]
        except Exception:
            pass
    return default


def inspect_checkpoint(state_dict, adaptive_dilations=(1, 2, 4, 8)):
    """state_dict から (adaptive, dilations, spectral) を返す。

    - spectral: log_sigma を持てば True（SpectralGaussianDW）。
    - adaptive: gate.2.weight を持てば True（ContentAdaptiveDW）。枝数は出力 ch 数で、
      adaptive_dilations と長さが合えばそれを、合わなければ 2**i を採用。
    """
    spectral = any(k.endswith("log_sigma") for k in state_dict)
    gate_keys = [k for k in state_dict if k.endswith("gate.2.weight")]
    if not gate_keys:
        return False, tuple(adaptive_dilations), spectral
    n = state_dict[gate_keys[0]].shape[0]
    if n == len(adaptive_dilations):
        return True, tuple(adaptive_dilations), spectral
    return True, tuple(2**i for i in range(n)), spectral


def build_model_from_checkpoint(
    weights,
    num_classes=13,
    device="cpu",
    adaptive_dilations=(1, 2, 4, 8),
    eval_mode=True,
    verbose=True,
):
    """重みファイルを読み、機構を自動判別して構築・load 済みのモデルを返す。

    優先順位:
      1. run_config.json の dw_mode (separable / spectral / dense) — 新しい統一セレクタ。
      2. state_dict の weight_h キー → separable (run_config 無い場合のフォールバック)。
      3. log_sigma → SpectralGaussianDW (旧 spectral)、gate.2.weight → ContentAdaptiveDW。
    dw_mode=spectral(機構B SpectralDW) と旧 spectral_gaussian は共に log_sigma を持つため、
    両者の区別には run_config.json の dw_mode が必須 (無ければ旧 spectral_gaussian と解釈)。
    """
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    dw_mode = read_run_arg(weights, "dw_mode", None)
    has_weight_h = any(k.endswith("weight_h") for k in sd)
    if dw_mode is None and has_weight_h:
        dw_mode = "separable"

    # use_local_branch は state_dict に残らない構築フラグ。学習時と評価時で食い違うと
    # forward 経路が変わり崩壊して見える(pure-spec 学習を local+gamma で評価する等)。
    use_local = bool(read_run_arg(weights, "spectral_use_local_branch", True))
    spectral_alpha = float(read_run_arg(weights, "spectral_alpha", 2.0))
    pad_factor = float(read_run_arg(weights, "spectral_pad_factor", 0.0))
    crop_quantile = float(read_run_arg(weights, "spectral_crop_quantile", 0.0))

    # dw_mode 未記録でも log_sigma が 2 次元 (C,K) なら多ガウス混合 (機構C) と判定。
    log_sigma_keys = [k for k in sd if k.endswith("log_sigma")]
    is_mixture = bool(log_sigma_keys) and sd[log_sigma_keys[0]].dim() == 2
    if dw_mode is None and is_mixture:
        dw_mode = "spectral_mix"

    # ガウス微分基底は coeff(C,M,M) という固有キーを持つ(log_sigma は 1D)。
    coeff_keys = [k for k in sd if k.endswith("coeff") and sd[k].dim() == 3]
    if dw_mode is None and coeff_keys:
        dw_mode = "gauss_deriv"

    if dw_mode == "gauss_deriv":
        # order N は coeff 形状 (C,M,M) の M-1 から復元 (run_config を優先)。
        order_from_sd = (sd[coeff_keys[0]].shape[1] - 1) if coeff_keys else 2
        gauss_deriv_order = int(read_run_arg(weights, "gauss_deriv_order", order_from_sd))
        # max_sigma はカーネルグリッド K と σ clamp を決めるので学習時と一致させる必要がある。
        max_sigma = read_run_arg(weights, "spectral_max_sigma", (32.0, 24.0, 12.0, 6.0))
        init_sigma = read_run_arg(weights, "spectral_init_sigma", 1.0)
        if verbose:
            print(
                f"    (dw_mode=gauss_deriv, order={gauss_deriv_order}, "
                f"max_sigma={max_sigma} を検出)"
            )
        model = UniConvNet_UNet_13CH(
            num_classes=num_classes,
            dw_mode="gauss_deriv",
            gauss_deriv_order=gauss_deriv_order,
            spectral_max_sigma=tuple(max_sigma)
            if isinstance(max_sigma, (list, tuple))
            else max_sigma,
            spectral_init_sigma=tuple(init_sigma)
            if isinstance(init_sigma, (list, tuple))
            else init_sigma,
        )
        model.load_state_dict(sd)
        model.to(device)
        return model.eval() if eval_mode else model

    if dw_mode in ("separable", "spectral", "spectral_mix"):
        # 多ガウス混合は K を state_dict の log_sigma 形状 (C,K) から復元 (run_config を優先)。
        num_gaussians = int(
            read_run_arg(
                weights,
                "spectral_num_gaussians",
                sd[log_sigma_keys[0]].shape[1] if is_mixture else 3,
            )
        )
        # 機構A separable の rank R は weight_h の形状 (C,R,1,K) から復元 (run_config を優先)。
        weight_h_keys = [k for k in sd if k.endswith("weight_h")]
        rank_from_sd = sd[weight_h_keys[0]].shape[1] if weight_h_keys else 1
        separable_rank = int(read_run_arg(weights, "separable_rank", rank_from_sd))
        if verbose:
            extra = (
                f", num_gaussians={num_gaussians}"
                if dw_mode == "spectral_mix"
                else (f", rank={separable_rank}" if dw_mode == "separable" else "")
            )
            print(f"    (dw_mode={dw_mode}{extra} をrun_config/state_dictから検出)")
        model = UniConvNet_UNet_13CH(
            num_classes=num_classes,
            dw_mode=dw_mode,
            spectral_use_local_branch=use_local,
            spectral_alpha=spectral_alpha,
            spectral_pad_factor=pad_factor,
            spectral_crop_quantile=crop_quantile,
            spectral_num_gaussians=num_gaussians,
            separable_rank=separable_rank,
        )
        model.load_state_dict(sd)
        model.to(device)
        return model.eval() if eval_mode else model

    # ── レガシー経路 (dense / spectral_gaussian / adaptive) ──
    adaptive, dilations, spectral = inspect_checkpoint(sd, adaptive_dilations)
    if verbose:
        if spectral:
            print(
                f"    (SpectralGaussianDW 検出: 周波数ガウス, use_local_branch={use_local})"
            )
        if adaptive:
            print(f"    (ContentAdaptiveDW 検出: dilations={list(dilations)})")
    model = UniConvNet_UNet_13CH(
        num_classes=num_classes,
        adaptive_dw=adaptive,
        adaptive_dilations=dilations,
        spectral_dw=spectral,
        spectral_use_local_branch=use_local if spectral else True,
    )
    model.load_state_dict(sd)
    model.to(device)
    return model.eval() if eval_mode else model
