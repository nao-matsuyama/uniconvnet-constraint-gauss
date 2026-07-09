# coding:utf-8
"""
計算コストのベンチマーク — 「ロバスト性 × 低コスト」両立を数値で示す。

各モデルについて:
  params          … 総パラメータ数
  FLOPs           … 1枚あたり (model_stats と同じ手法、DilatedDWConv 含む)
  throughput      … 実測スループット (img/sec) と latency (ms/img)
  RFA dilated MACs … RFAの depthwise conv の実 MACs (dilatedで一定)
  RFA dense-equiv  … 同じ受容野(=dilation*(k-1)+1)を「密カーネル」で得た場合の MACs
  saving ratio     … dense-equiv / dilated (大きいほど dilation の省コスト効果が大)

要点: dilation で受容野を広げてもタップ数 k² は不変なので FLOPs はほぼ一定。
      同じ受容野を密な大カーネルで作ると面積比で FLOPs が増える → その差が省コスト。

使い方:
  python3 src/benchmark.py \
    --weights /workspace/experiments/run_baseline/best_uniconvnet_unet.pth \
              /workspace/experiments/run_erf0.1/best_uniconvnet_unet.pth \
    --labels baseline erf0.1 \
    --input-size 256 --batch-size 8 \
    --out-dir /workspace/bench_results/20260628
"""

import argparse
import csv
import math
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(__file__))
from ckpt_utils import build_model_from_checkpoint
from model_stats import count_flops, count_parameters, human_readable
from model_uniconvnet_unet import UniConvNet_UNet_13CH
from models.content_adaptive_dw import ContentAdaptiveDW
from models.dilated_dw import DilatedDWConv
from models.separable_dw import SeparableDWConv
from models.spectral_dw import SpectralDW
from run_meta import write_manifest

try:
    from models.gaussian_derivative_dw import GaussianDerivativeDW
except ImportError:
    GaussianDerivativeDW = None


def measure_throughput(model, input_size, batch_size, device, warmup, iters):
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    sec = t1 - t0
    imgs = iters * batch_size
    return imgs / sec, sec / imgs * 1000.0  # img/sec, ms/img


def rfa_macs(model, input_size, device):
    """
    RFA(DilatedDWConv) の depthwise conv について
      dilated_macs    : 実際の MACs (現dilation, 推論は1本/学習は2本相当だが上限2本で計上)
      dense_equiv_macs: 同じ受容野 k_eff=dilation*(k-1)+1 を密カーネルで得た場合の MACs
    を返す (MACs)。
    """
    recs = []
    hooks = []

    def mk():
        def hook(mod, inp, out):
            b, c, h, w = out.shape
            recs.append(
                (mod.kernel_size, float(mod.current_dilation().detach()), c, h, w)
            )

        return hook

    def mk_adaptive():
        def hook(mod, inp, out):
            b, c, h, w = out.shape
            # ContentAdaptiveDW は全 dilation 枝を計算するので分岐数だけ MACs が増える。
            # 等価受容野は最大 dilation で見積もる。
            recs.append(
                (
                    "adaptive",
                    mod.kernel_size,
                    max(mod.dilations),
                    c,
                    h,
                    w,
                    len(mod.dilations),
                )
            )

        return hook

    def mk_separable():
        def hook(mod, inp, out):
            b, c, h, w = out.shape
            recs.append(
                ("separable", mod.kernel_size, c, h, w, getattr(mod, "rank", 1))
            )

        return hook

    def mk_spectral():
        def hook(mod, inp, out):
            b, c, h, w = out.shape
            wf = w // 2 + 1
            hp, wp = mod._crop_sizes(h, wf)
            sig = float(mod.current_sigma().mean().detach())
            recs.append(("spectral", mod.kernel_size, c, h, w, hp, wp, sig))

        return hook

    def mk_gauss_deriv():
        def hook(mod, inp, out):
            b, c, h, w = out.shape
            sig = float(mod.current_sigma().mean().detach())
            recs.append(
                ("gauss_deriv", mod.kernel_size, c, h, w, mod.num_basis, sig)
            )

        return hook

    for m in model.modules():
        if isinstance(m, DilatedDWConv):
            hooks.append(m.register_forward_hook(mk()))
        elif isinstance(m, ContentAdaptiveDW):
            hooks.append(m.register_forward_hook(mk_adaptive()))
        elif isinstance(m, SeparableDWConv):
            hooks.append(m.register_forward_hook(mk_separable()))
        elif GaussianDerivativeDW is not None and isinstance(m, GaussianDerivativeDW):
            hooks.append(m.register_forward_hook(mk_gauss_deriv()))
        elif isinstance(m, SpectralDW):
            hooks.append(m.register_forward_hook(mk_spectral()))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, input_size, input_size, device=device))
    for h in hooks:
        h.remove()

    actual = 0  # 機構の実 MACs (spectral は FFT+切り出し乗算の概算)
    dense_equiv = 0  # 同じ受容野を密カーネルで得た場合の MACs
    for rec in recs:
        kind = rec[0] if isinstance(rec[0], str) else "dilated"
        if kind == "dilated":
            k, d, c, h, w = rec
            actual += c * h * w * (k * k)
            k_eff = int(round(d * (k - 1) + 1))
            dense_equiv += c * h * w * (k_eff * k_eff)
        elif kind == "adaptive":
            _, k, d, c, h, w, branches = rec
            actual += c * h * w * (k * k) * branches
            k_eff = int(round(d * (k - 1) + 1))
            dense_equiv += c * h * w * (k_eff * k_eff)
        elif kind == "separable":
            _, k, c, h, w, r = rec
            actual += c * h * w * (2 * r * k)  # R 本の (1×K + K×1) = 2RK タップ
            dense_equiv += c * h * w * (k * k)  # 密 K×K
        elif kind == "spectral":
            _, k, c, h, w, hp, wp, sig = rec
            logn = max(1.0, math.log2(max(2, h * w)))
            fft = int(2 * 5.0 * c * h * w * logn)  # rfft2 + irfft2 概算 (MACs 相当)
            mult = int(c * hp * wp)  # 切り出し帯域の複素乗算
            actual += fft + mult
            k_eff = int(round(6 * sig) + 1)  # σ の ±3σ を張る密カーネル相当 RF
            dense_equiv += c * h * w * (k_eff * k_eff)
        elif kind == "gauss_deriv":
            _, k, c, h, w, m_basis, sig = rec
            # M 本の (水平 1×K + 垂直 K×1) = 2MK タップの分離 depthwise。
            actual += c * h * w * (2 * m_basis * k)
            dense_equiv += c * h * w * (k * k)  # 同じ K×K を密で得た場合
    return actual, dense_equiv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--adaptive-dilations",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="adaptive チェックポイントの dilation 枝 (コスト見積り用)",
    )
    parser.add_argument("--out-dir", default="/workspace/bench_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")

    if args.labels and len(args.labels) == len(args.weights):
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.dirname(w)) for w in args.weights]

    write_manifest(
        args.out_dir,
        "benchmark",
        {
            "input_size": args.input_size,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "device": device,
        },
        models=dict(zip(labels, args.weights)),
    )

    rows = []
    for l, w in zip(labels, args.weights):
        print(f"\n[{l}] {w}")
        # dw_mode(separable/spectral) を含む全機構を run_config/state_dict から自動判別。
        model = build_model_from_checkpoint(
            w,
            num_classes=13,
            device=device,
            eval_mode=True,
            adaptive_dilations=tuple(args.adaptive_dilations),
        )

        total, trainable = count_parameters(model)
        flops = count_flops(
            model, torch.zeros(1, 3, args.input_size, args.input_size, device=device)
        )
        ips, mspi = measure_throughput(
            model, args.input_size, args.batch_size, device, args.warmup, args.iters
        )
        dil, dense = rfa_macs(model, args.input_size, device)
        ratio = dense / max(dil, 1)

        rows.append(
            {
                "label": l,
                "params": total,
                "flops": flops,
                "img_per_sec": ips,
                "ms_per_img": mspi,
                "rfa_dilated_macs": dil,
                "rfa_dense_equiv_macs": dense,
                "saving_ratio": ratio,
            }
        )
        print(
            f"    params={human_readable(total)}  FLOPs={human_readable(flops)}  "
            f"{ips:.1f} img/s  {mspi:.2f} ms/img  RFA省コスト x{ratio:.2f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "label",
                "params",
                "FLOPs_1img",
                "img_per_sec",
                "ms_per_img",
                "rfa_dilated_macs",
                "rfa_dense_equiv_macs",
                "saving_ratio",
            ]
        )
        for r in rows:
            wr.writerow(
                [
                    r["label"],
                    r["params"],
                    r["flops"],
                    f"{r['img_per_sec']:.2f}",
                    f"{r['ms_per_img']:.3f}",
                    r["rfa_dilated_macs"],
                    r["rfa_dense_equiv_macs"],
                    f"{r['saving_ratio']:.3f}",
                ]
            )
    print(f"\n保存: {csv_path}")

    # コンソール表
    print("\n" + "=" * 92)
    print(
        f"{'model':<16}{'params':>10}{'FLOPs/img':>12}{'img/s':>10}{'ms/img':>10}"
        f"{'RFA dense/dilated':>20}"
    )
    print("-" * 92)
    for r in rows:
        print(
            f"{r['label']:<16}{human_readable(r['params']):>10}"
            f"{human_readable(r['flops']):>12}{r['img_per_sec']:>10.1f}"
            f"{r['ms_per_img']:>10.2f}{('x%.2f' % r['saving_ratio']):>20}"
        )
    print("=" * 92)
    print("\n※ RFA dense/dilated = 同じ受容野を密カーネルで作った場合の MACs 倍率。")
    print("   x>1 は dilation により RFA convを省コスト化できている度合い。")
    print(f"\n完了。結果フォルダ: {args.out_dir}")


if __name__ == "__main__":
    main()
