# coding:utf-8
"""
compare_erf — 複数モデルの層別 ERF を「同一図に重ねて」見比べる。

visualize_erf.py の 6 面図は 1 モデル 1 層ずつなので、baseline / pure_spec /
learned_sigma のように AGD の崩れ方を比較するには不向き。本スクリプトは各層について
  * 上段: ERF 中心断面 (peak 正規化) を全モデル重ね描き
  * 下段: その FFT 断面 (DC 正規化) を全モデル重ね描き
を 1 枚にまとめ、凡例に σ_moment と excess kurtosis を出す。ガウス(滑らか)か
尖り+ハロー(AGD 崩壊)かが一目で分かる。

使い方:
  python3 src/compare_erf.py \
    --weights <baseline.pth> <pure_spec.pth> <learned_sigma.pth> \
    --labels baseline pure_spec learned_sigma \
    --part encoder --input-size 512 --n-samples 30 \
    --out-dir /workspace/erf_results/compare
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
from ckpt_utils import build_model_from_checkpoint
from erf_sigma_table import sigma_and_kurtosis
from visualize_erf import compute_erf, get_target_modules


def erf_sections(erf):
    """ERF(2D) から (x_px, erf行 peak正規化, freq, |FFT行| DC正規化) を返す。"""
    eps = 1e-12
    H, W = erf.shape
    cy = H // 2
    row = erf[cy, :].astype(np.float64)
    row = row / (row.max() + eps)
    amp = np.abs(np.fft.rfft(erf[cy, :].astype(np.float64)))
    amp = amp / (amp[0] + eps)
    freq = np.fft.rfftfreq(W)
    return np.arange(W), row, freq, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--part", choices=["encoder", "decoder", "all"], default="all")
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument(
        "--zoom",
        type=int,
        default=160,
        help="ERF 断面の中心±zoom px だけ拡大表示 (0で全幅)",
    )
    ap.add_argument("--out-dir", default="/workspace/erf_results/compare")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")
    if args.labels and len(args.labels) == len(args.weights):
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.dirname(w)) for w in args.weights]

    os.makedirs(args.out_dir, exist_ok=True)

    # モデルを構築 (dw_mode 自動判別) & 各層の対象モジュールを取得
    models, targets_per = [], []
    for w in args.weights:
        m = build_model_from_checkpoint(
            w, num_classes=13, device=device, eval_mode=False
        )
        models.append(m)
        targets_per.append(dict(get_target_modules(m, args.part)))

    layer_names = list(targets_per[0].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))

    for layer in layer_names:
        print(f"\n[{layer}] 各モデルの ERF を計算中...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for i, (lab, m) in enumerate(zip(labels, models)):
            erf = compute_erf(
                m, targets_per[i][layer], args.input_size, args.n_samples, device
            )
            _, sm, ku = sigma_and_kurtosis(erf)
            x, row, freq, amp = erf_sections(erf)
            leg = f"{lab}  σ={sm:.1f} kurt={ku:+.1f}"
            axes[0].plot(x, row, color=colors[i], lw=1.4, label=leg)
            axes[1].plot(freq, amp, color=colors[i], lw=1.4, label=leg)

        cx = args.input_size // 2
        axes[0].set_title(f"{layer} — ERF cross-section (peak-normalized)")
        axes[0].set_xlabel("pixel")
        axes[0].set_ylabel("normalized gradient")
        if args.zoom > 0:
            axes[0].set_xlim(cx - args.zoom, cx + args.zoom)
        axes[0].axvline(cx, color="gray", lw=0.6, ls="--")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)

        axes[1].set_title(f"{layer} — FFT of ERF (DC-normalized)")
        axes[1].set_xlabel("cycles/pixel")
        axes[1].set_ylabel("normalized amplitude")
        axes[1].set_xlim(-0.02, 0.5)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

        plt.suptitle(
            f"ERF AGD comparison — {layer}  (smooth Gaussian = AGD held; "
            "spike+halo / lobes = AGD broken)",
            fontsize=12,
        )
        plt.tight_layout()
        p = os.path.join(args.out_dir, f"erf_compare_{layer}.png")
        plt.savefig(p, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  保存: {p}")

    print(f"\n完了。結果フォルダ: {args.out_dir}")


if __name__ == "__main__":
    main()
