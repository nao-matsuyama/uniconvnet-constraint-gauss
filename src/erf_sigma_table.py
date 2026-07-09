# coding:utf-8
"""
複数モデル × 全8層の ERF σ を一括算出して比較表(CSV)・グラフにする。

各層 (encoder_stage0..3 + decoder_up3/up2/up1/up0) について
visualize_erf.py と同じ手法で ERF を計算し:
  sigma    … ERF 断面の上側包絡線にフィットしたガウスの σ (px)  ← 受容野の広さ
  kurtosis … ERF 断面の過剰尖度 (ガウス=0, 芯が尖る=正)        ← AGD 崩れの指標
を出す。λ=0 / 0.01 / 0.1 などモデル間の RF 拡大と AGD 維持を一目で比較できる。

使い方:
  python3 src/erf_sigma_table.py \
    --weights /workspace/experiments/run_baseline/best_uniconvnet_unet.pth \
              /workspace/experiments/run_erf0.01/best_uniconvnet_unet.pth \
              /workspace/experiments/run_erf0.1/best_uniconvnet_unet.pth \
    --labels baseline erf0.01 erf0.1 \
    --input-size 512 --n-samples 50 \
    --out-dir /workspace/erf_results/sigma_compare
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))
from ckpt_utils import build_model_from_checkpoint
from run_meta import write_manifest
from visualize_erf import (
    _fit_gaussian,
    _upper_envelope,
    compute_erf,
    get_target_modules,
)

LAYER_ORDER = [
    "encoder_stage0",
    "encoder_stage1",
    "encoder_stage2",
    "encoder_stage3",
    "decoder_up3",
    "decoder_up2",
    "decoder_up1",
    "decoder_up0",
]


def sigma_and_kurtosis(erf):
    """
    ERF(2D) 中心行から3つの量を返す:
      sigma_fit    … 上側包絡線へのガウスフィット σ (gridding に弱く不安定)
      sigma_moment … ERF を1D密度とみなした2次モーメント σ = sqrt(var) (頑健)
      kurt         … 過剰尖度 (Gaussian=0, 芯が尖る=正)
    """
    eps = 1e-10
    cy = erf.shape[0] // 2
    row = erf[cy, :].astype(np.float64)
    row = row / (row.max() + eps)

    env = _upper_envelope(row)
    gp = _fit_gaussian(env)
    sigma_fit = float(abs(gp[2])) if gp is not None else float("nan")

    # ERF 断面を 1D 密度とみなした 2次モーメント σ と過剰尖度
    x = np.arange(len(row), dtype=np.float64)
    w = np.clip(row, 0, None)
    w = w / (w.sum() + eps)
    mu = (w * x).sum()
    var = (w * (x - mu) ** 2).sum()
    sigma_moment = float(np.sqrt(var))
    kurt = (w * (x - mu) ** 4).sum() / (var**2 + eps) - 3.0
    return sigma_fit, sigma_moment, float(kurt)


def analyze_model(weights, input_size, n_samples, device):
    # spectral/adaptive チェックポイントも自動判別して構築（compute_erf が eval 化）。
    model = build_model_from_checkpoint(weights, num_classes=13, device=device)

    targets = get_target_modules(model, "all")
    out = {}
    for label, module in targets:
        erf = compute_erf(model, module, input_size, n_samples, device)
        sig_fit, sig_mom, kurt = sigma_and_kurtosis(erf)
        out[label] = (sig_fit, sig_mom, kurt)
        print(
            f"    {label:18s} σ_fit={sig_fit:6.2f}  σ_moment={sig_mom:6.2f}  kurt={kurt:+.2f}"
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--out-dir", default="/workspace/erf_results/sigma_compare")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")

    if args.labels and len(args.labels) == len(args.weights):
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.dirname(w)) for w in args.weights]

    os.makedirs(args.out_dir, exist_ok=True)
    write_manifest(
        args.out_dir,
        "erf_sigma_table",
        {"input_size": args.input_size, "n_samples": args.n_samples},
        models=dict(zip(labels, args.weights)),
    )

    results = {}  # label -> {layer: (sigma_fit, sigma_moment, kurt)}
    for l, w in zip(labels, args.weights):
        print(f"\n[{l}] {w}")
        results[l] = analyze_model(w, args.input_size, args.n_samples, device)

    # ── CSV (σ_fit, σ_moment, kurtosis) ──
    csv_path = os.path.join(args.out_dir, "erf_sigma_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["layer"]
            + [f"{l}_sigma_fit" for l in labels]
            + [f"{l}_sigma_moment" for l in labels]
            + [f"{l}_kurtosis" for l in labels]
        )
        for layer in LAYER_ORDER:
            row = [layer]
            row += [f"{results[l][layer][0]:.3f}" for l in labels]
            row += [f"{results[l][layer][1]:.3f}" for l in labels]
            row += [f"{results[l][layer][2]:.3f}" for l in labels]
            w.writerow(row)
    print(f"\n保存: {csv_path}")

    # ── コンソール表 (σ_moment = 頑健版) ──
    print("\n" + "=" * (20 + 12 * len(labels)))
    print(" ERF σ_moment (px, 2次モーメント=頑健) 比較")
    print("=" * (20 + 12 * len(labels)))
    header = f"{'layer':<20}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    print("-" * len(header))
    for layer in LAYER_ORDER:
        line = f"{layer:<20}"
        for l in labels:
            line += f"{results[l][layer][1]:>12.2f}"
        print(line)

    # ── グラフ: (左) σ_moment 折れ線 / (右) 過剰尖度(AGD崩れ) 折れ線 ──
    x = np.arange(len(LAYER_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for l in labels:
        sig = [results[l][layer][1] for layer in LAYER_ORDER]  # σ_moment
        axes[0].plot(x, sig, marker="o", label=l)
    axes[0].axvline(3.5, color="gray", ls="--", lw=0.8)  # encoder|decoder 境界
    axes[0].text(1.5, axes[0].get_ylim()[1] * 0.95, "encoder", ha="center", fontsize=9)
    axes[0].text(5.5, axes[0].get_ylim()[1] * 0.95, "decoder", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(LAYER_ORDER, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("ERF σ_moment (px)")
    axes[0].set_title("Effective Receptive Field size (2nd-moment σ) per layer")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for l in labels:
        kt = [results[l][layer][2] for layer in LAYER_ORDER]  # kurtosis
        axes[1].plot(x, kt, marker="s", label=l)
    axes[1].axhline(0, color="green", ls="--", lw=1, label="Gaussian (kurt=0)")
    axes[1].axvline(3.5, color="gray", ls="--", lw=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LAYER_ORDER, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("excess kurtosis")
    axes[1].set_title("AGD deviation (excess kurtosis; 0 = Gaussian, + = peaked)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ERF σ & AGD comparison across models", fontsize=13)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "erf_sigma_compare.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {fig_path}")
    print(f"\n完了。結果フォルダ: {args.out_dir}")


if __name__ == "__main__":
    main()
