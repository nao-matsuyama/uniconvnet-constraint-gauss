# coding:utf-8
"""
erf_gmm_fit — ERF 断面を「N ガウスの和」でフィットし、単一ガウスと比較する。

RFA は a1/a2/a3 = k7/9/11 の 3 スケールを集約する。ERF が単一ガウスでは甘くても
**N(既定3)ガウスの和で綺麗に乗る**なら、ERF は「集約ガウス(aggregated Gaussian)」=
設計通りの AGD であり、単一ガウスの尖度は物差し違いの人工物、と結論できる。

各 (モデル×層) について ERF 中心断面 (peak 正規化, 対称・中心固定) を
  * 単一ガウス   : A·exp(-(x-c)²/2σ²)
  * N ガウス和   : Σ_i A_i·exp(-(x-c)²/2σ_i²)
でフィットし、σ_i と R² を比較。図に データ / 単一fit / N和fit / 各成分 を重ねる。

使い方:
  python3 src/erf_gmm_fit.py \
    --weights $BASE $SPEC $LEARNED --labels baseline pure_spec learned_sigma \
    --part encoder --n-components 3 --input-size 512 --n-samples 30 \
    --out-dir /workspace/erf_results/gmm
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

sys.path.append(os.path.dirname(__file__))
from ckpt_utils import build_model_from_checkpoint
from visualize_erf import compute_erf, get_target_modules


def _mix(x, c, params, cusp=False):
    """params=[A1,s1,A2,s2,...] の N 成分和 (中心 c 固定)。

    cusp=True のとき **第1成分だけラプラシアン** exp(-|x-c|/b)(尖った芯)、
    残りはガウス。cusp=False は全てガウス。
    """
    y = np.zeros_like(x, dtype=np.float64)
    for i in range(0, len(params), 2):
        A, s = params[i], params[i + 1]
        if cusp and i == 0:
            y = y + A * np.exp(-np.abs(x - c) / s)  # ラプラシアン(指数カスプ)
        else:
            y = y + A * np.exp(-((x - c) ** 2) / (2.0 * s * s))
    return y


def _fit_n(x, row, c, n, sm, cusp=False):
    """N 成分和をフィットして (params, R²) を返す。sm=σ_moment を init に使う。"""
    # init: 第1成分は細く(芯/カスプ ~1.5px)、残りは σ_moment から広くばらす。
    if n == 1:
        s0 = [max(2.0, sm)]
    else:
        s0 = [1.5] + list(np.linspace(max(sm, 3.0), 4.0 * max(sm, 3.0), n - 1))
    A0 = [0.6] + [0.4 / max(n - 1, 1)] * (n - 1)
    p0, lo, hi = [], [], []
    for A, s in zip(A0, s0):
        p0 += [A, s]
        lo += [0.0, 0.5]
        hi += [1.5, len(x)]

    def f(xx, *p):
        return _mix(xx, c, p, cusp=cusp)

    try:
        popt, _ = curve_fit(f, x, row, p0=p0, bounds=(lo, hi), maxfev=30000)
    except Exception:
        return None, float("nan")
    fit = _mix(x, c, popt, cusp=cusp)
    ss_res = np.sum((row - fit) ** 2)
    ss_tot = np.sum((row - row.mean()) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return popt, r2


def analyze(erf, n, cusp=False):
    eps = 1e-12
    H, W = erf.shape
    c = W // 2
    row = np.clip(erf[H // 2, :].astype(np.float64), 0, None)
    row = row / (row.max() + eps)
    x = np.arange(W, dtype=np.float64)
    # 参照 σ_moment
    w = row / (row.sum() + eps)
    sm = float(np.sqrt(((x - (w * x).sum()) ** 2 * w).sum()))
    p1, r1 = _fit_n(x, row, c, 1, sm)  # 単一ガウス (基準)
    pn, rn = _fit_n(x, row, c, n, sm, cusp=cusp)  # N成分和 (cusp なら芯だけ指数)
    return x, row, c, sm, p1, r1, pn, rn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--part", choices=["encoder", "decoder", "all"], default="all")
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument(
        "--cusp",
        action="store_true",
        help="N成分和の第1成分をラプラシアン(指数 exp(-|x|/b))にする。"
        "ERF中心の尖ったカスプ(残差/skip由来の局在)がガウスで乗らない仮説を検証。",
    )
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--zoom", type=int, default=120)
    ap.add_argument("--out-dir", default="/workspace/erf_results/gmm")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}  N成分={args.n_components}")
    if args.labels and len(args.labels) == len(args.weights):
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.dirname(w)) for w in args.weights]
    os.makedirs(args.out_dir, exist_ok=True)

    models, targ = [], []
    for w in args.weights:
        m = build_model_from_checkpoint(
            w, num_classes=13, device=device, eval_mode=False
        )
        models.append(m)
        targ.append(dict(get_target_modules(m, args.part)))
    layers = list(targ[0].keys())

    print(
        f"\n{'layer':<16}{'model':<16}{'R2(1G)':>8}{'R2(NG)':>8}   sigmas(NG, sorted)"
    )
    print("-" * 78)
    for layer in layers:
        n = len(labels)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
        for j, (lab, m) in enumerate(zip(labels, models)):
            erf = compute_erf(
                m, targ[j][layer], args.input_size, args.n_samples, device
            )
            x, row, c, sm, p1, r1, pn, rn = analyze(
                erf, args.n_components, cusp=args.cusp
            )
            sig_list = (
                sorted(float(pn[i + 1]) for i in range(0, len(pn), 2))
                if pn is not None
                else []
            )
            sig_str = ", ".join(f"{s:.1f}" for s in sig_list)
            print(
                f"{layer:<16}{lab:<16}{r1:>8.3f}{rn:>8.3f}   [{sig_str}]  (σ_mom={sm:.1f})"
            )

            ax = axes[0][j]
            ax.plot(x, row, color="tomato", lw=1.6, label="ERF", zorder=5)
            if p1 is not None:
                ax.plot(
                    x,
                    _mix(x, c, p1),
                    color="dodgerblue",
                    ls="--",
                    lw=1.6,
                    label=f"1 Gaussian  R²={r1:.3f}",
                )
            if pn is not None:
                kind = (
                    f"{args.n_components}G+cusp"
                    if args.cusp
                    else f"{args.n_components} Gaussians"
                )
                ax.plot(
                    x,
                    _mix(x, c, pn, cusp=args.cusp),
                    color="green",
                    lw=1.8,
                    label=f"{kind}  R²={rn:.3f}",
                )
                for i in range(0, len(pn), 2):
                    if args.cusp and i == 0:
                        comp = pn[i] * np.exp(-np.abs(x - c) / pn[i + 1])
                    else:
                        comp = pn[i] * np.exp(-((x - c) ** 2) / (2 * pn[i + 1] ** 2))
                    ax.plot(x, comp, color="green", ls=":", lw=0.9, alpha=0.7)
            ax.set_xlim(c - args.zoom, c + args.zoom)
            ax.set_title(f"{lab} | {layer}")
            ax.set_xlabel("pixel")
            ax.set_ylabel("normalized gradient")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        plt.suptitle(
            f"ERF single vs {args.n_components}-Gaussian mixture fit — {layer}  "
            "(NG が乗れば集約ガウス=AGD)",
            fontsize=12,
        )
        plt.tight_layout()
        p = os.path.join(args.out_dir, f"erf_gmm_{layer}.png")
        plt.savefig(p, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  → 保存: {p}")

    print(f"\n完了。結果フォルダ: {args.out_dir}")


if __name__ == "__main__":
    main()
