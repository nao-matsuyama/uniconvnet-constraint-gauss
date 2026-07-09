# coding:utf-8
"""
AGD プローブ — 各 depthwise 機構の「実効カーネル(インパルス応答)」を同一 RF で作り、
その AGD(漸近ガウス性)を定量比較する。学習不要でローカル CPU で完結する。

背景:
  UniConvNet の RFA は「滑らかな(低周波集中=ガウス的)大カーネル」で ERF がガウス
  (AGD)になることを前提にする。単一 dilated conv で大 RF を作ると、タップが疎に
  なりカーネルにゼロ穴が空く → 周波数域で **エイリアシング複製** が出て AGD が崩れる。
  機構A(separable) と 機構B(spectral) は dense/ガウスの意味論を保つので AGD を保つ、
  という主張を、以下 3 指標で数値化する:

    sigma_moment    … 実効カーネル断面の 2次モーメント σ (px) = 実 RF の広さ
    excess_kurtosis … 断面の過剰尖度 (0 = ガウス, 疎grid/尖りで非0)
    fft_sidelobe    … |FFT| が同σガウス包絡線を超える最大量 (エイリアシング複製の高さ,
                      0 = 滑らかな低域通過, 大 = 高周波に複製ローブ = AGD 崩壊)

比較する 4 機構 (すべて実効 RF σ_target を揃える):
  dense    : σ を張れる大きさの密ガウス K×K            … 参照(理想 AGD)
  dilated  : 小さい基底ガウス K0×K0 を dilation d で疎展開 … AGD が崩れる版(対照)
  separable: 1D ガウス × 1D ガウス (機構A)                … 2D=1D×1D なので AGD 厳密保存
  spectral : 解析ガウス低域通過 Ŵ(σ)=exp(-2π²σ²f²) (機構B) … 定義からガウス = AGD 保存

使い方:
  python3 src/agd_probe.py --sigma 6 --dilation 4 --canvas 96 \
      --out-dir /workspace/erf_results/agd_probe
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from models.gaussian_derivative_dw import GaussianDerivativeDW
from models.separable_dw import SeparableDWConv
from models.spectral_dw import SpectralDW


def _gauss1d(k, sigma):
    x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
    g = torch.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def impulse(canvas):
    x = torch.zeros(1, 1, canvas, canvas)
    x[0, 0, canvas // 2, canvas // 2] = 1.0
    return x


def _dense_kernel(sigma):
    k = 2 * int(round(3 * sigma)) + 1  # ±3σ を張る密カーネル
    g = _gauss1d(k, sigma)
    return (g[:, None] * g[None, :]).view(1, 1, k, k), k


def resp_dense(canvas, sigma):
    w, k = _dense_kernel(sigma)
    return F.conv2d(impulse(canvas), w, padding=k // 2)[0, 0]


def resp_dilated(canvas, sigma, dilation, base_k=7):
    # 基底 K0×K0 ガウス(タップ単位 σ0 = σ/d)を dilation d で疎展開 → 実効 σ = σ0*d = σ。
    sigma0 = sigma / dilation
    g = _gauss1d(base_k, sigma0)
    w = (g[:, None] * g[None, :]).view(1, 1, base_k, base_k)
    pad = dilation * (base_k - 1) // 2
    return F.conv2d(impulse(canvas), w, padding=pad, dilation=dilation)[0, 0]


def resp_separable(canvas, sigma, base_k=None):
    # 機構A: 1D ガウス × 1D ガウス。実コードの SeparableDWConv を通す。
    k = base_k or (2 * int(round(3 * sigma)) + 1)
    g = _gauss1d(k, sigma)
    m = SeparableDWConv(1, k)
    with torch.no_grad():
        m.weight_h.zero_().view(k).copy_(g)
        m.weight_v.zero_().view(k).copy_(g)
        m.bias.zero_()
    return m(impulse(canvas))[0, 0].detach()


def resp_gauss_deriv(canvas, sigma, order=2, mode="smooth"):
    # ガウス微分基底の実効カーネル(インパルス応答)。coeff を手で置いて基底の振る舞いを見る。
    #   mode="smooth" : order-0 成分のみ = 純ガウス(dense/spectral と一致するはず)。
    #   mode="edge"   : 1次 ∂x 成分 = 方向性エッジ(純ガウスが表現できない構造だが、周波数
    #                   包絡はガウスに縛られ dilated のような複製ローブが出ない = AGD 側の主張)。
    k = 2 * int(round(3 * sigma)) + 1
    m = GaussianDerivativeDW(
        1, kernel_size=k, order=max(order, 1), init_sigma=sigma, max_sigma=max(8.0, sigma * 2)
    )
    with torch.no_grad():
        m.coeff.zero_()
        if mode == "smooth":
            m.coeff[0, 0, 0] = 1.0  # 平滑(order-0)
        else:
            m.coeff[0, 0, 1] = 1.0  # ∂x(垂直エッジ検出器)
        m.bias.zero_()
    return m(impulse(canvas))[0, 0].detach()


def resp_spectral(canvas, sigma, alpha=1.0):
    # 機構B: 解析ガウス低域通過(純スペクトル, local 枝オフ)。実コードの SpectralDW を通す。
    # alpha を大きくすると切り出しが緩く(η→1) = フルスペクトルに近づく。
    m = SpectralDW(
        1,
        kernel_size=7,
        init_sigma=sigma,
        max_sigma=max(64.0, sigma * 2),
        alpha=alpha,
        use_local_branch=False,
    )
    return m(impulse(canvas))[0, 0].detach()


def agd_metrics(kernel2d):
    """実効カーネル(2D, torch)から (sigma_moment, excess_kurtosis, fft_sidelobe)。"""
    eps = 1e-12
    k = kernel2d.detach().cpu().numpy().astype(np.float64)
    H, W = k.shape
    cy = H // 2
    # erf_sigma_table と同じ規約: 負値は 0 にクリップして 1D 密度とみなす。
    row = np.clip(k[cy, :], 0, None)
    row = row / (row.sum() + eps)

    x = np.arange(W, dtype=np.float64)
    mu = (row * x).sum()
    var = (row * (x - mu) ** 2).sum()
    sigma_moment = math.sqrt(max(var, eps))
    kurt = (row * (x - mu) ** 4).sum() / (var**2 + eps) - 3.0

    # FFT 断面: 同σガウス包絡線をどれだけ超えるか(エイリアシング複製の高さ)。
    line = k[cy, :].astype(np.float64)
    amp = np.abs(np.fft.rfft(line))
    amp = amp / (amp[0] + eps)  # DC 正規化
    freqs = np.fft.rfftfreq(W)
    # 実効 σ の理想ガウス周波数包絡線 exp(-2π²σ²f²)
    env = np.exp(-2 * math.pi**2 * sigma_moment**2 * freqs**2)
    # 主ローブを外れた高周波側で包絡線を超えた最大量(複製ローブ) = AGD 崩れ量
    mask = env < 0.05  # 主ローブが十分減衰した帯域
    sidelobe = float(np.max((amp - env)[mask])) if mask.any() else 0.0
    sidelobe = max(0.0, sidelobe)
    return sigma_moment, float(kurt), sidelobe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sigma", type=float, default=6.0, help="全機構で揃える実効 RF σ (px)"
    )
    ap.add_argument("--dilation", type=int, default=4, help="dilated 対照の dilation d")
    ap.add_argument(
        "--canvas", type=int, default=96, help="インパルス応答のキャンバス辺長"
    )
    ap.add_argument("--out-dir", default="/workspace/erf_results/agd_probe")
    args = ap.parse_args()

    sig, d, N = args.sigma, args.dilation, args.canvas
    # ラベルは matplotlib の CJK 豆腐化を避けるため ASCII (機構A=mech-A, 機構B=mech-B)。
    configs = [
        ("dense", resp_dense(N, sig)),
        (f"dilated(d={d})", resp_dilated(N, sig, d)),
        ("separable(mech-A)", resp_separable(N, sig)),
        ("spectral(mech-B,a=1)", resp_spectral(N, sig, alpha=1.0)),
        # ガウス微分 order-0 = 純ガウス。dense と厳密一致 → 基底が AGD を保つことの確認。
        # (order≥1 のエッジ核は DC≈0 で agd_metrics(DC正規化)が破綻するため表には載せない。
        #  エッジ表現力は実タスクでの N=2 vs N=0 比較と effective_kernel 可視化で見る。)
        ("gauss-deriv(smooth)", resp_gauss_deriv(N, sig, mode="smooth")),
    ]

    print("=" * 74)
    print(f" AGD プローブ  (目標 σ={sig}px, canvas={N}, dilated d={d})")
    print("=" * 74)
    print(
        f"{'mechanism':<22}{'sigma_moment':>13}{'excess_kurt':>13}{'fft_sidelobe':>13}"
    )
    print("-" * 74)
    rows = []
    for name, resp in configs:
        sm, ku, sl = agd_metrics(resp)
        rows.append((name, sm, ku, sl, resp))
        print(f"{name:<22}{sm:>13.2f}{ku:>+13.3f}{sl:>13.4f}")
    print("=" * 74)
    print(" excess_kurt≈0 かつ fft_sidelobe≈0 = AGD 保持 (ガウス, 滑らかな低域通過)。")
    print(
        " dilated は fft_sidelobe が大 = 周波数複製ローブ = 疎サンプリングで AGD 崩壊。"
    )
    print(
        " separable は dense と厳密一致 (2Dガウス=1Dガウス積)。spectral は複製ローブ無し。"
    )

    # ── 機構B: α(=切り出し) に対する コスト(∝η²) と AGD の依存 ──
    print("\n" + "-" * 74)
    print(f" spectral の切り出し依存 (σ={sig}px, cost∝η², full と比較)")
    print(
        f"{'alpha':>8}{'eta':>10}{'cost~eta^2':>12}{'fft_sidelobe':>14}{'excess_kurt':>13}{'L2_vs_full':>12}"
    )
    print("-" * 74)
    full = resp_spectral(N, sig, alpha=1e9).cpu().numpy()
    for a in [0.5, 1.0, 2.0, 4.0, 8.0]:
        mm = SpectralDW(
            1,
            kernel_size=7,
            init_sigma=sig,
            max_sigma=max(64.0, sig * 2),
            alpha=a,
            use_local_branch=False,
        )
        eta = mm.current_eta()
        resp = mm(impulse(N))[0, 0].detach().cpu().numpy()
        _, ku, sl = agd_metrics(torch.from_numpy(resp))
        l2 = float(np.linalg.norm(resp - full) / (np.linalg.norm(full) + 1e-12))
        print(f"{a:>8.1f}{eta:>10.3f}{eta**2:>12.4f}{sl:>14.4f}{ku:>+13.3f}{l2:>12.2e}")
    print("-" * 74)
    print(
        " α↑(帯域を広く残す)ほど full に近づき AGD 改善、α↓ほど安いが僅かに複製が出る。"
    )

    # ── 図: (上) 実効カーネル断面, (下) FFT 断面 + 同σガウス包絡線 ──
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib 無しのため図はスキップ)")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    ncol = len(rows)
    fig, axes = plt.subplots(2, ncol, figsize=(5 * ncol, 8))
    for j, (name, sm, ku, sl, resp) in enumerate(rows):
        k = resp.detach().cpu().numpy()
        cy = k.shape[0] // 2
        row = k[cy, :]
        axes[0, j].plot(row, color="tomato")
        axes[0, j].set_title(f"{name}\nσ={sm:.1f} kurt={ku:+.2f}")
        axes[0, j].grid(True, alpha=0.3)

        amp = np.abs(np.fft.rfft(row))
        amp = amp / (amp[0] + 1e-12)
        freqs = np.fft.rfftfreq(len(row))
        env = np.exp(-2 * math.pi**2 * sm**2 * freqs**2)
        axes[1, j].plot(freqs, amp, color="steelblue", label="|FFT| of kernel")
        axes[1, j].plot(
            freqs, env, color="orange", ls="--", label="Gaussian envelope(σ)"
        )
        axes[1, j].fill_between(
            freqs,
            env,
            amp,
            where=(amp > env),
            color="red",
            alpha=0.3,
            label="sidelobe (AGD break)",
        )
        axes[1, j].set_title(f"FFT: sidelobe={sl:.3f}")
        axes[1, j].set_xlabel("cycles/px")
        axes[1, j].set_ylim(0, 1.05)
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].legend(fontsize=7)
    plt.suptitle(
        f"AGD probe — effective kernel & spectrum (target σ={sig}px)", fontsize=14
    )
    plt.tight_layout()
    p = os.path.join(args.out_dir, f"agd_probe_sigma{int(sig)}_d{d}.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n保存: {p}")


if __name__ == "__main__":
    main()
