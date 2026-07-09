"""
Effective Receptive Field (ERF) 可視化 — U-Net 最終出力を対象

Luo et al. 2017 の手法に従い:
  1. ランダム入力を生成 (requires_grad=True)
  2. フォワードパス → 最終出力 logit (1, 13, H, W) を取得
  3. logit の中心ピクセルに勾配=1、他は0をセット (全クラス合計)
  4. backward() で入力まで逆伝播
  5. |∂logit_center / ∂x_input| を N 回平均

  → U-Net 全体 (encoder + decoder) の ERF を可視化する

使い方:
  python3 src/visualize_receptive_field.py
  python3 src/visualize_receptive_field.py \
      --weights /workspace/experiments/run_xxx/best_uniconvnet_unet.pth \
      --input-size 256 --n-samples 100 --out erf_unet.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

sys.path.append(os.path.dirname(__file__))
from model_uniconvnet_unet import UniConvNet_UNet_13CH


def compute_erf_output(model, input_size=256, n_samples=100, device="cuda"):
    """
    Luo et al. 2017 の方法で U-Net 最終出力の ERF を計算。

    Returns:
        erf: (H_in, W_in) numpy array
    """
    model.eval()
    erf_accum = None

    for sample_idx in range(n_samples):
        x = torch.randn(1, 3, input_size, input_size, device=device, requires_grad=True)

        logits = model(x)  # (1, 13, H, W)
        H_out, W_out = logits.shape[2], logits.shape[3]
        cy, cx = H_out // 2, W_out // 2

        # 論文 Section 3: "place a gradient signal of 1 at the center of the
        # output plane and 0 everywhere else"
        # 全クラス (13ch) の中心ピクセルに 1 を置く
        grad_out = torch.zeros_like(logits)
        grad_out[0, :, cy, cx] = 1.0

        model.zero_grad()
        logits.backward(gradient=grad_out)

        with torch.no_grad():
            grad_input = x.grad.abs().mean(dim=1).squeeze(0).cpu()  # (H_in, W_in)

        if erf_accum is None:
            erf_accum = grad_input
        else:
            erf_accum += grad_input

        if (sample_idx + 1) % 20 == 0:
            print(f"  {sample_idx + 1}/{n_samples} 完了")

    erf = (erf_accum / n_samples).numpy()
    return erf


def _gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def _upper_envelope(row):
    """局所最大値をスプライン補間した上側包絡線（常に row 以上）。"""
    x = np.arange(len(row), dtype=float)
    distance = max(3, len(row) // 30)
    peaks, _ = find_peaks(row, distance=distance)
    idx = np.unique(np.concatenate([[0], peaks, [len(row) - 1]]))
    if len(idx) < 3:
        return row.copy()
    f = interp1d(idx.astype(float), row[idx], kind="cubic",
                 bounds_error=False, fill_value=0.0)
    return np.maximum(f(x), row)


def _fit_gaussian(row):
    x = np.arange(len(row), dtype=float)
    try:
        p0 = [row.max(), float(np.argmax(row)), len(row) / 8.0]
        popt, _ = curve_fit(_gaussian, x, row, p0=p0, maxfev=5000)
        return popt
    except Exception:
        return None


def plot_erf(erf, save_path):
    eps = 1e-10
    H, W = erf.shape
    cy, cx = H // 2, W // 2

    # 正規化
    erf_linear = erf / (erf.max() + eps)
    erf_log = np.log(erf + eps)
    erf_log = (erf_log - erf_log.min()) / (erf_log.max() - erf_log.min() + eps)

    # 2D FFT（振幅スペクトル、中心シフト済み）
    fft2 = np.fft.fftshift(np.abs(np.fft.fft2(erf_linear)))
    fft2_log = np.log(fft2 + eps)
    fft2_log = (fft2_log - fft2_log.min()) / (fft2_log.max() - fft2_log.min() + eps)

    # 上側包絡線を計算し、それにガウスをフィット
    row_linear = erf_linear[cy, :]
    x_px = np.arange(W, dtype=float)
    freq_x = np.fft.fftshift(np.fft.fftfreq(W))
    upper_env = _upper_envelope(row_linear)
    gauss_params = _fit_gaussian(upper_env)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ── 上段: ERF ──────────────────────────────────────
    im0 = axes[0, 0].imshow(erf_linear, cmap="hot")
    axes[0, 0].set_title("ERF — U-Net Output  (Linear)")
    axes[0, 0].axhline(cy, color="cyan", lw=0.8, ls="--")
    axes[0, 0].axvline(cx, color="cyan", lw=0.8, ls="--")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(erf_log, cmap="hot")
    axes[0, 1].set_title("ERF — U-Net Output  (Log)")
    axes[0, 1].axhline(cy, color="cyan", lw=0.8, ls="--")
    axes[0, 1].axvline(cx, color="cyan", lw=0.8, ls="--")
    plt.colorbar(im1, ax=axes[0, 1])

    # 水平断面（線形スケール）＋上側包絡線＋ガウス包絡線フィット
    axes[0, 2].plot(x_px, row_linear, color="tomato", lw=1.2, label="ERF", zorder=3)
    axes[0, 2].plot(x_px, upper_env, color="limegreen", lw=1.0, ls=":", zorder=4,
                    label="Upper envelope")
    axes[0, 2].axvline(cx, color="gray", lw=0.8, ls="--")
    sigma_fit = None
    if gauss_params is not None:
        A_fit, mu_fit, sigma_fit = gauss_params
        sigma_fit = abs(sigma_fit)
        gauss_curve = _gaussian(x_px, A_fit, mu_fit, sigma_fit)
        axes[0, 2].fill_between(x_px, 0, gauss_curve, alpha=0.25, color="dodgerblue", zorder=1)
        axes[0, 2].plot(x_px, gauss_curve, color="dodgerblue", lw=1.5, ls="--", zorder=2,
                        label=f"Gaussian fit to envelope\n$\\mu$={mu_fit:.1f}, $\\sigma$={sigma_fit:.1f}")
    axes[0, 2].set_title("ERF cross-section (linear) + Gaussian envelope")
    axes[0, 2].set_xlabel("Pixel position (width)")
    axes[0, 2].set_ylabel("Normalized gradient")
    axes[0, 2].set_xlim(0, W)
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # ── 下段: 2D FFT ───────────────────────────────────
    im2 = axes[1, 0].imshow(fft2_log, cmap="viridis")
    axes[1, 0].set_title("2D FFT of ERF (log amplitude)")
    axes[1, 0].axhline(H // 2, color="cyan", lw=0.8, ls="--")
    axes[1, 0].axvline(W // 2, color="cyan", lw=0.8, ls="--")
    plt.colorbar(im2, ax=axes[1, 0])

    # [1,1] FFT linear: 上側包絡線 → ガウスフィット
    fft2_linear = fft2 / (fft2.max() + eps)
    fft_row = fft2_linear[H // 2, :]
    x_fft = np.arange(len(fft_row), dtype=float)
    upper_env_fft = _upper_envelope(fft_row)
    gp_fft = _fit_gaussian(upper_env_fft)

    axes[1, 1].plot(freq_x, fft_row, color="steelblue", lw=1.2, label="FFT of ERF", zorder=3)
    axes[1, 1].plot(freq_x, upper_env_fft, color="limegreen", lw=1.0, ls=":", zorder=4,
                    label="Upper envelope")
    gauss_fft_curve = None
    if gp_fft is not None:
        A_fft, mu_fft, sigma_fft_f = gp_fft
        sigma_fft_f = abs(sigma_fft_f)
        gauss_fft_curve = _gaussian(x_fft, A_fft, mu_fft, sigma_fft_f)
        axes[1, 1].fill_between(freq_x, 0, gauss_fft_curve, alpha=0.25, color="orange", zorder=1)
        axes[1, 1].plot(freq_x, gauss_fft_curve, color="orange", lw=1.5, ls="--", zorder=2,
                        label=f"Gaussian fit to envelope\n$\\sigma_f$={sigma_fft_f:.1f} px")
    axes[1, 1].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1, 1].set_title("FFT cross-section (linear) + Gaussian envelope")
    axes[1, 1].set_xlabel("Spatial frequency (cycles/pixel)")
    axes[1, 1].set_ylabel("Normalized amplitude")
    axes[1, 1].set_xlim(-0.5, 0.5)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # [1,2] FFT log: linear で求めた包絡線の log 表示（再フィットなし）
    fft_row_log = fft2_log[H // 2, :]
    upper_env_fft_log = _upper_envelope(fft_row_log)
    axes[1, 2].plot(freq_x, fft_row_log, color="steelblue", lw=1.2, label="FFT of ERF (log)", zorder=3)
    axes[1, 2].plot(freq_x, upper_env_fft_log, color="limegreen", lw=1.0, ls=":", zorder=4,
                    label="Upper envelope")
    if gauss_fft_curve is not None:
        g_log = np.log(gauss_fft_curve + eps)
        g_log = (g_log - g_log.min()) / (g_log.max() - g_log.min() + eps)
        axes[1, 2].fill_between(freq_x, 0, g_log, alpha=0.25, color="orange", zorder=1)
        axes[1, 2].plot(freq_x, g_log, color="orange", lw=1.5, ls="--", zorder=2,
                        label="Gaussian envelope (log of linear)")
    axes[1, 2].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1, 2].set_title("FFT cross-section (log) + Gaussian envelope")
    axes[1, 2].set_xlabel("Spatial frequency (cycles/pixel)")
    axes[1, 2].set_ylabel("Normalized log amplitude")
    axes[1, 2].set_xlim(-0.5, 0.5)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    sigma_str = f"σ={sigma_fit:.1f} px" if sigma_fit is not None else "fit failed"
    plt.suptitle(
        f"ERF — UniConvNet-T U-Net (full model)  ({sigma_str})\n"
        "(Luo et al. 2017: ∂logit_center / ∂x_input)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="平均化するサンプル数 (論文は 1000)")
    parser.add_argument("--out-dir", default="/workspace/erf_results/unet",
                        help="結果を保存するフォルダ")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")

    model = UniConvNet_UNet_13CH(num_classes=13)
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(
            torch.load(args.weights, map_location="cpu", weights_only=True)
        )
        print(f"重みロード: {args.weights}")
    else:
        print("重みなし（ランダム初期化）で可視化します")
    model.to(device)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"ERF 計算中 (U-Net output, input={args.input_size}x{args.input_size}, "
          f"samples={args.n_samples})...")
    erf = compute_erf_output(
        model,
        input_size=args.input_size,
        n_samples=args.n_samples,
        device=device,
    )

    out_path = os.path.join(args.out_dir, "erf_unet_output.png")
    plot_erf(erf, out_path)
    print(f"\n完了。結果フォルダ: {args.out_dir}")


if __name__ == "__main__":
    main()
