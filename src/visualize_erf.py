"""
Effective Receptive Field (ERF) 可視化 — エンコーダー / デコーダー両対応

Luo et al. 2017 "Understanding the Effective Receptive Field in Deep CNNs"
の手法に従い:
  1. ランダム入力を生成 (requires_grad=True)
  2. フォワードパス → 対象モジュールの出力を捕捉
  3. その特徴マップの中心ピクセルに勾配=1、他は0をセット
  4. backward() で入力まで逆伝播
  5. |∂feat_center / ∂x_input| を N 回平均

対象:
  encoder : backbone.stages[0..3]  (U-Net 収縮パス 4 層)
  decoder : up3, up2, up1, up0      (U-Net 拡張パス 4 層)

使い方:
  python3 src/visualize_erf.py --weights <pth>                # encoder + decoder 全層
  python3 src/visualize_erf.py --weights <pth> --part encoder # encoder のみ
  python3 src/visualize_erf.py --weights <pth> --part decoder # decoder のみ
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


def get_target_modules(model, part):
    """
    可視化対象モジュールを (ラベル, module) のリストで返す。

    encoder: stages[0..3]  → H/4, H/8, H/16, H/32
    decoder: up3,up2,up1,up0 → H/16, H/8, H/4, H/2
    """
    backbone = model.backbone
    encoder = [
        ("encoder_stage0", backbone.stages[0]),
        ("encoder_stage1", backbone.stages[1]),
        ("encoder_stage2", backbone.stages[2]),
        ("encoder_stage3", backbone.stages[3]),
    ]
    decoder = [
        ("decoder_up3", model.up3),
        ("decoder_up2", model.up2),
        ("decoder_up1", model.up1),
        ("decoder_up0", model.up0),
    ]
    if part == "encoder":
        return encoder
    if part == "decoder":
        return decoder
    return encoder + decoder


def compute_erf(model, target_module, input_size, n_samples, device):
    """
    Luo et al. 2017 の方法で target_module の出力中心ピクセルに対する
    入力 ERF を計算する。

    Returns:
        erf: (H_in, W_in) numpy array
    """
    model.eval()
    erf_accum = None

    for sample_idx in range(n_samples):
        x = torch.randn(1, 3, input_size, input_size, device=device, requires_grad=True)

        captured = {}

        def hook_fn(module, inp, out):
            captured["feat"] = out
            out.retain_grad()

        hook = target_module.register_forward_hook(hook_fn)
        model(x)
        hook.remove()

        feat = captured["feat"]  # (1, C, H_feat, W_feat)
        H_feat, W_feat = feat.shape[2], feat.shape[3]
        cy, cx = H_feat // 2, W_feat // 2

        # 論文 Section 3: 出力平面の中心に勾配 1、他は 0 を置いて逆伝播
        grad_feat = torch.zeros_like(feat)
        grad_feat[0, :, cy, cx] = 1.0  # 全チャネルに 1

        model.zero_grad()
        feat.backward(gradient=grad_feat)

        with torch.no_grad():
            grad_input = x.grad.abs().mean(dim=1).squeeze(0).cpu()  # (H_in, W_in)

        erf_accum = grad_input if erf_accum is None else erf_accum + grad_input

        if (sample_idx + 1) % 20 == 0:
            print(f"    {sample_idx + 1}/{n_samples} 完了")

    return (erf_accum / n_samples).numpy()


def _gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def _upper_envelope(row):
    """
    局所最大値をキュービックスプライン補間した上側包絡線を返す。
    補間値が元の曲線を下回る箇所は元の値で上書きする（常に row 以上を保証）。
    """
    x = np.arange(len(row), dtype=float)
    distance = max(3, len(row) // 30)
    peaks, _ = find_peaks(row, distance=distance)
    # 両端を必ず含める
    idx = np.unique(np.concatenate([[0], peaks, [len(row) - 1]]))
    if len(idx) < 3:
        return row.copy()
    f = interp1d(
        idx.astype(float), row[idx], kind="cubic", bounds_error=False, fill_value=0.0
    )
    envelope = f(x)
    return np.maximum(envelope, row)


def _fit_gaussian(row):
    """1D 配列 row にガウス関数をフィットして (A, mu, sigma) を返す。"""
    x = np.arange(len(row), dtype=float)
    try:
        p0 = [row.max(), float(np.argmax(row)), len(row) / 8.0]
        popt, _ = curve_fit(_gaussian, x, row, p0=p0, maxfev=5000)
        return popt
    except Exception:
        return None


def plot_erf(erf, title_prefix, save_path):
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
    axes[0, 0].set_title(f"{title_prefix} — Linear")
    axes[0, 0].axhline(cy, color="cyan", lw=0.8, ls="--")
    axes[0, 0].axvline(cx, color="cyan", lw=0.8, ls="--")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(erf_log, cmap="hot")
    axes[0, 1].set_title(f"{title_prefix} — Log")
    axes[0, 1].axhline(cy, color="cyan", lw=0.8, ls="--")
    axes[0, 1].axvline(cx, color="cyan", lw=0.8, ls="--")
    plt.colorbar(im1, ax=axes[0, 1])

    # 水平断面（線形スケール）＋上側包絡線＋ガウス包絡線フィット
    axes[0, 2].plot(x_px, row_linear, color="tomato", lw=1.2, label="ERF", zorder=3)
    axes[0, 2].plot(
        x_px,
        upper_env,
        color="limegreen",
        lw=1.0,
        ls=":",
        zorder=4,
        label="Upper envelope",
    )
    axes[0, 2].axvline(cx, color="gray", lw=0.8, ls="--")
    sigma_fit = None
    if gauss_params is not None:
        A_fit, mu_fit, sigma_fit = gauss_params
        sigma_fit = abs(sigma_fit)
        gauss_curve = _gaussian(x_px, A_fit, mu_fit, sigma_fit)
        axes[0, 2].fill_between(
            x_px, 0, gauss_curve, alpha=0.25, color="dodgerblue", zorder=1
        )
        axes[0, 2].plot(
            x_px,
            gauss_curve,
            color="dodgerblue",
            lw=1.5,
            ls="--",
            zorder=2,
            label=f"Gaussian fit to envelope\n$\\mu$={mu_fit:.1f}, $\\sigma$={sigma_fit:.1f}",
        )
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

    axes[1, 1].plot(
        freq_x, fft_row, color="steelblue", lw=1.2, label="FFT of ERF", zorder=3
    )
    axes[1, 1].plot(
        freq_x,
        upper_env_fft,
        color="limegreen",
        lw=1.0,
        ls=":",
        zorder=4,
        label="Upper envelope",
    )
    gauss_fft_curve = None
    if gp_fft is not None:
        A_fft, mu_fft, sigma_fft_f = gp_fft
        sigma_fft_f = abs(sigma_fft_f)
        gauss_fft_curve = _gaussian(x_fft, A_fft, mu_fft, sigma_fft_f)
        axes[1, 1].fill_between(
            freq_x, 0, gauss_fft_curve, alpha=0.25, color="orange", zorder=1
        )
        axes[1, 1].plot(
            freq_x,
            gauss_fft_curve,
            color="orange",
            lw=1.5,
            ls="--",
            zorder=2,
            label=f"Gaussian fit to envelope\n$\\sigma_f$={sigma_fft_f:.1f} px",
        )
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
    axes[1, 2].plot(
        freq_x,
        fft_row_log,
        color="steelblue",
        lw=1.2,
        label="FFT of ERF (log)",
        zorder=3,
    )
    axes[1, 2].plot(
        freq_x,
        upper_env_fft_log,
        color="limegreen",
        lw=1.0,
        ls=":",
        zorder=4,
        label="Upper envelope",
    )
    if gauss_fft_curve is not None:
        g_log = np.log(gauss_fft_curve + eps)
        g_log = (g_log - g_log.min()) / (g_log.max() - g_log.min() + eps)
        axes[1, 2].fill_between(freq_x, 0, g_log, alpha=0.25, color="orange", zorder=1)
        axes[1, 2].plot(
            freq_x,
            g_log,
            color="orange",
            lw=1.5,
            ls="--",
            zorder=2,
            label="Gaussian envelope (log of linear)",
        )
    axes[1, 2].axvline(0, color="gray", lw=0.8, ls="--")
    axes[1, 2].set_title("FFT cross-section (log) + Gaussian envelope")
    axes[1, 2].set_xlabel("Spatial frequency (cycles/pixel)")
    axes[1, 2].set_ylabel("Normalized log amplitude")
    axes[1, 2].set_xlim(-0.5, 0.5)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    sigma_str = f"σ={sigma_fit:.1f} px" if sigma_fit is not None else "fit failed"
    plt.suptitle(
        f"ERF — UniConvNet-T U-Net  [{title_prefix}]  ({sigma_str})\n"
        "(Luo et al. 2017: ∂feat_center / ∂x_input)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--part",
        choices=["encoder", "decoder", "all"],
        default="all",
        help="可視化対象。encoder / decoder / all (両方)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="入力画像サイズ。大きいほど正確だが遅い",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100, help="平均化するサンプル数 (論文は 1000)"
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/erf_results",
        help="結果を保存するルートフォルダ",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="モデル識別子。複数モデルを同じ out-dir に出すとき衝突を防ぐ "
        "(例 --tag pure_spec → erf_pure_spec_<layer>.png、タイトルにも付記)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")

    if args.weights and os.path.exists(args.weights):
        # separable/spectral/adaptive チェックポイントも dw_mode 自動判別で構築・load。
        from ckpt_utils import build_model_from_checkpoint

        model = build_model_from_checkpoint(
            args.weights, num_classes=13, device=device, eval_mode=False
        )
        print(f"重みロード: {args.weights}")
    else:
        print("重みなし（ランダム初期化）で可視化します")
        model = UniConvNet_UNet_13CH(num_classes=13).to(device)

    targets = get_target_modules(model, args.part)

    for label, module in targets:
        # encoder / decoder でサブフォルダを分ける
        sub = "encoder" if label.startswith("encoder") else "decoder"
        out_dir = os.path.join(args.out_dir, sub)
        os.makedirs(out_dir, exist_ok=True)

        print(
            f"\n[{label}] ERF 計算中 "
            f"(input={args.input_size}x{args.input_size}, samples={args.n_samples})..."
        )
        erf = compute_erf(model, module, args.input_size, args.n_samples, device)

        pref = f"{args.tag}_" if args.tag else ""
        title = f"{args.tag} | {label}" if args.tag else label
        out_path = os.path.join(out_dir, f"erf_{pref}{label}.png")
        plot_erf(erf, title, out_path)

    print(f"\n全層完了。結果フォルダ: {args.out_dir}")


if __name__ == "__main__":
    main()
