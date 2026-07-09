"""
モデルのパラメータ数と FLOPs を計測するスクリプト。

使い方:
  python3 src/model_stats.py
  python3 src/model_stats.py --input-size 256 --weights /workspace/experiments/run_xxx/best_uniconvnet_unet.pth
"""

import argparse
import math
import os
import sys

import torch

sys.path.append(os.path.dirname(__file__))
from model_uniconvnet_unet import UniConvNet_UNet_13CH
from models.dilated_dw import DilatedDWConv
from models.spectral_dw import SpectralDW

try:
    from models.separable_dw import SeparableDWConv
except ImportError:
    SeparableDWConv = ()  # 機構A 未導入でも isinstance が安全に False になる

try:
    from models.gaussian_derivative_dw import GaussianDerivativeDW
except ImportError:
    GaussianDerivativeDW = ()  # 未導入でも isinstance が安全に False になる


# ─────────────────────────────────────────────
# パラメータ数
# ─────────────────────────────────────────────
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────
# FLOPs (手動フック方式、外部ライブラリ不要)
# ─────────────────────────────────────────────
def count_flops(model, input_tensor):
    """
    Conv2d / Linear / BatchNorm2d / LayerNorm の FLOPs を
    フォワードフックで集計する。
    """
    flops_log = []

    def conv_hook(module, inp, out):
        b, c_out, h_out, w_out = out.shape
        c_in = inp[0].shape[1]
        kh, kw = (
            module.kernel_size
            if isinstance(module.kernel_size, tuple)
            else (module.kernel_size, module.kernel_size)
        )
        groups = module.groups
        # MACs = B * C_out * H_out * W_out * (C_in/groups * kH * kW)
        macs = b * c_out * h_out * w_out * (c_in // groups) * kh * kw
        flops_log.append(2 * macs)  # FLOPs = 2 * MACs

    def linear_hook(module, inp, out):
        b = inp[0].shape[0] if inp[0].ndim == 2 else inp[0].numel() // inp[0].shape[-1]
        macs = b * module.in_features * module.out_features
        flops_log.append(2 * macs)

    def bn_hook(module, inp, out):
        flops_log.append(2 * out.numel())

    def ln_hook(module, inp, out):
        flops_log.append(2 * out.numel())

    def dilated_dw_hook(module, inp, out):
        # DilatedDWConv は floor/ceil の 2 つの depthwise conv を実行する。
        # depthwise なので 1 conv の MACs = B * C * H * W * (k*k)。
        # タップ数 k*k は dilation に依らず一定 (= 構造スパース)。
        b, c, h, w = out.shape
        k = module.kernel_size
        macs_one = b * c * h * w * k * k
        flops_log.append(2 * (2 * macs_one))  # 2 conv (floor+ceil) × FLOPs=2*MACs

    def separable_dw_hook(module, inp, out):
        # 機構A: R 本の (1×K と K×1) depthwise を直列に流し和を取る。
        # 各 rank で 2*K タップ、合計 2*R*K タップ (密 K×K の K²/2RK 倍の削減)。
        b, c, h, w = out.shape
        k = module.kernel_size
        r = getattr(module, "rank", 1)
        macs = b * c * h * w * (2 * r * k)
        flops_log.append(2 * macs)

    def gauss_deriv_hook(module, inp, out):
        # ガウス微分基底: M=N+1 本の (水平 1×K → 垂直 K×1) 分離 depthwise の和。
        # タップ 2*M*K (密 K² の K/2M 倍削減)。FFT 非使用 = cuDNN 高速。
        b, c, h, w = out.shape
        k = module.kernel_size
        m_basis = module.num_basis
        macs = b * c * h * w * (2 * m_basis * k)
        flops_log.append(2 * macs)

    def spectral_dw_hook(module, inp, out):
        # 機構B: rfft2 + 動的切り出し帯域での複素乗算 + irfft2。
        #   FFT: 2 変換 (fwd rfft2 + inv irfft2) ≈ 2 * (5 * C*H*W*log2(H*W)) FLOPs (実FFT概算)。
        #   複素乗算: 切り出し帯域 H'×W' のみ、C * H' * W' * 6 FLOPs。
        #   local 枝 (use_local_branch=True 時): 密 depthwise K² MACs。
        b, c, h, w = out.shape
        wf = w // 2 + 1
        logn = max(1.0, math.log2(max(2, h * w)))
        fft_flops = 2 * (5.0 * c * h * w * logn) * b
        hp, wp = module._crop_sizes(h, wf)
        mult_flops = 6.0 * c * hp * wp * b
        flops_log.append(fft_flops + mult_flops)
        if getattr(module, "use_local_branch", False):
            k = module.kernel_size
            flops_log.append(2 * (b * c * h * w * k * k))

    hooks = []
    for m in model.modules():
        if isinstance(m, DilatedDWConv):
            # Conv2d より先に判定 (DilatedDWConv は Conv2d を継承しない独自モジュール)
            hooks.append(m.register_forward_hook(dilated_dw_hook))
        elif isinstance(m, SeparableDWConv):
            hooks.append(m.register_forward_hook(separable_dw_hook))
        elif GaussianDerivativeDW and isinstance(m, GaussianDerivativeDW):
            hooks.append(m.register_forward_hook(gauss_deriv_hook))
        elif isinstance(m, SpectralDW):
            hooks.append(m.register_forward_hook(spectral_dw_hook))
        elif isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, torch.nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(bn_hook))
        elif isinstance(m, torch.nn.LayerNorm):
            hooks.append(m.register_forward_hook(ln_hook))

    with torch.no_grad():
        model(input_tensor)

    for h in hooks:
        h.remove()

    return sum(flops_log)


def human_readable(n):
    if n >= 1e9:
        return f"{n/1e9:.2f} G"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.2f} K"
    return str(n)


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="評価する .pth ファイルのパス")
    parser.add_argument(
        "--input-size", type=int, default=256, help="入力画像サイズ (正方形)"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UniConvNet_UNet_13CH(num_classes=13)
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(
            torch.load(args.weights, map_location="cpu", weights_only=True)
        )
        print(f"重みロード: {args.weights}")
    model.to(device).eval()

    # パラメータ数
    total_params, trainable_params = count_parameters(model)

    # FLOPs
    dummy = torch.zeros(
        args.batch_size, 3, args.input_size, args.input_size, device=device
    )
    flops = count_flops(model, dummy)

    print("\n" + "=" * 45)
    print(f"  モデル      : UniConvNet-T U-Net (13 class)")
    print(
        f"  入力サイズ  : {args.batch_size} x 3 x {args.input_size} x {args.input_size}"
    )
    print("-" * 45)
    print(f"  総パラメータ数   : {human_readable(total_params)} ({total_params:,})")
    print(
        f"  学習可能パラメータ: {human_readable(trainable_params)} ({trainable_params:,})"
    )
    print(f"  FLOPs (1枚)      : {human_readable(flops)} ({flops:,})")
    print("=" * 45)


if __name__ == "__main__":
    main()
