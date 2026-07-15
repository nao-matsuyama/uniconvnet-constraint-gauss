# coding:utf-8
"""
UniConvNet — DCNv3 を標準 PyTorch の depthwise conv で代替した実装。

オリジナルの uniconvnet_t_1k_224_ema.pth と state_dict のキー名が一致するため、
DCNv3 以外のパラメータ (ConvMod / MLPLayer / LayerNorm / gamma) は
load_pretrained_backbone によってそのまま転移される。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dcnv3_pytorch import DCNv3_pytorch
except ImportError:
    from dcnv3_pytorch import DCNv3_pytorch

try:
    from .dilated_dw import DilatedDWConv
except ImportError:
    from dilated_dw import DilatedDWConv

try:
    from .content_adaptive_dw import ContentAdaptiveDW
except ImportError:
    from content_adaptive_dw import ContentAdaptiveDW

try:
    from .spectral_gaussian_dw import SpectralGaussianDW
except ImportError:
    from spectral_gaussian_dw import SpectralGaussianDW

try:
    from .spectral_dw import SpectralDW
except ImportError:
    from spectral_dw import SpectralDW

try:
    from .spectral_mixture_dw import SpectralMixtureDW
except ImportError:
    from spectral_mixture_dw import SpectralMixtureDW

try:
    from .separable_dw import SeparableDWConv
except ImportError:
    try:
        from separable_dw import SeparableDWConv
    except ImportError:
        SeparableDWConv = None  # 機構A 未導入環境でも import を壊さない

try:
    from .gaussian_derivative_dw import (
        GaussianDerivativeDW,
        gauss_deriv_kernel_size,
    )
except ImportError:
    try:
        from gaussian_derivative_dw import (
            GaussianDerivativeDW,
            gauss_deriv_kernel_size,
        )
    except ImportError:
        GaussianDerivativeDW = None  # 未導入環境でも import を壊さない

try:
    from .gaussian_pyramid_dw import GaussianPyramidDW
except ImportError:
    try:
        from gaussian_pyramid_dw import GaussianPyramidDW
    except ImportError:
        GaussianPyramidDW = None  # 未導入環境でも import を壊さない

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    # timm が無い場合のフォールバック
    def trunc_normal_(tensor, std=0.02, **kwargs):
        with torch.no_grad():
            return tensor.normal_(0, std)

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
            return x / keep * mask


# ──────────────────────────────────────────────
# LayerNorm (channels_first / channels_last 両対応)
# ──────────────────────────────────────────────
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def build_norm_layer(
    dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6
):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(nn.Sequential())  # permute は forward で対応
        layers.append(nn.BatchNorm2d(dim))
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(_ToChannelsLast())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(_ToChannelsFirst())
    return nn.Sequential(*layers)


class _ToChannelsFirst(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class _ToChannelsLast(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1)


# ──────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────
class MLPLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


# ──────────────────────────────────────────────
# ConvMod (オリジナルと同じ実装)
# ──────────────────────────────────────────────
class ConvMod(nn.Module):
    def __init__(
        self,
        dim,
        adaptive_dw=False,
        adaptive_dilations=(1, 2, 4, 8),
        spectral_dw=False,
        spectral_init_sigma=1.0,
        spectral_init_gamma=0.0,
        spectral_max_sigma=32.0,
        spectral_use_local_branch=True,
        spectral_alpha=2.0,
        spectral_pad_factor=0.0,
        spectral_crop_quantile=0.0,
        spectral_num_gaussians=3,
        separable_rank=1,
        gauss_deriv_order=2,
        gauss_pyramid_growth=1.6,
        gauss_freeze_scale=False,
        dw_mode="dense",
    ):
        super().__init__()

        # 機構の解決: 明示 dw_mode が最優先。レガシー boolean は mode 文字列へ写像する。
        #   dense            : DilatedDWConv (init dilation=1 = 密 depthwise, 既定・後方互換)
        #   separable        : SeparableDWConv        (機構A: 1×K・K×1 分離)
        #   spectral         : SpectralDW             (機構B: 周波数 + 動的スペクトル切り出し)
        #   spectral_mix     : SpectralMixtureDW      (機構C: 多ガウス混合 + 動的切り出し)
        #   gauss_deriv      : GaussianDerivativeDW   (ガウス微分基底で RF を σ のみに縛る本命)
        #   gauss_pyramid    : GaussianPyramidDW      (多スケール純ガウス + pointwise DoG)
        #   spectral_gaussian: SpectralGaussianDW     (旧・フルサイズ周波数ガウス)
        #   adaptive         : ContentAdaptiveDW      (旧・コンテンツ適応 dilation)
        if dw_mode not in (
            "dense",
            "separable",
            "spectral",
            "spectral_mix",
            "gauss_deriv",
            "gauss_pyramid",
        ):
            raise ValueError(f"未知の dw_mode: {dw_mode}")
        mode = dw_mode
        if dw_mode == "dense":  # レガシー boolean は dense 既定のときだけ効かせる
            if spectral_dw:
                mode = "spectral_gaussian"
            elif adaptive_dw:
                mode = "adaptive"
        self.dw_mode = mode

        def _dw(ch, k):
            # a*.2 の depthwise 枝。mode で機構を差し替える。dense/spectral/spectral_gaussian
            # は weight/bias が (ch,1,k,k) 同形・同キー (a*.2.weight/bias) なので事前学習
            # 重みが転移される。separable は weight_h/weight_v へ SVD 分解で初期化する
            # (load_pretrained_backbone 側で変換)。
            if mode == "spectral":
                # 機構B: gamma=0 初期で元 conv と一致 (転移温存)、σ 大で帯域を切り詰め安く。
                return SpectralDW(
                    ch,
                    kernel_size=k,
                    init_sigma=spectral_init_sigma,
                    init_gamma=spectral_init_gamma,
                    max_sigma=spectral_max_sigma,
                    alpha=spectral_alpha,
                    pad_factor=spectral_pad_factor,
                    crop_quantile=spectral_crop_quantile,
                    use_local_branch=spectral_use_local_branch,
                )
            if mode == "spectral_mix":
                # 機構C: 多ガウス混合。周波数包絡を K ガウス和にし ERF を集約ガウス(AGD)へ。
                # 分離 rank-K 構築 + 振幅考慮の動的切り出しで単一ガウス並みのコスト。
                return SpectralMixtureDW(
                    ch,
                    kernel_size=k,
                    num_gaussians=spectral_num_gaussians,
                    init_sigma=spectral_init_sigma,
                    init_gamma=spectral_init_gamma,
                    max_sigma=spectral_max_sigma,
                    alpha=spectral_alpha,
                    pad_factor=spectral_pad_factor,
                    crop_quantile=spectral_crop_quantile,
                    use_local_branch=spectral_use_local_branch,
                )
            if mode == "separable":
                # 機構A: R 本の 1×K・K×1 分離の和 (rank-R)。ガウス的カーネルなら
                # 2D=1D×1D で AGD を保存しつつ、R を上げると非分離構造 (境界) を回収。
                if SeparableDWConv is None:
                    raise ImportError(
                        "separable_dw.SeparableDWConv を import できません"
                    )
                return SeparableDWConv(ch, kernel_size=k, rank=separable_rank)
            if mode == "gauss_deriv":
                # ガウス微分基底: カーネルを W=H(σ)·A·H(σ)ᵀ に構造的に閉じ込め、RF スケールを
                # σ のみに縛る(local 枝・gamma 無し = ガウス性がハード制約)。カーネルサイズは
                # stage の max_sigma を ±3σ 張れる大きさに広げ、σ で大 RF を作れるようにする。
                if GaussianDerivativeDW is None:
                    raise ImportError(
                        "gaussian_derivative_dw.GaussianDerivativeDW を import できません"
                    )
                gk = gauss_deriv_kernel_size(spectral_max_sigma, base_k=k)
                return GaussianDerivativeDW(
                    ch,
                    kernel_size=gk,
                    order=gauss_deriv_order,
                    init_sigma=spectral_init_sigma,
                    max_sigma=spectral_max_sigma,
                )
            if mode == "gauss_pyramid":
                # 多スケール純ガウス: 枝ごとに σ を増加 (a1<a2<a3)。枝は kernel_size k(7/9/11)で
                # 識別し idx=0/1/2、σ_branch = init_sigma·growth^idx。カスケードのガウス半群で
                # 実効σはさらに増大。境界は pointwise(v-conv) の DoG に任せる(local枝/gamma なし)。
                if GaussianPyramidDW is None:
                    raise ImportError(
                        "gaussian_pyramid_dw.GaussianPyramidDW を import できません"
                    )
                branch_idx = {7: 0, 9: 1, 11: 2}.get(k, 0)
                sigma_branch = float(spectral_init_sigma) * (
                    float(gauss_pyramid_growth) ** branch_idx
                )
                return GaussianPyramidDW(
                    ch,
                    kernel_size=k,
                    init_sigma=sigma_branch,
                    max_sigma=spectral_max_sigma,
                    alpha=spectral_alpha,
                    pad_factor=spectral_pad_factor,
                    crop_quantile=spectral_crop_quantile,
                    freeze_scale=gauss_freeze_scale,
                )
            if mode == "spectral_gaussian":
                return SpectralGaussianDW(
                    ch,
                    kernel_size=k,
                    init_sigma=spectral_init_sigma,
                    init_gamma=spectral_init_gamma,
                    max_sigma=spectral_max_sigma,
                    use_local_branch=spectral_use_local_branch,
                )
            if mode == "adaptive":
                return ContentAdaptiveDW(
                    ch, kernel_size=k, dilations=tuple(adaptive_dilations)
                )
            return DilatedDWConv(ch, kernel_size=k)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        # depthwise conv を学習可能 dilation 版に差し替え (キー a1.2.weight/bias は不変)
        self.a1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 1),
            nn.GELU(),
            _dw(dim // 4, 7),
        )
        self.v1 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, 1),
            nn.GELU(),
            _dw(dim // 2, 9),
        )
        self.v2 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v21 = nn.Conv2d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv2d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            _dw(dim * 3 // 4, 11),
        )
        self.v3 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv2d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv2d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.dim = dim

    def forward(self, x):
        x = self.norm1(x)
        x_split = torch.split(x, self.dim // 4, dim=1)

        a = self.a1(x_split[0])
        mul = self.v11(a * self.v1(x_split[0]))
        x1 = self.conv3_1(self.v12(x_split[1])) + a
        x1 = torch.cat([x1, mul], dim=1)

        x1 = self.norm2(x1)
        a = self.a2(x1)
        mul = self.v21(a * self.v2(x1))
        x2 = self.conv3_2(self.v22(x_split[2])) + self.proj2(a)
        x2 = torch.cat([x2, mul], dim=1)

        x2 = self.norm3(x2)
        a = self.a3(x2)
        mul = self.v31(a * self.v3(x2))
        x3 = self.conv3_3(self.v32(x_split[3])) + self.proj3(a)
        return torch.cat([x3, mul], dim=1)


# ──────────────────────────────────────────────
# DCNv3 代替: 標準 depthwise grouped conv
# キー名を DCNv3 と合わせることで非 dcn 重みの転移を妨げない。
# ──────────────────────────────────────────────
class _DepthwiseConvCompat(nn.Module):
    """
    DCNv3 と同じフォワードシグネチャを持ち、
    channels_last (B, H, W, C) 入力を受け取る標準 depthwise conv。
    """

    def __init__(
        self, channels, kernel_size=3, stride=1, pad=1, dilation=1, group=None, **kwargs
    ):
        super().__init__()
        groups = group or max(1, channels // 8)
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = nn.LayerNorm(channels)

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, a=0.01)

    def forward(self, x):
        # x: (B, H, W, C) channels_last
        x = x.permute(0, 3, 1, 2)  # → (B, C, H, W)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # → (B, H, W, C)
        return self.norm(x)


# ──────────────────────────────────────────────
# Block (オリジナルと同じキー構造)
# ──────────────────────────────────────────────
class Block(nn.Module):
    def __init__(
        self,
        dim,
        drop=0.0,
        drop_path=0.0,
        mlp_ratio=4,
        layer_scale_init_value=1e-5,
        adaptive_dw=False,
        adaptive_dilations=(1, 2, 4, 8),
        spectral_dw=False,
        spectral_init_sigma=1.0,
        spectral_init_gamma=0.0,
        spectral_max_sigma=32.0,
        spectral_use_local_branch=True,
        spectral_alpha=2.0,
        spectral_pad_factor=0.0,
        spectral_crop_quantile=0.0,
        spectral_num_gaussians=3,
        separable_rank=1,
        gauss_deriv_order=2,
        gauss_pyramid_growth=1.6,
        gauss_freeze_scale=False,
        dw_mode="dense",
        **kwargs,
    ):
        super().__init__()
        self.attn = ConvMod(
            dim,
            adaptive_dw=adaptive_dw,
            adaptive_dilations=adaptive_dilations,
            spectral_dw=spectral_dw,
            spectral_init_sigma=spectral_init_sigma,
            spectral_init_gamma=spectral_init_gamma,
            spectral_max_sigma=spectral_max_sigma,
            spectral_use_local_branch=spectral_use_local_branch,
            spectral_alpha=spectral_alpha,
            spectral_pad_factor=spectral_pad_factor,
            spectral_crop_quantile=spectral_crop_quantile,
            spectral_num_gaussians=spectral_num_gaussians,
            separable_rank=separable_rank,
            gauss_deriv_order=gauss_deriv_order,
            gauss_pyramid_growth=gauss_pyramid_growth,
            gauss_freeze_scale=gauss_freeze_scale,
            dw_mode=dw_mode,
        )
        self.mlp = MLPLayer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = build_norm_layer(dim, "LN")
        self.norm2 = build_norm_layer(dim, "LN")
        # 本物の DCNv3 (純 PyTorch 版・コンパイル不要)。
        # 元の models/UniConvNet.py と同じパラメータ・キー名なので
        # 事前学習の dcn 重み (dw_conv/offset/mask/input_proj/output_proj) が転移される。
        self.dcn = DCNv3_pytorch(
            channels=dim,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=dim // 8,
            offset_scale=1.0,
            act_layer="GELU",
            norm_layer="LN",
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.attn(x)
        )
        x = x.permute(0, 2, 3, 1)  # → channels_last
        x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x.permute(0, 3, 1, 2)  # → channels_first


# ──────────────────────────────────────────────
# UniConvNet 本体
# ──────────────────────────────────────────────
class UniConvNet(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=(2, 2, 8, 2),
        dims=(64, 128, 256, 512),
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        drop=0.0,
        adaptive_dw=False,
        adaptive_dilations=(1, 2, 4, 8),
        spectral_dw=False,
        spectral_use_local_branch=True,
        spectral_init_sigma=1.0,
        spectral_init_gamma=0.0,
        spectral_max_sigma=(32.0, 24.0, 12.0, 6.0),
        spectral_alpha=2.0,
        spectral_pad_factor=0.0,
        spectral_crop_quantile=0.0,
        spectral_num_gaussians=3,
        separable_rank=1,
        gauss_deriv_order=2,
        gauss_pyramid_growth=1.6,
        gauss_freeze_scale=False,
        dw_mode="dense",
    ):
        super().__init__()

        # spectral_max_sigma / spectral_init_sigma は stage 別 (深層ほど特徴マップが
        # 小さいので小さく)。スカラー/短いリストは 4 stage に展開 (足りない分は末尾値で埋める)。
        def _to4(v):
            if not isinstance(v, (list, tuple)):
                return [float(v)] * 4
            v = [float(x) for x in v]
            return (v + [v[-1]] * 4)[:4]

        spectral_max_sigma = _to4(spectral_max_sigma)
        spectral_init_sigma = _to4(spectral_init_sigma)

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, 3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(drop),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], 3, stride=2, padding=1),
                )
            )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        adaptive_dw=adaptive_dw,
                        adaptive_dilations=adaptive_dilations,
                        spectral_dw=spectral_dw,
                        spectral_use_local_branch=spectral_use_local_branch,
                        spectral_init_sigma=spectral_init_sigma[i],
                        spectral_init_gamma=spectral_init_gamma,
                        spectral_max_sigma=spectral_max_sigma[i],
                        spectral_alpha=spectral_alpha,
                        spectral_pad_factor=spectral_pad_factor,
                        spectral_crop_quantile=spectral_crop_quantile,
                        spectral_num_gaussians=spectral_num_gaussians,
                        separable_rank=separable_rank,
                        gauss_deriv_order=gauss_deriv_order,
                        gauss_pyramid_growth=gauss_pyramid_growth,
                        gauss_freeze_scale=gauss_freeze_scale,
                        dw_mode=dw_mode,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        return self.head(self.forward_features(x))
