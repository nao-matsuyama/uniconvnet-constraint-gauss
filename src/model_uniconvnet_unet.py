# coding:utf-8
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))

try:
    from models.UniConvNet import UniConvNet
except ImportError:
    from model_pytorch import Unet2D as UniConvNet


class UpBlock(nn.Module):
    """U-Netデコーダ用のアップサンプリング＆スキップ結合ブロック"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 異なるステージ間のサイズ微差を吸収するためのパディング補正
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UniConvNet_UNet_13CH_Complete(nn.Module):
    """UniConvNetをバックボーンにした、骨シンチセグメンテーション（13クラス）用U-Net"""

    def __init__(
        self,
        num_classes=13,
        adaptive_dw=False,
        adaptive_dilations=(1, 2, 4, 8),
        spectral_dw=False,
        spectral_init_sigma=1.0,
        spectral_init_gamma=0.0,
        spectral_max_sigma=(32.0, 24.0, 12.0, 6.0),
        spectral_use_local_branch=True,
        spectral_alpha=2.0,
        spectral_pad_factor=0.0,
        spectral_crop_quantile=0.0,
        spectral_num_gaussians=3,
        separable_rank=1,
        gauss_deriv_order=2,
        dw_mode="dense",
    ):
        super().__init__()
        # UniConvNet-T の設定で初期化 (depths=[3,3,15,3], dims=[64,128,256,512])
        self.backbone = UniConvNet(
            depths=[3, 3, 15, 3],
            dims=[64, 128, 256, 512],
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
            dw_mode=dw_mode,
        )
        dims = [64, 128, 256, 512]

        # 2. U-Netのデコーダ（拡大パス）を定義
        # 下位ステージからのスキップ結合を受け取りつつ、解像度を引き上げる
        self.up3 = UpBlock(in_ch=dims[3], skip_ch=dims[2], out_ch=dims[2])  # 512 -> 256
        self.up2 = UpBlock(in_ch=dims[2], skip_ch=dims[1], out_ch=dims[1])  # 256 -> 128
        self.up1 = UpBlock(in_ch=dims[1], skip_ch=dims[0], out_ch=dims[0])  # 128 -> 64

        # 3. 最外周（ステム直後）のスキップ結合用ブロック
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(dims[0], dims[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
        )

        # 4. 最終出力（骨マスクの13クラスへマッピング、サイズを完全に入力画像と一致させる）
        self.final_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(dims[0] // 2, num_classes, kernel_size=1),
        )

        print(
            f"🚀 骨シンチセグメンテーション用 UniConvNet-U-Net (出力: {num_classes}ch) をビルドしました。"
        )

    def forward(self, x):
        # 元の画像サイズを記録 [H, W]
        orig_size = x.shape[2:]

        # --- エントリ（ステム）層 ---
        # 最初のdownsample_layerを適用 (2回連続stride=2のConvがあるため、解像度は1/4になる)
        s0 = self.backbone.downsample_layers[0](x)
        s0 = self.backbone.stages[0](s0)  # Stage 0 出力 (1/4サイズ)

        # --- 各ステージのフォワード (U-Netのエンコーダ・縮小パス) ---
        # Stage 1 (1/8サイズ)
        s1 = self.backbone.downsample_layers[1](s0)
        s1 = self.backbone.stages[1](s1)

        # Stage 2 (1/16サイズ)
        s2 = self.backbone.downsample_layers[2](s1)
        s2 = self.backbone.stages[2](s2)

        # Stage 3 (1/32サイズ: 最下層のボトルネック)
        s3 = self.backbone.downsample_layers[3](s2)
        s3 = self.backbone.stages[3](s3)

        # --- U-Netデコーダ（拡大パス & スキップ結合） ---
        x_up = self.up3(s3, s2)  # Stage 3 と Stage 2 を結合
        x_up = self.up2(x_up, s1)  # Stage 1 と結合
        x_up = self.up1(x_up, s0)  # Stage 0 と結合

        # 最外周の解像度復元
        x_up = self.up0(x_up)

        # 最終クラスマッピング
        logits = self.final_head(x_up)

        # 念のため、モデルの出力画像サイズを入力画像と完全に一致させる補正
        if logits.shape[2:] != orig_size:
            logits = F.interpolate(
                logits, size=orig_size, mode="bilinear", align_corners=True
            )

        return logits


# 既存のtrain.pyでの呼び出し名に統合
UniConvNet_UNet_13CH = UniConvNet_UNet_13CH_Complete
