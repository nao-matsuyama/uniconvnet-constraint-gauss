"""
UniConvNet-T をエンコーダーとした U-Net セグメンテーションモデル

ネットワーク構造:
  入力 (B, 3, H, W)
    ↓ stem (stride 4)
  Stage1: (B, 64,  H/4,  W/4)   ─── skip1
    ↓ downsample (stride 2)
  Stage2: (B, 128, H/8,  W/8)   ─── skip2
    ↓ downsample (stride 2)
  Stage3: (B, 256, H/16, W/16)  ─── skip3
    ↓ downsample (stride 2)
  Stage4: (B, 512, H/32, W/32)  ← bottleneck

  デコーダー
  dec3: 512+256 → 256  (H/32 → H/16)
  dec2: 256+128 → 128  (H/16 → H/8)
  dec1: 128+64  →  64  (H/8  → H/4)
  head:  64 → num_classes (H/4 → H)
"""

import sys
import os
import importlib.util

# models_N0-XL フォルダ名にハイフンがあるため直接 import できないので動的ロード
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "uniconvnet_n0xl",
    os.path.join(_here, "models_N0-XL", "UniConvNet.py"),
)
_mod = importlib.util.module_from_spec(_spec)
# ops_dcnv3 を参照できるよう検索パスに追加
sys.path.insert(0, _here)
_spec.loader.exec_module(_mod)
UniConvNet = _mod.UniConvNet

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# エンコーダー: UniConvNet-T の 4 ステージを取り出す
# ─────────────────────────────────────────────
class UniConvNetEncoder(nn.Module):
    """
    UniConvNet-T (depths=[3,3,15,3], dims=[64,128,256,512]) を
    エンコーダーとして使い、4 段階の特徴マップを返す。
    """

    def __init__(self, pretrained_path: str | None = None):
        super().__init__()
        self.backbone = UniConvNet(
            depths=[3, 3, 15, 3],
            dims=[64, 128, 256, 512],
        )

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            state = ckpt.get("model", ckpt)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"[Encoder] 重みロード完了: {pretrained_path}")
            if missing:
                print(f"  missing keys  : {len(missing)}")
            if unexpected:
                print(f"  unexpected keys: {len(unexpected)}")

    def forward(self, x):
        """
        Returns:
            f1: (B,  64, H/4,  W/4)
            f2: (B, 128, H/8,  W/8)
            f3: (B, 256, H/16, W/16)
            f4: (B, 512, H/32, W/32)
        """
        feats = []
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            feats.append(x)
        return feats  # [f1, f2, f3, f4]


# ─────────────────────────────────────────────
# デコーダーブロック
# ─────────────────────────────────────────────
class DecoderBlock(nn.Module):
    """
    アップサンプル → skip connection と concat → Conv×2
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# メインモデル: UniConvNet-T U-Net
# ─────────────────────────────────────────────
class UniConvNetUNet(nn.Module):
    """
    UniConvNet-T をエンコーダーとした U-Net。

    Args:
        num_classes    : セグメンテーションクラス数 (2値なら 1)
        pretrained_path: uniconvnet_t_1k_224_ema.pth へのパス (省略可)
        decoder_channels: デコーダー各段の出力チャネル数
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained_path: str | None = None,
        decoder_channels: tuple[int, int, int] = (256, 128, 64),
    ):
        super().__init__()

        # エンコーダー (dims=[64, 128, 256, 512])
        self.encoder = UniConvNetEncoder(pretrained_path)
        enc_dims = [64, 128, 256, 512]

        d0, d1, d2 = decoder_channels

        # デコーダー
        # H/32 → H/16 : bottleneck(512) + skip3(256) → d0
        self.dec3 = DecoderBlock(enc_dims[3], enc_dims[2], d0)
        # H/16 → H/8  : d0 + skip2(128) → d1
        self.dec2 = DecoderBlock(d0, enc_dims[1], d1)
        # H/8  → H/4  : d1 + skip1(64)  → d2
        self.dec1 = DecoderBlock(d1, enc_dims[0], d2)

        # H/4 → H : stem が 4x ダウンサンプルしているので 4x アップサンプル
        self.head = nn.Sequential(
            nn.ConvTranspose2d(d2, d2, kernel_size=4, stride=4),
            nn.BatchNorm2d(d2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)  ← H, W は 32 の倍数を推奨
        Returns:
            (B, num_classes, H, W)
        """
        f1, f2, f3, f4 = self.encoder(x)

        d = self.dec3(f4, f3)
        d = self.dec2(d, f2)
        d = self.dec1(d, f1)
        out = self.head(d)

        # 万一サイズがずれた場合の安全弁
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return out


# ─────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="uniconvnet_t_1k_224_ema.pth",
                        help="学習済み重みファイルのパス")
    parser.add_argument("--classes", type=int, default=1,
                        help="セグメンテーションクラス数")
    parser.add_argument("--size", type=int, default=224,
                        help="テスト画像サイズ (正方形)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}")

    weights = args.weights if os.path.exists(args.weights) else None
    if weights is None:
        print(f"[警告] {args.weights} が見つかりません。重みなしで初期化します。")

    print("モデルを構築中...")
    model = UniConvNetUNet(num_classes=args.classes, pretrained_path=weights).to(device)
    model.eval()

    dummy = torch.randn(2, 3, args.size, args.size, device=device)
    with torch.no_grad():
        out = model(dummy)

    print(f"\n入力サイズ : {tuple(dummy.shape)}")
    print(f"出力サイズ : {tuple(out.shape)}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"総パラメータ数: {n_params:.1f} M")

    assert out.shape == (2, args.classes, args.size, args.size), \
        f"出力サイズが想定と異なります: {out.shape}"
    print("\n✓ 動作確認 OK")
