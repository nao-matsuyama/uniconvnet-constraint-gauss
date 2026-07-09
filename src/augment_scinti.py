# coding:utf-8
"""
骨シンチ seg 用のオンライン データ拡張ラッパ。

現状 `ScintiMultiClassDataset` は正規化+3ch化のみで**拡張が一切無い**。学習ログでは
train loss が ~0.05 まで落ちて val Dice は 0.94 で頭打ち=**明確な過学習**。受容野(RF)は
このタスクの精度に無関係と確定済みなので、精度(特に worst 群の汎化)を底上げする最も定石な
一手が拡張。この wrapper を train 側の Subset だけに噛ませる(val は不変=既存の比較基盤と整合)。

拡張の設計(骨シンチの事前知識に合わせる):
  1. **左右スワップ付き水平反転**: 全身骨シンチはほぼ左右対称。ただし 13 クラスには
     8:R-Arm/9:L-Arm・10:R-Leg/11:L-Leg の**左右クラス**があるので、反転時はマスクの
     左右ラベルを入れ替える(8↔9, 10↔11)。肋骨/肩甲鎖骨/骨盤(5/6/7)は左右統合済みで反転不変。
     → 実効データを倍増できる強力な拡張。
  2. **小アフィン**(回転/スケール/平行移動): 姿勢・体格のばらつきを模擬。左右は入れ替えない。
     img は bilinear、mask は nearest で補間(ラベルを混ぜない)。はみ出しは背景(0)埋め。
  3. **強度ジッタ**(gamma/brightness/gaussian noise): シンチのカウントノイズ・露出差を模擬。img のみ。

既定は全て弱め。train.py の `--aug` で有効化(未指定なら素通し=従来と厳密一致)。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# 反転時に入れ替える左右ラベルのペア (13クラス: 8:R-Arm↔9:L-Arm, 10:R-Leg↔11:L-Leg)
LR_SWAP_PAIRS = ((8, 9), (10, 11))


def _hflip_with_lr_swap(img, mask, lr_pairs):
    """水平反転 + マスクの左右ラベル入れ替え。img (3,H,W), mask (H,W) long。"""
    img = torch.flip(img, dims=[-1])
    mask = torch.flip(mask, dims=[-1])
    if lr_pairs:
        swapped = mask.clone()
        for a, b in lr_pairs:
            swapped[mask == a] = b
            swapped[mask == b] = a
        mask = swapped
    return img, mask


def _random_affine(img, mask, max_rotate_deg, max_scale, max_translate, generator):
    """回転(±deg)・スケール(1±s)・平行移動(±t·辺)をランダムに適用。
    img=bilinear, mask=nearest, はみ出し=0(背景)。左右は入れ替えない(幾何変換のみ)。"""
    if max_rotate_deg <= 0 and max_scale <= 0 and max_translate <= 0:
        return img, mask
    device = img.device

    def _u(lo, hi):
        return (
            torch.rand(1, generator=generator, device=device) * (hi - lo) + lo
        ).item()

    ang = _u(-max_rotate_deg, max_rotate_deg) * torch.pi / 180.0
    scale = 1.0 + _u(-max_scale, max_scale)
    tx = _u(-max_translate, max_translate)  # normalized [-1,1] grid 座標での並進
    ty = _u(-max_translate, max_translate)

    cos, sin = torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))
    # affine_grid の theta は「出力→入力」座標写像。回転+等方スケールの逆変換を入れる。
    inv = 1.0 / scale
    theta = torch.tensor(
        [
            [cos * inv, -sin * inv, tx],
            [sin * inv, cos * inv, ty],
        ],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    c, h, w = img.shape
    grid = F.affine_grid(theta, size=(1, c, h, w), align_corners=False)
    img_o = F.grid_sample(
        img.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(0)
    mask_f = mask.to(torch.float32).unsqueeze(0).unsqueeze(0)
    mask_o = F.grid_sample(
        mask_f, grid, mode="nearest", padding_mode="zeros", align_corners=False
    )
    mask_o = mask_o.squeeze(0).squeeze(0).round().to(mask.dtype)
    return img_o, mask_o


def _intensity_jitter(img, gamma_mag, bright_mag, noise_std, generator):
    """img (3,H,W, [0,1]) に gamma/brightness/gaussian noise。mask は不変。"""
    device = img.device

    def _u(lo, hi):
        return (
            torch.rand(1, generator=generator, device=device) * (hi - lo) + lo
        ).item()

    if gamma_mag > 0:
        gamma = torch.exp(torch.tensor(_u(-gamma_mag, gamma_mag)))  # 幾何的に対称
        img = img.clamp(0, 1).pow(gamma)
    if bright_mag > 0:
        img = img + _u(-bright_mag, bright_mag)
    if noise_std > 0:
        img = (
            img + torch.randn(img.shape, generator=generator, device=device) * noise_std
        )
    return img.clamp(0, 1)


class AugmentedScintiDataset(Dataset):
    """train 用 Subset を包み、__getitem__ 毎にランダム拡張を適用する。

    Args:
        base: (img[3,H,W] float, mask[H,W] long) を返す Dataset/Subset。
        hflip_prob: 左右スワップ付き水平反転の確率(0で無効)。
        rotate_deg / scale / translate: 小アフィンの上限。
        gamma / brightness / noise: 強度ジッタの強さ(img のみ)。
        lr_pairs: 反転時に入れ替える左右ラベル対。
        seed: 乱数シード(worker ごとに index を混ぜて再現性と多様性を両立)。
    """

    def __init__(
        self,
        base,
        hflip_prob=0.5,
        rotate_deg=10.0,
        scale=0.1,
        translate=0.05,
        gamma=0.1,
        brightness=0.05,
        noise=0.0,
        lr_pairs=LR_SWAP_PAIRS,
        seed=42,
    ):
        self.base = base
        self.hflip_prob = hflip_prob
        self.rotate_deg = rotate_deg
        self.scale = scale
        self.translate = translate
        self.gamma = gamma
        self.brightness = brightness
        self.noise = noise
        self.lr_pairs = tuple(lr_pairs) if lr_pairs else ()
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        # epoch/worker をまたいで多様な、しかし再現可能な乱数系列。
        g = torch.Generator()
        g.manual_seed(
            (
                self.seed * 1_000_003
                + idx * 9_176
                + int(torch.randint(0, 2**31 - 1, (1,)).item())
            )
            % (2**31 - 1)
        )
        if self.hflip_prob > 0 and torch.rand(1, generator=g).item() < self.hflip_prob:
            img, mask = _hflip_with_lr_swap(img, mask, self.lr_pairs)
        img, mask = _random_affine(
            img, mask, self.rotate_deg, self.scale, self.translate, g
        )
        img = _intensity_jitter(img, self.gamma, self.brightness, self.noise, g)
        return img.contiguous(), mask.contiguous().long()
