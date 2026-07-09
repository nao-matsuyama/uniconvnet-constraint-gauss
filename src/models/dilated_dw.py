# coding:utf-8
"""
DilatedDWConv — 学習可能な「連続 dilation」を持つ depthwise 畳み込み。

狙い:
  * 受容野(RF)を学習過程で可変にする   … dilation を学習パラメータにする
  * スパース計算                       … タップ数は k² 固定のまま dilation で広域をカバー
                                          (RF が広がっても計算量は増えない = 構造的スパース)
  * 事前学習重みの転移                 … weight/bias のキー名を nn.Conv2d と一致させる

連続 dilation の実現:
  dilation d = exp(log_dilation) (>=1) を floor/ceil の 2 つの整数 dilation の
  線形補間で表現する:
      out = (1 - frac) * conv(x; dilation=d_floor) + frac * conv(x; dilation=d_ceil)
      frac = d - d_floor   (∈ [0,1), log_dilation について微分可能)
  整数 dilation の conv は cuDNN がそのまま高速処理する。
  init_dilation=1 (log_dilation=0) では d_floor=1, frac=0 となり、
  元の dense depthwise conv と数値的に一致する。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedDWConv(nn.Module):
    def __init__(self, channels, kernel_size, init_dilation=1.0, max_dilation=8.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size は奇数のみ対応"
        self.channels = channels
        self.kernel_size = kernel_size
        self.max_dilation = float(max_dilation)

        # nn.Conv2d(depthwise) と同じ形・キー名 → 事前学習重みが転移される
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 学習可能 dilation (log 空間で保持し、常に正)
        self.log_dilation = nn.Parameter(torch.tensor(float(math.log(init_dilation))))

    # ── 現在の dilation (clamp 済み) ──
    def current_dilation(self):
        return torch.clamp(torch.exp(self.log_dilation), 1.0, self.max_dilation)

    def forward(self, x):
        k = self.kernel_size
        d = self.current_dilation()

        d_floor = max(1, int(math.floor(d.item())))
        d_ceil = d_floor + 1
        frac = d - d_floor  # tensor, grad は log_dilation へ流れる
        frac_val = float(frac.detach())

        # stride=1, padding=dilation*(k-1)/2 で出力サイズは入力と同じ (k 奇数)
        pad_f = d_floor * (k - 1) // 2
        out_f = F.conv2d(x, self.weight, self.bias, stride=1,
                         padding=pad_f, dilation=d_floor, groups=self.channels)

        # 推論時 (勾配不要) で dilation がほぼ整数なら 2本目を省略 → 計算半減。
        # 学習時は frac の勾配が必要なので常に2本計算する。
        if (not self.training) and frac_val < 1e-6:
            return out_f

        pad_c = d_ceil * (k - 1) // 2
        out_c = F.conv2d(x, self.weight, self.bias, stride=1,
                         padding=pad_c, dilation=d_ceil, groups=self.channels)
        return (1.0 - frac) * out_f + frac * out_c

    # ── 有効 RF の広がり (weight エネルギー重み付き RMS 半径) ──
    # ERF 正則化のターゲットに使う。log_dilation と weight について微分可能。
    def rf_spread(self):
        k = self.kernel_size
        d = self.current_dilation()
        coords = torch.arange(k, device=self.weight.device, dtype=self.weight.dtype) \
            - (k - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        r2 = (xx ** 2 + yy ** 2) * (d ** 2)              # dilation でスケールした各タップの r^2

        w2 = (self.weight.abs().mean(dim=0).squeeze(0)) ** 2   # (k,k) チャネル平均エネルギー
        w2 = w2 / (w2.sum() + 1e-8)
        return torch.sqrt((w2 * r2).sum() + 1e-8)
