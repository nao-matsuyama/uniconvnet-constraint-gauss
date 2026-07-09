# coding:utf-8
"""
ContentAdaptiveDW — コンテンツ適応の depthwise 畳み込み。

「どこで・どれだけ受容野を広げるか」を画像内容から画素ごとに決める。

機構:
  * 1 つの depthwise カーネルを複数の固定 dilation {d1,d2,...} で適用 (重み共有)
  * 軽量ゲートが画素ごとに dilation 選択の重み g_d(x) を softmax で出力
  * 出力 = Σ_d g_d(x) * conv(x; dilation=d)
  → 曖昧で広域文脈が要る画素では大 dilation 枝が選ばれ、局所で十分な画素では
    小 dilation が選ばれる = "適切な拡大"(場所依存)。

期待 dilation マップ E[d](x) = Σ_d g_d(x)*d を可視化すれば「どこを広げたか」が分かる。
タップ数は各枝とも k^2 (スパース)。学習可能 dilation(単一スカラー)と違い、大 dilation 枝を
常に計算するので「広げないと学べない」鶏卵局所解に陥りにくい。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAdaptiveDW(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=(1, 4, 16), gate_hidden=16):
        super().__init__()
        assert kernel_size % 2 == 1
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = list(dilations)

        # 全 dilation で共有する depthwise カーネル
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 画素ごとの dilation 選択ゲート (入力依存)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, gate_hidden, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, len(self.dilations), 1),
        )
        # 可視化用: 直近の期待 dilation マップ (B,1,H,W) を保持
        self._exp_dil = None

    def _gate_weights(self, x):
        return F.softmax(self.gate(x), dim=1)  # (B, n_d, H, W)

    def forward(self, x):
        g = self._gate_weights(x)
        out = 0.0
        for i, d in enumerate(self.dilations):
            pad = d * (self.kernel_size - 1) // 2
            conv_d = F.conv2d(x, self.weight, self.bias, stride=1,
                              padding=pad, dilation=d, groups=self.channels)
            out = out + g[:, i:i + 1] * conv_d
        # 期待 dilation マップを記録 (勾配不要)
        with torch.no_grad():
            dvec = torch.tensor(self.dilations, dtype=g.dtype, device=g.device).view(1, -1, 1, 1)
            self._exp_dil = (g * dvec).sum(dim=1, keepdim=True).detach()
        return out

    def expected_dilation(self, x):
        """画素ごとの期待 dilation E[d](x) = Σ g_d(x)*d を返す (B,1,H,W)。"""
        g = self._gate_weights(x)
        dvec = torch.tensor(self.dilations, dtype=g.dtype, device=g.device).view(1, -1, 1, 1)
        return (g * dvec).sum(dim=1, keepdim=True)

    def mean_expected_dilation(self):
        """直近 forward の期待 dilation の平均 (スカラー)。"""
        return float(self._exp_dil.mean()) if self._exp_dil is not None else float("nan")
