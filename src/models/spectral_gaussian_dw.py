# coding:utf-8
"""
SpectralGaussianDW — 周波数領域ガウスで受容野(RF)を制御する depthwise 畳み込み。

狙い:
  dilation のように空間を疎サンプリングして大RFを"擬似的に"作るのではなく、
  周波数領域でガウス低域通過 H(ω)=exp(-σ²|ω|²/2) を掛けて大RFを直接作る。

  * ガウスの双対性 : 空間ガウス幅 σ ⇄ 周波数で幅 ∝ 1/σ。大RF = 狭い低域通過。
  * 低コスト       : FFT で O(N log N)。RF の大小に依らずコスト一定
                     (dilation の gridding 由来の ERF 不安定も無い)。
  * 学習可能 RF    : σ を学習 → 必要なだけ RF を伸ばす。各層/各チャネルで別 σ を
                     持てるので「U-Net 各層で RF が異なる」を構造的に表現できる。
  * AGD と整合     : ERF が定義からガウス (論文の Aggregated Gaussian 観察と一致)。

構成 (事前学習転移 + identity init):
  out = local_conv(x; weight,bias)  +  gamma * spectral_gauss(x; σ)
  * local_conv は元の depthwise (weight/bias は nn.Conv2d と同形・同キー a*.2.weight/bias
    → 事前学習重みが転移)。局所/高周波成分を担当。
  * spectral 枝は per-channel 学習 σ のガウス低域通過 (循環畳み込み, GFNet 流に
    パディング無し)。広域/低周波の集約を担当。
  * gamma 初期=0 → 学習開始時は元の conv と厳密一致 (転移を壊さない)。
    プローブ(スクラッチ学習)では init_gamma=1 で最初から spectral 枝を有効にする。

注意:
  * FFT は循環畳み込み → 境界で巻き込みが起きる (GFNet と同じ既知の割り切り)。
    実モデル特徴マップで問題なら将来 reflect パディングを足す。
  * 純ガウス低域通過は高周波を捨てる(ボケ)。local_conv 枝と住み分けることで補完。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGaussianDW(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=7,
        init_sigma=1.0,
        init_gamma=0.0,
        max_sigma=64.0,
        use_local_branch=True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size は奇数のみ対応"
        self.channels = channels
        self.kernel_size = kernel_size
        self.max_sigma = float(max_sigma)
        self.target_sigma = float(init_sigma)  # σ-warmup の到達目標 (= 初期 σ)
        self.use_local_branch = bool(use_local_branch)

        # 局所枝 (nn.Conv2d depthwise と同形・同キー → 事前学習重み転移)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 周波数ガウス枝: per-channel σ (log 空間で常に正)、gate gamma
        self.log_sigma = nn.Parameter(
            torch.full((channels,), float(math.log(init_sigma)))
        )
        self.gamma = nn.Parameter(torch.full((channels,), float(init_gamma)))

        # σ-warmup 用の実効 σ 上限キャップ。warmup 中に start(小)→max_sigma へ漸増し、
        # 実効 σ = min(clamp(exp(log_sigma), .., max_sigma), sigma_cap)。
        # init_sigma を target に置いて cap を開けば「σ を小→大に育てる」coarse-to-fine。
        # persistent=False: state_dict に含めない → 既存チェックポイントの load を壊さない。
        self.register_buffer(
            "sigma_cap", torch.tensor(float(max_sigma)), persistent=False
        )

    # ── 現在の σ (clamp 済み + warmup cap, px) ──
    def current_sigma(self):
        sigma = torch.clamp(torch.exp(self.log_sigma), 1e-3, self.max_sigma)
        return torch.minimum(sigma, self.sigma_cap)  # (C,)

    @torch.no_grad()
    def set_sigma_cap(self, value):
        """実効 σ の上限を設定 (max_sigma を超えない)。σ-warmup スケジューラから呼ぶ。"""
        self.sigma_cap.fill_(min(float(value), self.max_sigma))

    def _spectral(self, x):
        B, C, H, W = x.shape
        sigma = self.current_sigma().view(1, C, 1, 1)
        # 周波数座標 (cycles/pixel) → 角周波数 2π f。σ は px 単位の空間ガウス標準偏差。
        fy = torch.fft.fftfreq(H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        fx = torch.fft.rfftfreq(W, device=x.device, dtype=x.dtype).view(
            1, 1, 1, W // 2 + 1
        )
        w2 = (2 * math.pi) ** 2 * (fy**2 + fx**2)  # (1,1,H,Wf)
        env = torch.exp(-0.5 * (sigma**2) * w2)  # (1,C,H,Wf) broadcast
        Xf = torch.fft.rfft2(x, norm="ortho")
        return torch.fft.irfft2(Xf * env, s=(H, W), norm="ortho")

    def forward(self, x):
        spec = self._spectral(x)
        if not self.use_local_branch:
            return spec

        k = self.kernel_size
        local = F.conv2d(
            x, self.weight, self.bias, padding=k // 2, groups=self.channels
        )
        return local + self.gamma.view(1, -1, 1, 1) * spec

    # ── RF の広がり (= σ そのもの, px)。正則化ターゲットにも使える ──
    def rf_spread(self):
        return self.current_sigma().mean()

    def mean_sigma(self):
        return float(self.current_sigma().mean().detach())

    def mean_gamma(self):
        return float(self.gamma.abs().mean().detach())
