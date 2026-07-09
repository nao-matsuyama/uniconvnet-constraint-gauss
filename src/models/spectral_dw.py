# coding:utf-8
"""
SpectralDW — 周波数領域 depthwise 畳み込み + 動的スペクトル切り出し (RFFT truncation)。

狙い (SpectralGaussianDW との違い):
  SpectralGaussianDW は「常にフルサイズ」で rfft2→ガウス乗算→irfft2 する。RF の大小に
  依らずスペクトル全域を扱うので、点単位乗算コストは一定。
  本モジュールは **受容野 σ が大きいほどスペクトルを小さく切り詰める**。大きく滑らかな
  (広RF = 低周波集中) カーネルは低周波にエネルギーが集中するため、高周波帯を捨てても
  情報損失が小さい。切り出し後の帯域 H'×W' で点単位複素乗算 → コスト ∝ η² ≈ (α/σ)²。

パイプライン (タスク仕様):
    [A (B,C,H,W)] ─RFFT2D─► Â (B,C,H,⌊W/2⌋+1) ─Crop(η)─► Â' (B,C,H',W')
                          Ŵ(σ) ─Crop(η)─► Ŵ' (C,H',W')
    Ŷ' = Â' ⊙ Ŵ'  ─ゼロ埋め戻し─► Ŷ (B,C,H,⌊W/2⌋+1) ─IRFFT2D(s=H,W)─► Y (B,C,H,W)

切り出し比率:
    η = clamp(α / σ_ref, η_min, 1)          (α: チューニング用ハイパラ)
    H' = round(η·H),  W' = round(η·(⌊W/2⌋+1))
  σ_ref は層内チャネルの **最小 σ**(= 周波数で最も広い = 帯域を一番必要とするチャネル)。
  こうすると帯域は「一番シャープなチャネル」に合わせて確保され、全チャネルで失う
  エネルギーが最小 = AGD をほぼ厳密に保つ。層全体が滑らか (全 σ 大) のときだけ縮む。

rfft2 の低周波帯の取り方 (fftshift しない):
  * H 軸 (dim=-2) は 0..H-1 に正/負両方の周波数。低周波を残すには
    **先頭 ⌈H'/2⌉ 行 (0..+f) と末尾 ⌊H'/2⌋ 行 (−f)** を残す (共役対称を保つ)。
  * W 軸 (dim=-1) は rfft なので 0..Nyquist。先頭 W' 列を残す。
  * 復元は irfft2 に直接小テンソルを渡すと H 軸の正負が混ざるため、必ず
    フルサイズ (H,⌊W/2⌋+1) にゼロ埋めしてから irfft2(s=(H,W)) する。

重みの表現:
  * 既定 (free_weight=False, 推奨): 空間ガウス幅 σ を解析的に周波数化した
    Ŵ(σ)=exp(−2π²σ²·f²)。(i) ERF が定義からガウス = AGD をネイティブ保証、
    (ii) 切り出しで失うエネルギー最小、(iii) パラメータが σ のみで軽い。
  * free_weight=True: 自由な空間カーネル weight(C,1,k,k) を rfft2 して Ŵ を得る。
    表現力は上がるが AGD 保証は失う。切り出し帯域だけ rfft することはできないので
    (自由カーネルは全帯域を持つ) この場合フルサイズ rfft のみ切り出す。

構成 (事前学習転移 + identity init, SpectralGaussianDW と同じ流儀):
    out = local_conv(x; weight,bias) + gamma * spectral_trunc(x; σ)
  * local_conv は元の depthwise (weight/bias が a*.2.weight/bias と同形・同キー → 転移)。
  * gamma 初期=0 → 学習開始時は元の conv と厳密一致 (回帰なし)。
  * use_local_branch=False で純スペクトル (境界/コストのベンチ用)。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralDW(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=7,
        init_sigma=1.0,
        init_gamma=0.0,
        max_sigma=64.0,
        alpha=2.0,
        min_keep=4,
        free_weight=False,
        use_local_branch=True,
        pad_factor=0.0,
        crop_quantile=0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size は奇数のみ対応"
        self.channels = channels
        self.kernel_size = kernel_size
        self.max_sigma = float(max_sigma)
        self.target_sigma = float(init_sigma)  # σ-warmup の到達目標 (= 初期 σ)
        # 切り出し比 η = α/σ のスケール。α=2 で境界周波数のガウス値 exp(-2π²)≈3e-9 = 実質
        # 無損失(AGD 厳密, cost∝η²≈(2/σ)²)。α=1 は境界 0.7% で僅かにリンギング(cost 1/4)。
        self.alpha = float(alpha)
        self.min_keep = int(min_keep)  # 各軸で最低限残す周波数ビン数
        self.free_weight = bool(free_weight)
        self.use_local_branch = bool(use_local_branch)
        # pad_factor>0: rfft2 前に reflect パディング(境界の巻き込みアーティファクト対策)。
        #   入力を各軸 ±round(pad_factor·辺長) reflect し、処理後に中央を切り出す。0で無効(循環)。
        self.pad_factor = float(pad_factor)
        # crop_quantile>0: 切り出しの σ_ref を層内 min でなく分位点にする(既定0=min)。
        #   σ が per-channel に分化して 1ch でも σ 極小になると η→1 で truncation が死ぬのを防ぐ。
        self.crop_quantile = float(crop_quantile)

        # 局所枝 (nn.Conv2d depthwise と同形・同キー → 事前学習重み転移)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 周波数ガウス枝: per-channel σ (log 空間で常に正)、gate gamma
        self.log_sigma = nn.Parameter(
            torch.full((channels,), float(math.log(init_sigma)))
        )
        self.gamma = nn.Parameter(torch.full((channels,), float(init_gamma)))

        if self.free_weight:
            # 自由スペクトル用の空間カーネル (rfft して Ŵ を得る)。転移用の weight とは別。
            self.spectral_weight = nn.Parameter(
                torch.zeros(channels, 1, kernel_size, kernel_size)
            )
            with torch.no_grad():
                # 中心タップ=1 の恒等近傍で開始 (Ŵ≈1 → ボケない)
                self.spectral_weight[:, 0, kernel_size // 2, kernel_size // 2] = 1.0

        # σ-warmup 用の実効 σ 上限キャップ (SpectralGaussianDW と互換)。
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

    # ── 動的切り出しサイズ (η = α/σ_ref, σ_ref = 層内 min または分位点 σ) ──
    def _crop_sizes(self, H, Wf):
        # 切り出しサイズは σ の離散関数 → grad は流さない (DilatedDWConv の floor/ceil と同流儀)。
        sig = self.current_sigma().detach()
        if self.crop_quantile > 0:
            sigma_ref = float(torch.quantile(sig, self.crop_quantile))
        else:
            sigma_ref = float(sig.min())
        eta = min(1.0, self.alpha / max(sigma_ref, 1e-6))
        Hp = int(round(eta * H))
        Wp = int(round(eta * Wf))
        # 各軸で最低ビン数を確保 (小さすぎる帯域で情報を落としすぎない)。H は偶奇両帯を残す。
        Hp = max(min(self.min_keep, H), min(Hp, H))
        Wp = max(min(self.min_keep, Wf), min(Wp, Wf))
        return Hp, Wp

    def _gauss_env(self, fy, fx, dtype, device):
        """周波数座標 (cycles/px) から per-channel ガウス Ŵ(σ)=exp(−2π²σ²(fy²+fx²))。

        Returns: (C, Hp, Wp) 実テンソル。
        """
        sigma = self.current_sigma().to(device=device, dtype=dtype).view(-1, 1, 1)
        f2 = (fy.view(1, -1, 1) ** 2) + (fx.view(1, 1, -1) ** 2)  # (1,Hp,Wp)
        return torch.exp(-2.0 * (math.pi**2) * (sigma**2) * f2)  # (C,Hp,Wp)

    def _spectral(self, x):
        # pad_factor>0 なら reflect パディングして循環畳み込みの巻き込みを抑える。
        # 処理は padded サイズで行い、最後に中央 (H,W) を切り出して返す。
        if self.pad_factor > 0 and not self.free_weight:
            B, C, H, W = x.shape
            ph = min(H - 1, int(round(self.pad_factor * H)))
            pw = min(W - 1, int(round(self.pad_factor * W)))
            if ph > 0 or pw > 0:
                xp = F.pad(x, (pw, pw, ph, ph), mode="reflect")
                yp = self._spectral_core(xp)
                return yp[:, :, ph : ph + H, pw : pw + W]
        return self._spectral_core(x)

    def _spectral_core(self, x):
        B, C, H, W = x.shape
        Wf = W // 2 + 1
        Xf = torch.fft.rfft2(x, norm="ortho")  # (B,C,H,Wf) 複素

        if self.free_weight:
            # 自由カーネルは全帯域を持つので切り出さず、フルサイズで乗算 (AGD 非保証モード)。
            Wsp = torch.fft.rfft2(
                self.spectral_weight, s=(H, W), norm="ortho"
            )  # (C,1,H,Wf)
            return torch.fft.irfft2(
                Xf * Wsp.squeeze(1).unsqueeze(0), s=(H, W), norm="ortho"
            )

        Hp, Wp = self._crop_sizes(H, Wf)
        top = (Hp + 1) // 2  # 先頭 ⌈H'/2⌉ 行 (DC + 正の周波数)
        bot = Hp // 2  # 末尾 ⌊H'/2⌋ 行 (負の周波数)

        # ── 低周波帯を切り出し (H 軸: 先頭 top + 末尾 bot 行、W 軸: 先頭 Wp 列) ──
        if bot > 0:
            Xband = torch.cat([Xf[:, :, :top, :Wp], Xf[:, :, H - bot :, :Wp]], dim=2)
        else:
            Xband = Xf[:, :, :top, :Wp]  # (B,C,Hp,Wp)

        # 切り出し帯域に対応する周波数座標 (px 単位, cycles/px)
        fy_full = torch.fft.fftfreq(H, device=x.device, dtype=x.dtype)  # (H,)
        fx = torch.fft.rfftfreq(W, device=x.device, dtype=x.dtype)[:Wp]  # (Wp,)
        if bot > 0:
            fy = torch.cat([fy_full[:top], fy_full[H - bot :]])  # (Hp,)
        else:
            fy = fy_full[:top]

        env = self._gauss_env(fy, fx, x.dtype, x.device)  # (C,Hp,Wp) 実
        Yband = Xband * env.unsqueeze(0)  # (B,C,Hp,Wp)

        # ── フルサイズにゼロ埋めで戻す (切り捨てた高周波 = 0) → irfft2 ──
        Yf = x.new_zeros(B, C, H, Wf, dtype=Xf.dtype)
        Yf[:, :, :top, :Wp] = Yband[:, :, :top, :]
        if bot > 0:
            Yf[:, :, H - bot :, :Wp] = Yband[:, :, top:, :]
        return torch.fft.irfft2(Yf, s=(H, W), norm="ortho")

    def forward(self, x):
        spec = self._spectral(x)
        if not self.use_local_branch:
            return spec

        k = self.kernel_size
        local = F.conv2d(
            x, self.weight, self.bias, padding=k // 2, groups=self.channels
        )
        return local + self.gamma.view(1, -1, 1, 1) * spec

    # ── 現在の切り出し比 η (診断/コスト見積り用) ──
    def current_eta(self):
        sigma_ref = float(self.current_sigma().min().detach())
        return min(1.0, self.alpha / max(sigma_ref, 1e-6))

    # ── RF の広がり (= σ そのもの, px)。ERF 正則化ターゲットにも使える ──
    def rf_spread(self):
        return self.current_sigma().mean()

    def mean_sigma(self):
        return float(self.current_sigma().mean().detach())

    def mean_gamma(self):
        return float(self.gamma.abs().mean().detach())
