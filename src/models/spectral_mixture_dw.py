# coding:utf-8
"""
SpectralMixtureDW — 多ガウス混合を周波数領域で表現する depthwise 畳み込み(機構C)。

背景 (なぜ「単一ガウス」から「多ガウス混合」へ):
  機構B SpectralDW は周波数包絡を **単一ガウス** Ŵ(f)=exp(−2π²σ²|f|²) で表す。だが
  AGD 検証 (erf_gmm_fit.py) が示すのは「RFA の ERF は単一ガウスでは甘く、N(=3)ガウスの
  **和** で綺麗に乗る = 集約ガウス(aggregated Gaussian)」こと。RFA は a1/a2/a3(k7/9/11)の
  3 スケールを集約するので、受容野は本質的に **複数スケールのガウスの重ね合わせ**。そこで
  周波数包絡そのものを K ガウスの混合にする:

      Ŵ_c(f) = Σ_{k=1}^K w_{c,k} · exp(−2π²σ_{c,k}²|f|²),   |f|² = f_y² + f_x²

  ガウスのフーリエ双対 FT{ガウス}=ガウス より、これは空間核が **ガウス混合**

      w_c(r) = Σ_k w_{c,k} · N(r; 0, σ_{c,k}² I)

  であることと厳密に等価。よって ERF は定義から集約ガウス = AGD をネイティブに、しかも
  単一ガウスより表現力高く保証する。K=1 は SpectralDW の単一ガウスに厳密一致(真の一般化)。

効率的計算の 3 つの要 (多ガウスでもコストを単一ガウス並みに保つ):
  (1) 分離 rank-K 構築 : 等方ガウスは軸方向に分離
        exp(−2π²σ²(f_y²+f_x²)) = exp(−2π²σ²f_y²) · exp(−2π²σ²f_x²)
      なので K 混合包絡は **rank-K** テンソル = K 個の外積の和。1D 指数
        G_y[c,k,·]=exp(−2π²σ²f_y²) (C,K,H'),  G_x[c,k,·] (C,K,W')
      から einsum で (C,H',W') を組む。超越関数(exp)評価は K·C·(H'+W') 回で、素朴な
      2D 評価 K·C·H'·W' より軽い。バッチ複素乗算は **1 回**(K 回でない)ので B 依存の
      主コストは単一ガウスと同一。
  (2) 振幅考慮の動的切り出し : 必要帯域は「まだ重みを持つ最もシャープな成分」で決まる。
      広い成分(σ大)は超低域に居るので、振幅が無視できるシャープ成分は帯域予算から外す:
        σ_ref = min{ σ_{c,k} : w_{c,k} ≥ τ },   η = clamp(α/σ_ref, η_min, 1)
      単一ガウスの η=α/σ より厳密にタイト。AGD 忠実度を保ったまま帯域を詰める。
  (3) DC 保存の凸混合 : w は softmax(Σ_k w=1, w≥0)。各ガウスの DC 利得=1 なので包絡は
      f=0 で 1 → spectral 枝は平均を保つ多スケール低域通過。残差 local + gamma·spectral に
      安定に足せる(gamma=0 初期で元 conv と厳密一致)。

rfft2 の低周波帯の取り方(SpectralDW と同じ, fftshift しない):
  * H 軸 (dim=-2): 先頭 ⌈H'/2⌉ 行(DC+正) と 末尾 ⌊H'/2⌋ 行(負)を残す(共役対称保持)。
  * W 軸 (dim=-1): rfft なので 0..Nyquist、先頭 W' 列を残す。
  * 復元は必ずフルサイズ (H,⌊W/2⌋+1) にゼロ埋め → irfft2(s=(H,W))。

構成 (事前学習転移 + identity init, SpectralDW と同流儀):
    out = local_conv(x; weight,bias) + gamma * spectral_mixture(x; {σ_k, w_k})
  * weight/bias は a*.2.weight/bias と同形・同キー → 事前学習 depthwise が転移。
  * 新規パラメータ log_sigma(C,K)/mix_logits(C,K)/gamma(C) は strict=False で吸収。
  * log_sigma という名前は train.py の σ 専用 param group(高lr/WD0)に自動で入る。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralMixtureDW(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=7,
        num_gaussians=3,
        init_sigma=1.0,
        init_spread=2.0,
        init_gamma=0.0,
        max_sigma=64.0,
        alpha=2.0,
        min_keep=4,
        weight_thresh=0.05,
        use_local_branch=True,
        pad_factor=0.0,
        crop_quantile=0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size は奇数のみ対応"
        assert num_gaussians >= 1, "num_gaussians は 1 以上"
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_gaussians = int(num_gaussians)
        self.max_sigma = float(max_sigma)
        self.target_sigma = float(init_sigma)  # σ-warmup の到達目標(= 初期 σ の中心)
        self.alpha = float(alpha)
        self.min_keep = int(min_keep)  # 各軸で最低限残す周波数ビン数
        # weight_thresh τ: 帯域予算に含める最小混合重み。振幅がこれ未満の成分は「無視できる」
        #   として σ_ref の計算から外す(→ シャープでも重み薄い成分に帯域を奪われない)。
        self.weight_thresh = float(weight_thresh)
        self.use_local_branch = bool(use_local_branch)
        # pad_factor>0: rfft2 前に reflect パディング(循環畳み込みの巻き込み対策)。0で無効。
        self.pad_factor = float(pad_factor)
        # crop_quantile>0: 切り出しの σ_ref を層内 min でなく分位点にする(既定0=min)。
        self.crop_quantile = float(crop_quantile)

        # 局所枝 (nn.Conv2d depthwise と同形・同キー → 事前学習重み転移)
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # 多ガウス枝: per-channel × per-component の σ(log 空間で常に正)と混合ロジット。
        # 初期 σ は init_sigma を中心に幾何級数で [init/spread, init*spread] へ広げ、
        # 最初から fine〜coarse の複数スケールを張る。混合ロジット=0 → 初期は等重み。
        K = self.num_gaussians
        if K == 1:
            factors = [1.0]
        else:
            logr = math.log(float(init_spread))
            factors = [math.exp(logr * (2.0 * i / (K - 1) - 1.0)) for i in range(K)]
        sig0 = [min(max(float(init_sigma) * f, 1e-2), self.max_sigma) for f in factors]
        log_sigma0 = torch.tensor([math.log(s) for s in sig0])  # (K,)
        self.log_sigma = nn.Parameter(
            log_sigma0.view(1, K).repeat(channels, 1)
        )  # (C,K)
        self.mix_logits = nn.Parameter(torch.zeros(channels, K))  # softmax→等重み初期
        self.gamma = nn.Parameter(torch.full((channels,), float(init_gamma)))

        # σ-warmup 用の実効 σ 上限キャップ(全成分に一様適用)。
        # persistent=False: state_dict に含めない → 既存 ckpt の load を壊さない。
        self.register_buffer(
            "sigma_cap", torch.tensor(float(max_sigma)), persistent=False
        )

    # ── 現在の σ (clamp 済み + warmup cap, px)。(C,K) ──
    def current_sigma(self):
        sigma = torch.clamp(torch.exp(self.log_sigma), 1e-3, self.max_sigma)
        return torch.minimum(sigma, self.sigma_cap)  # (C,K)

    # ── 混合重み w = softmax(mix_logits) (C,K)、Σ_k w=1, w≥0 ──
    def current_weights(self):
        return torch.softmax(self.mix_logits, dim=1)  # (C,K)

    # ── 実効 σ (混合の二次モーメント sqrt(Σ w σ²), px)。(C,) ERF 正則化ターゲット向け ──
    def effective_sigma(self):
        sigma = self.current_sigma()
        w = self.current_weights()
        return torch.sqrt((w * sigma**2).sum(dim=1) + 1e-12)  # (C,)

    @torch.no_grad()
    def set_sigma_cap(self, value):
        """実効 σ の上限を設定 (max_sigma を超えない)。σ-warmup スケジューラから呼ぶ。"""
        self.sigma_cap.fill_(min(float(value), self.max_sigma))

    # ── 帯域予算の基準 σ_ref: 有意成分(w≥τ)の最小 σ を per-ch で取り層内で集約 ──
    def _sigma_ref(self):
        sig = self.current_sigma().detach()  # (C,K)
        w = self.current_weights().detach()  # (C,K)
        # 各 ch の最大重み成分は必ず含める(全成分が τ 未満でも空集合にしない)。
        keep = (w >= self.weight_thresh) | (w >= w.max(dim=1, keepdim=True).values)
        big = torch.full_like(sig, self.max_sigma * 10.0)
        sig_ref_perch = torch.where(keep, sig, big).min(dim=1).values  # (C,)
        if self.crop_quantile > 0:
            return float(torch.quantile(sig_ref_perch, self.crop_quantile))
        return float(sig_ref_perch.min())

    # ── 動的切り出しサイズ (η = α/σ_ref) ──
    def _crop_sizes(self, H, Wf):
        # 切り出しサイズは σ の離散関数 → grad は流さない(SpectralDW と同流儀)。
        sigma_ref = self._sigma_ref()
        eta = min(1.0, self.alpha / max(sigma_ref, 1e-6))
        Hp = int(round(eta * H))
        Wp = int(round(eta * Wf))
        Hp = max(min(self.min_keep, H), min(Hp, H))
        Wp = max(min(self.min_keep, Wf), min(Wp, Wf))
        return Hp, Wp

    def _mixture_env(self, fy, fx, dtype, device):
        """周波数座標 (cycles/px) から per-channel 多ガウス混合包絡を分離 rank-K で構築。

        Ŵ_c(f) = Σ_k w_{c,k}·exp(−2π²σ_{c,k}²(f_y²+f_x²))
               = Σ_k w_{c,k}·[exp(−2π²σ²f_y²)]⊗[exp(−2π²σ²f_x²)]  (rank-K 外積和)
        Returns: (C, Hp, Wp) 実テンソル。
        """
        sigma = self.current_sigma().to(device=device, dtype=dtype)  # (C,K)
        w = self.current_weights().to(device=device, dtype=dtype)  # (C,K)
        sig2 = (sigma**2).unsqueeze(-1)  # (C,K,1)
        coef = 2.0 * (math.pi**2)
        fy2 = (fy.to(device=device, dtype=dtype) ** 2).view(1, 1, -1)  # (1,1,Hp)
        fx2 = (fx.to(device=device, dtype=dtype) ** 2).view(1, 1, -1)  # (1,1,Wp)
        Gy = torch.exp(-coef * sig2 * fy2)  # (C,K,Hp)
        Gx = torch.exp(-coef * sig2 * fx2)  # (C,K,Wp)
        Gyw = Gy * w.unsqueeze(-1)  # 混合重みを y 側に畳む (C,K,Hp)
        # rank-K 再構成: k を縮約して (C,Hp,Wp)。バッチ非依存 = 1/B の軽コスト。
        return torch.einsum("ckh,ckw->chw", Gyw, Gx)

    def _spectral(self, x):
        # pad_factor>0 なら reflect パディングして循環畳み込みの巻き込みを抑える。
        if self.pad_factor > 0:
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

        env = self._mixture_env(fy, fx, x.dtype, x.device)  # (C,Hp,Wp) 実
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
        return min(1.0, self.alpha / max(self._sigma_ref(), 1e-6))

    # ── RF の広がり(= 実効 σ, px)。ERF 正則化ターゲットにも使える ──
    def rf_spread(self):
        return self.effective_sigma().mean()

    def mean_sigma(self):
        return float(self.effective_sigma().mean().detach())

    def mean_gamma(self):
        return float(self.gamma.abs().mean().detach())

    # ── 有効成分数 (参加率 1/Σw²) の層平均。混合がいくつのガウスを実質使っているかの診断 ──
    def mean_num_effective(self):
        w = self.current_weights().detach()
        return float((1.0 / (w**2).sum(dim=1)).mean())
