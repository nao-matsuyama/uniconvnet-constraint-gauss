# coding:utf-8
"""
GaussianDerivativeDW — ガウス微分(スケールスペース)基底で受容野を張る depthwise 畳み込み。

狙い (なぜ「後付けの周波数ガウス枝」でなく「基底そのものをガウスに縛る」か):
  機構B/C (SpectralDW / SpectralMixtureDW) はガウス性を周波数包絡としてハードに課すが、
  構造が out = local_conv(x) + gamma·spectral(x) の **加法枝** で、gamma が自由・初期0。
  結果ネットは gamma≈0 に落として周波数枝を丸ごと無効化できてしまう(骨シンチで実測)。
  つまり「ガウス性の制約」が任意(オプトイン)で、外されうる。一方 use_local_branch=False の
  純ガウス(0次)は低域通過ボケでエッジを潰し境界が甘くなる(pure-spec で worst 境界悪化)。

  本モジュールは depthwise カーネルそのものを **学習可能スケール σ のガウスとその低次微分の
  張る空間** に構造的に閉じ込める:

      w_c(x, y) = Σ_{p,q=0}^{N} a_{c,pq} · ∂_x^p ∂_y^q G_{σ_c}(x, y)
                = Σ_{p,q} a_{c,pq} · h_p(x; σ_c) · h_q(y; σ_c)         (等方 → 軸分離)

  ここで h_n(·; σ) = ∂^n G_σ = He_n(·/σ)·exp(-·²/2σ²) は 1次元ガウス微分(エルミート×ガウス)。
  これにより:
    (1) 受容野の広がりは σ_c ただ一つが支配 → 「RF を広げる = σ を上げる」以外の空間スケール
        ノブが存在しない。ガウス性が RF 拡大の制約として構造的に結合する(本命の狙い)。
    (2) 0次ガウスの弱点(境界ボケ)を解消: 1次項=方向性エッジ, 2次項=リッジ/ブロブ。微分項が
        境界表現を担うので純ガウスのようには境界を潰さない。
    (3) AGD(ガウス ERF): 多項式×ガウスはガウス裾を持つので実効カーネルの包絡は σ で決まる
        ガウスのまま(疎サンプリングの dilated のような高周波複製ローブが出ない)。有限分散核の
        積み重ねは CLT で ERF がガウスへ収束 = AGD を構造的に保つ。微分項は「ガウス窓の内側の
        ゆらぎ」で裾を太らせない。
    (4) local 枝・gamma を **持たない**(演算子全体が基底の内側) = ガウス性がハード制約。

次数 N が「制約の強さ ↔ 表現力」の唯一のノブ:
    N=0 : 純ガウス(= 機構B pure-spec, 最強の制約・最弱の表現力)
    N=2 : 6/9 基底(1, ∂x, ∂y, ∂xx, ∂xy, ∂yy 相当) = エッジ/リッジ十分(既定)
    N→大: 無制約に漸近

効率的 forward (rank-(N+1) 分離):
  2D カーネル W = H A Hᵀ (H:(K,M) 基底行列, A:(M,M) 係数, M=N+1) は rank ≤ M。
    W[i,j] = Σ_m h_m[i] · B_m[j],   B = A Hᵀ  (B_m[j] = 各 m の水平フィルタ)
  なので K×K 密畳み込みでなく、M 本の (水平 1×K → 垂直 K×1) 分離畳み込みの和で計算できる:
    out = Σ_{m=0}^{N} conv_v( conv_h(x, B_m), h_m )
  コスト 2MK タップ (K=41,N=2 で 246 vs 密 1681)。σ, A の両方に微分可能。

事前学習転移 (init_from_dense):
  密カーネル W0(C,1,K0,K0) を K×K に中央埋め込みし、基底へ最小二乗射影して係数 A を初期化:
      A = pinv(H) · W0_emb · pinv(H)ᵀ     (H A Hᵀ が W0_emb の最良近似)
  分離(SVD)と同様、制約で表現力が落ちる分は init で「最良のガウス微分近似」から始めて
  fine-tune で回復する(gamma で恒等一致させる後付け転移はしない = 制約を最初から効かせる)。

パラメータ形状:
  log_sigma(C,)   … per-channel σ(log 空間)。名前を log_sigma にすると train.py の
                    σ 専用 param group(高lr/WD0)へ自動で入る。σ-warmup の sigma_cap も互換。
  coeff(C, M, M)  … per-channel の係数行列 A。この 3D キー coeff で機構を自動判別できる。
  bias(C,)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gauss_deriv_kernel_size(max_sigma, base_k=7, radius=3.0, cap=41):
    """max_sigma(px) を ±radius·σ 張れる奇数カーネルサイズへ。base_k 以上, cap 以下。"""
    k = 2 * int(math.ceil(radius * float(max_sigma))) + 1
    k = max(int(base_k), min(k, int(cap)))
    if k % 2 == 0:
        k += 1
    return k


class GaussianDerivativeDW(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        order=2,
        init_sigma=1.0,
        max_sigma=8.0,
        min_sigma=0.35,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size は奇数のみ対応"
        assert order >= 0, "order(次数 N)は 0 以上"
        self.channels = channels
        self.kernel_size = int(kernel_size)
        self.order = int(order)
        self.num_basis = self.order + 1  # M = N+1
        self.max_sigma = float(max_sigma)
        self.min_sigma = float(min_sigma)
        self.target_sigma = float(init_sigma)  # σ-warmup の到達目標(= 初期 σ)

        # per-channel σ(log 空間で常に正)。名前 log_sigma → σ 専用 param group 対象。
        self.log_sigma = nn.Parameter(
            torch.full((channels,), float(math.log(max(init_sigma, min_sigma))))
        )
        # 係数行列 A(C,M,M)。init_from_dense で密カーネルの最良近似に上書きされる。
        # 既定(事前学習なし)は order-0(平滑)恒等寄り: A[:,0,0]=1, 他=0。
        coeff0 = torch.zeros(channels, self.num_basis, self.num_basis)
        coeff0[:, 0, 0] = 1.0
        self.coeff = nn.Parameter(coeff0)
        self.bias = nn.Parameter(torch.zeros(channels))

        # σ-warmup 用の実効 σ 上限キャップ(Spectral 系と互換)。
        # persistent=False: state_dict に含めない → 既存 ckpt の load を壊さない。
        self.register_buffer(
            "sigma_cap", torch.tensor(float(max_sigma)), persistent=False
        )

    # ── 現在の σ (clamp 済み + warmup cap, px) ──
    def current_sigma(self):
        sigma = torch.clamp(torch.exp(self.log_sigma), self.min_sigma, self.max_sigma)
        return torch.minimum(sigma, self.sigma_cap)  # (C,)

    @torch.no_grad()
    def set_sigma_cap(self, value):
        """実効 σ の上限を設定(max_sigma を超えない)。σ-warmup スケジューラから呼ぶ。"""
        self.sigma_cap.fill_(min(float(value), self.max_sigma))

    # ── 1次元ガウス微分基底 H(σ): (C, K, M)。列 m = h_m = He_m(x/σ)·exp(-x²/2σ²) ──
    def _hermite_basis(self, sigma):
        C, K, M = self.channels, self.kernel_size, self.num_basis
        x = torch.arange(K, device=sigma.device, dtype=sigma.dtype) - (K - 1) / 2.0
        t = x.view(1, K) / sigma.view(C, 1).clamp_min(1e-3)  # (C,K) = x/σ
        g = torch.exp(-0.5 * t * t)  # (C,K) ガウス包絡
        # 確率論的エルミート漸化式: He_0=1, He_1=t, He_{k+1}=t·He_k − k·He_{k-1}
        He = [torch.ones_like(t)]
        if M >= 2:
            He.append(t)
        for k in range(1, M - 1):
            He.append(t * He[k] - k * He[k - 1])
        cols = []
        for m in range(M):
            h = He[m] * g  # (C,K)
            # per-channel L2 正規化(条件数改善 = 係数 A のスケールを揃える)
            h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            cols.append(h)
        return torch.stack(cols, dim=2)  # (C,K,M)

    def forward(self, x):
        C, K, M = self.channels, self.kernel_size, self.num_basis
        sigma = self.current_sigma()  # (C,)
        H = self._hermite_basis(sigma)  # (C,K,M)
        # 水平フィルタ B = A Hᵀ : (C,M,K)。B[c,m,j] = Σ_n A[c,m,n] H[c,j,n]
        B = torch.einsum("cmn,ckn->cmk", self.coeff, H)  # (C,M,K)
        out = None
        for m in range(M):
            wh = B[:, m, :].reshape(C, 1, 1, K)  # 水平 1×K
            wv = H[:, :, m].reshape(C, 1, K, 1)  # 垂直 K×1
            t = F.conv2d(x, wh, None, padding=(0, K // 2), groups=C)
            ym = F.conv2d(t, wv, None, padding=(K // 2, 0), groups=C)
            out = ym if out is None else out + ym
        return out + self.bias.view(1, -1, 1, 1)

    @torch.no_grad()
    def init_from_dense(self, dense_weight, dense_bias=None):
        """密カーネル W0(C,1,K0,K0) を基底へ最小二乗射影して coeff を初期化。

        A = pinv(H) · W0_emb · pinv(H)ᵀ  (H A Hᵀ が W0_emb の最良近似)。
        K0<K なら中央ゼロ埋め、K0>K なら中央クロップ。
        """
        C, K = self.channels, self.kernel_size
        K0 = dense_weight.shape[-1]
        W0 = dense_weight.reshape(C, dense_weight.shape[-2], K0).to(torch.float32)
        Wemb = W0.new_zeros(C, K, K)
        if K >= K0:
            off = (K - K0) // 2
            Wemb[:, off : off + K0, off : off + K0] = W0
        else:  # 密の方が大きい → 中央クロップ
            o = (K0 - K) // 2
            Wemb = W0[:, o : o + K, o : o + K].contiguous()
        H = self._hermite_basis(self.current_sigma().to(torch.float32))  # (C,K,M)
        Hp = torch.linalg.pinv(H)  # (C,M,K)
        A = torch.einsum("cmk,ckl,cnl->cmn", Hp, Wemb, Hp)  # (C,M,M)
        self.coeff.copy_(A.to(self.coeff.dtype))
        if dense_bias is not None:
            self.bias.copy_(dense_bias.to(self.bias.dtype))

    # ── 実効カーネル(C,1,K,K) を再構成(診断/可視化用) ──
    def effective_kernel(self):
        H = self._hermite_basis(self.current_sigma())  # (C,K,M)
        W = torch.einsum("ckm,cmn,cln->ckl", H, self.coeff, H)  # (C,K,K)
        return W.unsqueeze(1)

    # ── RF の広がり(= σ そのもの, px)。ERF 正則化ターゲットにも使える ──
    def rf_spread(self):
        return self.current_sigma().mean()

    def mean_sigma(self):
        return float(self.current_sigma().mean().detach())


def load_dense_into_gaussian_derivative(container, dense_state_dict, verbose=True):
    """container 内の全 GaussianDerivativeDW を密チェックポイントの weight/bias から
    ガウス微分基底への最小二乗射影で初期化する。

    Args:
        container: named_modules() を持つ nn.Module(backbone など)。
        dense_state_dict: 密モデルの state_dict(キー <module_name>.weight/bias)。
                          prefix(module./backbone.) は事前に剥がしておくこと。
    Returns:
        変換したモジュール数。
    """
    n = 0
    for name, m in container.named_modules():
        if not isinstance(m, GaussianDerivativeDW):
            continue
        w = dense_state_dict.get(name + ".weight")
        b = dense_state_dict.get(name + ".bias")
        if w is None:
            continue
        if w.dim() != 4 or w.shape[0] != m.channels or w.shape[1] != 1:
            if verbose:
                print(f"    ⚠️ {name}: 密 weight shape {tuple(w.shape)} 不一致、スキップ")
            continue
        m.init_from_dense(w, b)
        n += 1
    if verbose and n:
        print(
            f"    🔗 {n} 個の GaussianDerivativeDW を密カーネルの"
            f" ガウス微分基底(order={container_first_order(container)}) 射影で初期化"
        )
    return n


def container_first_order(container):
    """診断表示用: 最初に見つかった GaussianDerivativeDW の order。"""
    for _, m in container.named_modules():
        if isinstance(m, GaussianDerivativeDW):
            return m.order
    return "?"
