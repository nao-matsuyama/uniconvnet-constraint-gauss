# coding:utf-8
"""
SeparableDWConv — 非対称カーネル分解 (機構A, rank-R 一般化)。

狙い:
  K×K の depthwise 空間フィルタを、R 本の **水平 1×K** と **垂直 K×1** の和
  (rank-R 分離) で近似する。
    y = Σ_{r=1}^R DWConv_{K×1}^{(r)}( DWConv_{1×K}^{(r)}(x) )   (各 depthwise, groups=C)
  * パラメータ/FLOPs: K² → 2RK に減 (K=11, R=1 で 121→22 タップ=5.5x, R=2 で 44=2.75x)。
  * AGD 保存: 2次元ガウス G(x,y)=g(x)·g(y) は 1次元ガウスの積。ガウス的カーネルなら
    分離は厳密に成り立つので、この機構は「滑らかな (低周波集中=ガウス的) 大カーネル」
    という設計思想を壊さず、単一 dilated のような疎サンプリング (AGD 崩壊) を避ける。
  * rank を上げる意味: 事前学習カーネルの rank-1 energy は ~0.80 (=20% の非分離構造を
    喪失)。rank-1 は滑らかで AGD/kurtosis を保つ一方、鋭い非分離構造 (境界) を表現できず
    境界が甘くなる。R を上げると SVD 上位 R 成分で ~0.90+ を回収でき、境界を戻せる可能性。
    R=K で密カーネルに厳密一致 (分離近似ではなくなる)。→ rank が精度-効率-AGD を連続制御。

畳み込みの向き (PyTorch conv2d = cross-correlation) と分解の対応 (1 本の rank について):
    t[y,x]   = Σ_j weight_h[j] · x[y,    x+j-p]      (水平 1×K)
    out[y,x] = Σ_i weight_v[i] · t[y+i-p, x]         (垂直 K×1)
             = Σ_i Σ_j weight_v[i] weight_h[j] · x[y+i-p, x+j-p]
  → 1 本の合成カーネル K2[i,j] = weight_v[i] · weight_h[j] (rank-1)。R 本の和が rank-R。

事前学習重みからの初期化 (init_from_dense):
  元の密カーネル W(C,1,K,K) を **チャネルごとに rank-R SVD 近似** する:
      W_c ≈ Σ_{r=1}^R s_r · u_r v_rᵀ         (s_r: 特異値, u_r: 左=行/H, v_r: 右=列/W)
      weight_v[c,r] = √s_r · u_r     (C,R,K,1)
      weight_h[c,r] = √s_r · v_r     (C,R,1,K)
  合成 Σ_r weight_v[c,r,i]·weight_h[c,r,j] = Σ_r s_r u_r[i] v_r[j] = W_c の rank-R 近似
  (= Eckart–Young の意味で最良の rank-R 近似)。bias はそのまま転移。

パラメータ形状は weight_h(C,R,1,K)/weight_v(C,R,K,1)/bias(C)。R は shape[1] から復元できる
(評価時に ckpt_utils が weight_h の rank を読む)。事前学習チェックポイントは
load_dense_into_separable() で <name>.weight → weight_h/weight_v へ変換して転移する。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableDWConv(nn.Module):
    def __init__(self, channels, kernel_size, rank=1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size は奇数のみ対応"
        assert rank >= 1, "rank は 1 以上"
        assert rank <= kernel_size, "rank は kernel_size 以下 (R=K で密に厳密一致)"
        self.channels = channels
        self.kernel_size = kernel_size
        self.rank = rank

        # R 本の 水平 1×K と 垂直 K×1 の depthwise 重み。
        # 形状 (C,R,1,K)/(C,R,K,1): 各 rank r の slice [:, r:r+1] が conv2d の
        # depthwise 重み (C,1,1,K)/(C,1,K,1) にそのまま使える。
        self.weight_h = nn.Parameter(torch.empty(channels, rank, 1, kernel_size))
        self.weight_v = nn.Parameter(torch.empty(channels, rank, kernel_size, 1))
        self.bias = nn.Parameter(torch.zeros(channels))
        self._reset_parameters()

    def _reset_parameters(self):
        # 事前学習が無い新規構築でも安定に動くよう、rank0 の合成が中心デルタ (≈恒等
        # depthwise) + 小ノイズ、他 rank は小ノイズのみになる初期化。実運用では
        # init_from_dense で密カーネルから上書きされる。
        k = self.kernel_size
        with torch.no_grad():
            self.weight_h.zero_()
            self.weight_v.zero_()
            self.weight_h[:, 0, 0, k // 2] = 1.0
            self.weight_v[:, 0, k // 2, 0] = 1.0
            self.weight_h.add_(torch.randn_like(self.weight_h) * 0.02)
            self.weight_v.add_(torch.randn_like(self.weight_v) * 0.02)
            self.bias.zero_()

    @torch.no_grad()
    def init_from_dense(self, dense_weight, dense_bias=None):
        """密カーネル W(C,1,K,K) を rank-R SVD 分解して weight_h/weight_v を設定する。"""
        C, _, K, K2 = dense_weight.shape
        assert (C, K, K2) == (self.channels, self.kernel_size, self.kernel_size), (
            f"shape 不一致: dense {tuple(dense_weight.shape)} vs "
            f"({self.channels},1,{self.kernel_size},{self.kernel_size})"
        )
        R = self.rank
        W = dense_weight.reshape(C, K, K).to(torch.float32)
        # チャネルごと (バッチ) SVD: U(C,K,K) S(C,K) Vh(C,K,K)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        n_sv = S.shape[1]  # = K (>= R なので通常全 rank を埋められる)
        self.weight_h.zero_()
        self.weight_v.zero_()
        for r in range(min(R, n_sv)):
            sr = S[:, r].clamp_min(0.0).sqrt()  # (C,)
            ur = U[:, :, r]  # (C,K) 左 = 行(H)方向
            vr = Vh[:, r, :]  # (C,K) 右 = 列(W)方向
            wv = (sr.unsqueeze(1) * ur).to(dense_weight.dtype)  # (C,K)
            wh = (sr.unsqueeze(1) * vr).to(dense_weight.dtype)  # (C,K)
            self.weight_v[:, r, :, 0].copy_(wv)
            self.weight_h[:, r, 0, :].copy_(wh)
        if dense_bias is not None:
            self.bias.copy_(dense_bias.to(self.bias.dtype))

    def forward(self, x):
        k = self.kernel_size
        out = None
        for r in range(self.rank):
            # rank r の depthwise 重み (C,1,1,K)/(C,1,K,1) を slice で取り出す。
            wh = self.weight_h[:, r : r + 1, :, :]
            wv = self.weight_v[:, r : r + 1, :, :]
            # 水平 1×K (padding は W 方向のみ) → 垂直 K×1 (padding は H 方向のみ)
            t = F.conv2d(x, wh, None, padding=(0, k // 2), groups=self.channels)
            yr = F.conv2d(t, wv, None, padding=(k // 2, 0), groups=self.channels)
            out = yr if out is None else out + yr
        return out + self.bias.view(1, -1, 1, 1)

    # ── 有効 RF の広がり (合成カーネルの weight エネルギー重み付き RMS 半径) ──
    # 診断用 (dilation=1 固定なのでカーネル内広がりのみ)。
    def rf_spread(self):
        k = self.kernel_size
        # 合成カーネル K2[i,j] = Σ_r weight_v[c,r,i]·weight_h[c,r,j] を再構成 (rank 和)
        wv = self.weight_v.reshape(self.channels, self.rank, k)  # (C,R,K) H方向
        wh = self.weight_h.reshape(self.channels, self.rank, k)  # (C,R,K) W方向
        k2 = torch.einsum("cri,crj->cij", wv, wh)  # (C,K,K)
        coords = torch.arange(k, device=k2.device, dtype=k2.dtype) - (k - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        r2 = xx**2 + yy**2  # (K,K)
        w2 = (k2.abs().mean(dim=0)) ** 2  # (K,K) チャネル平均エネルギー
        w2 = w2 / (w2.sum() + 1e-8)
        return torch.sqrt((w2 * r2).sum() + 1e-8)


def load_dense_into_separable(container, dense_state_dict, verbose=True):
    """container 内の全 SeparableDWConv を、密チェックポイントの対応 weight/bias から
    rank-R SVD で初期化する (R は各モジュールの rank に従う)。

    Args:
        container: named_modules() を持つ nn.Module (backbone など)。
        dense_state_dict: 密モデルの state_dict (キーは <module_name>.weight/bias)。
                          prefix (module./backbone.) は事前に剥がしておくこと。
    Returns:
        変換したモジュール数。
    """
    n = 0
    max_rank = 1
    for name, m in container.named_modules():
        if not isinstance(m, SeparableDWConv):
            continue
        w = dense_state_dict.get(name + ".weight")
        b = dense_state_dict.get(name + ".bias")
        if w is None:
            continue
        if tuple(w.shape) != (m.channels, 1, m.kernel_size, m.kernel_size):
            if verbose:
                print(
                    f"    ⚠️ {name}: 密 weight shape {tuple(w.shape)} 不一致、スキップ"
                )
            continue
        m.init_from_dense(w, b)
        max_rank = max(max_rank, m.rank)
        n += 1
    if verbose and n:
        print(
            f"    🔗 {n} 個の SeparableDWConv を密カーネルの rank-{max_rank} SVD で初期化"
        )
    return n
