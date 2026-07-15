# coding:utf-8
"""
GaussianPyramidDW — 多スケール純ガウス depthwise (Gaussian scale-space pyramid, dw_mode=gauss_pyramid)。

狙い (RFA の作り方の再設計):
  現状の RFA は大小の**学習カーネル**(k7/9/11)を組み合わせて漸近ガウス受容野(AGD)を「作る」。
  本機構は各枝を最初から**規定スケールの純ガウス**にし、AGD を後付けで作らず最初からガウス純度を
  保つ。ConvMod の a1/a2/a3 を σ1<σ2<σ3 の純ガウス(周波数領域, 動的スペクトル切り出し)にして
  スケール空間ピラミッドを張る。

なぜこれで「徐々に大きいガウス」かつ「境界も保てる」か:
  (1) カスケード = ガウス半群: a1→a2→a3 と直列なので G_σ1 * G_σ2 = G_√(σ1²+σ2²)。小ガウスを
      重ねるだけで実効σが自然に増大(= 熱拡散/スケール空間そのもの)。per-branch の growth で
      さらに σ1<σ2<σ3 を明示的に広げる。
  (2) エッジは depthwise でなく pointwise で作る: 純ガウスは低域通過でボケるが、ConvMod の
      pointwise(1×1) conv (v1/v11/v12…) と加算スキップが**異なるスケールのガウス応答を線形結合**
      できる → DoG(Difference of Gaussians) = バンドパス = エッジ検出器を学習して境界を復元。
      → depthwise は純ガウスのままでよく、境界表現を channel-mixing に追い出す(gauss_deriv が
      微分項を depthwise 内に入れるのと対照的な思想)。

実装 (SpectralDW の薄い派生):
  周波数純ガウス自体は SpectralDW(周波数ガウス + 動的切り出し)を use_local_branch=False にした
  ものと厳密に同じなので、それを継承して use_local_branch=False を強制するだけ。
  - 純ガウス(local枝/gamma なし): 演算子は純粋な周波数ガウス低域通過。
  - σ は per-channel 学習可能(既定)。freeze_scale=True で固定スケジュール(純スケール空間)。
  - σ大でも周波数計算はカーネルサイズ非依存 O(N logN)、切り出しで cost∝η²。大きな空間カーネル
    (gauss_deriv の K=41 のような)が要らないのが周波数版の利点。
  - state_dict は SpectralDW と同形(weight/bias/log_sigma/gamma)。機構の区別は run_config の
    dw_mode="gauss_pyramid" で行う(ckpt_utils)。SpectralDW の派生なので model_stats/benchmark/
    erf_regularization/visualize_sigma の isinstance(SpectralDW) 判定にそのまま乗る。
"""

try:
    from .spectral_dw import SpectralDW
except ImportError:
    from spectral_dw import SpectralDW


class GaussianPyramidDW(SpectralDW):
    def __init__(
        self,
        channels,
        kernel_size,
        init_sigma=1.0,
        max_sigma=32.0,
        alpha=2.0,
        pad_factor=0.0,
        crop_quantile=0.0,
        freeze_scale=False,
    ):
        super().__init__(
            channels,
            kernel_size=kernel_size,
            init_sigma=init_sigma,
            init_gamma=0.0,  # 純ガウス: local 枝/gamma は使わない
            max_sigma=max_sigma,
            alpha=alpha,
            use_local_branch=False,  # 純ガウス低域通過のみ(この機構の肝)
            pad_factor=pad_factor,
            crop_quantile=crop_quantile,
        )
        # freeze_scale=True で σ を固定(純スケール空間)。既定は学習可能(sigma-lr で駆動)。
        self.freeze_scale = bool(freeze_scale)
        if self.freeze_scale:
            self.log_sigma.requires_grad_(False)
