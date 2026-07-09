# coding:utf-8
"""
ERF 正則化 — 学習可能 dilation を「適切な ERF」へ寄せる損失。

各 DilatedDWConv の有効 RF 広がり rf_spread() を、目標の広がり target_spread に
近づける二乗誤差を平均する。target_spread は ERF 可視化 (visualize_erf.py) で
得たガウスフィット σ を基準に決める想定。

  L_erf = mean_i ( rf_spread_i - target_i )^2

stage 別ターゲット:
  U-Net の低層 (stage0) はエッジなど局所情報、高層 (stage3) は広域/帯域情報を
  見るはずなので、target を stage ごとに変えられる。
  target_spread に 4 要素のリストを渡すと stage0..3 に割り当てる。
  スカラーを渡すと全 stage 共通。

学習損失に L_total = L_task + lambda * L_erf として加える。lambda=0 で無効。
"""

import torch

try:
    from .dilated_dw import DilatedDWConv
except ImportError:
    from dilated_dw import DilatedDWConv

try:
    from .spectral_gaussian_dw import SpectralGaussianDW
except ImportError:
    from spectral_gaussian_dw import SpectralGaussianDW

try:
    from .spectral_dw import SpectralDW
except ImportError:
    from spectral_dw import SpectralDW

try:
    from .spectral_mixture_dw import SpectralMixtureDW
except ImportError:
    from spectral_mixture_dw import SpectralMixtureDW

try:
    from .gaussian_derivative_dw import GaussianDerivativeDW
except ImportError:
    from gaussian_derivative_dw import GaussianDerivativeDW


# rf_spread() を持つ RF 機構 (dilation 版 / 周波数ガウス版 / 動的切り出し版 / 多ガウス混合版 /
# ガウス微分基底版) を正則化・診断対象に。
_RF_TYPES = (
    DilatedDWConv,
    SpectralGaussianDW,
    SpectralDW,
    SpectralMixtureDW,
    GaussianDerivativeDW,
)


def _rf_scalar(m):
    """診断用の代表 RF スカラー: DilatedDWConv は dilation、Spectral は σ(px) 平均。"""
    if hasattr(m, "current_dilation"):
        return m.current_dilation().detach()
    if hasattr(m, "current_sigma"):
        return m.current_sigma().mean().detach()
    return torch.zeros(())


def _get_backbone(model):
    m = model.module if hasattr(model, "module") else model
    return m.backbone


def collect_dilated_modules(model):
    """model 内の RF 機構 (DilatedDWConv / SpectralGaussianDW) をすべて集める (flat)。"""
    return [m for m in model.modules() if isinstance(m, _RF_TYPES)]


def collect_dilated_by_stage(model):
    """
    backbone.stages[i] ごとに RF 機構をまとめて返す。
    Returns: dict {stage_idx: [module, ...]}
    """
    backbone = _get_backbone(model)
    by_stage = {}
    for i, stage in enumerate(backbone.stages):
        by_stage[i] = [m for m in stage.modules() if isinstance(m, _RF_TYPES)]
    return by_stage


def _normalize_targets(target_spread, n_stages):
    """スカラー or リストを n_stages 個の per-stage 目標に正規化。"""
    if isinstance(target_spread, (list, tuple)):
        ts = [float(t) for t in target_spread]
        if len(ts) < n_stages:  # 足りなければ最後の値で埋める
            ts = ts + [ts[-1]] * (n_stages - len(ts))
        return ts[:n_stages]
    return [float(target_spread)] * n_stages


def erf_reg_loss(model, target_spread):
    """
    ERF 正則化損失と診断量を返す。

    Args:
        target_spread: float (全 stage 共通) または list[float] (stage 別)

    Returns:
        loss        : スカラー tensor (微分可能)
        diagnostics : dict(mean_dilation, mean_spread, per_stage_spread, n_modules)
    """
    by_stage = collect_dilated_by_stage(model)
    n_stages = len(by_stage)
    targets = _normalize_targets(target_spread, n_stages)

    losses, spreads_all, dil_all, per_stage_spread = [], [], [], []
    for i in range(n_stages):
        mods = by_stage[i]
        t = targets[i]
        stage_spreads = []
        for m in mods:
            s = m.rf_spread()
            losses.append((s - t) ** 2)
            stage_spreads.append(s)
            spreads_all.append(s.detach())
            dil_all.append(_rf_scalar(m))
        if stage_spreads:
            per_stage_spread.append(float(torch.stack(stage_spreads).mean().detach()))
        else:
            per_stage_spread.append(0.0)

    if not losses:
        zero = torch.zeros((), requires_grad=True)
        return zero, {
            "mean_dilation": 0.0,
            "mean_spread": 0.0,
            "per_stage_spread": [],
            "n_modules": 0,
        }

    loss = torch.stack(losses).mean()
    diag = {
        "mean_dilation": float(torch.stack(dil_all).mean()),
        "mean_spread": float(torch.stack(spreads_all).mean()),
        "per_stage_spread": [round(s, 3) for s in per_stage_spread],
        "n_modules": len(spreads_all),
    }
    return loss, diag


def dilation_parameters(model):
    """学習可能 RF パラメータ (DilatedDWConv.log_dilation / Spectral.log_sigma) を返す。"""
    params = []
    for m in collect_dilated_modules(model):
        if hasattr(m, "log_dilation"):
            params.append(m.log_dilation)
        elif hasattr(m, "log_sigma"):
            params.append(m.log_sigma)
    return params
