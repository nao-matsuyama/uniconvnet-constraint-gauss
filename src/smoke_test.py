# coding:utf-8
"""
ローカル CPU スモークテスト (GPU 不要)。

学習可能 dilation RFA + ERF 正則化 + FLOPs カウントが壊れていないかを、
小さなランダム入力で素早く検証する。torch さえ入っていれば動く。

  python3 src/smoke_test.py

確認項目:
  1. モデル構築 & forward (出力 shape = 入力 shape)
  2. DilatedDWConv が存在し log_dilation が学習可能
  3. init dilation=1 (事前学習 dense conv と一致する初期状態)
  4. ERF 正則化損失が有限で、log_dilation に勾配が流れる
  5. stage 別 target が効く (per_stage_spread が出る)
  6. model_stats の FLOPs / パラメータ数が計算できる
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(__file__))
from model_stats import count_flops, count_parameters, human_readable
from model_uniconvnet_unet import UniConvNet_UNet_13CH
from models.dilated_dw import DilatedDWConv
from models.erf_regularization import (
    collect_dilated_by_stage,
    collect_dilated_modules,
    erf_reg_loss,
)
from models.spectral_dw import SpectralDW
from models.spectral_mixture_dw import SpectralMixtureDW

try:
    from models.separable_dw import SeparableDWConv
except ImportError:
    SeparableDWConv = None

try:
    from models.gaussian_derivative_dw import GaussianDerivativeDW
except ImportError:
    GaussianDerivativeDW = None


def check(cond, msg):
    mark = "✅" if cond else "❌"
    print(f"  {mark} {msg}")
    if not cond:
        raise AssertionError(msg)


def _enlarge_rf(module):
    """RF を大きくする (機構ごとに操作対象が違う): dilation / σ を増やす。"""
    with torch.no_grad():
        if hasattr(module, "log_dilation"):
            module.log_dilation.fill_(float(torch.log(torch.tensor(3.0))))
        elif hasattr(module, "log_sigma"):
            module.log_sigma.fill_(float(torch.log(torch.tensor(8.0))))


def smoke_mode(mode, build_kwargs=None):
    """dense/separable/spectral 各モードの共通スモーク。

    確認: (1) forward 出力 shape 一致, (2) RF パラメータ(σ/重み)へ勾配が流れる,
          (3) RF を大きくしても出力 shape 不変。
    """
    print(f"\n===== dw_mode = {mode} =====")
    device = "cpu"
    torch.manual_seed(0)
    model = UniConvNet_UNet_13CH(
        num_classes=13, dw_mode=mode, **(build_kwargs or {})
    ).to(device)
    x = torch.randn(1, 3, 64, 64, device=device)

    y = model(x)
    check(y.shape == (1, 13, 64, 64), f"[{mode}] forward 出力 shape {tuple(y.shape)}")

    # RF 機構モジュールを収集 (dense=DilatedDWConv / spectral=SpectralDW /
    # spectral_mix=SpectralMixtureDW / separable=SeparableDWConv)
    rf_types = [DilatedDWConv, SpectralDW, SpectralMixtureDW]
    if SeparableDWConv is not None:
        rf_types.append(SeparableDWConv)
    if GaussianDerivativeDW is not None:
        rf_types.append(GaussianDerivativeDW)
    rf_types = tuple(rf_types)
    mods = [m for m in model.modules() if isinstance(m, rf_types)]
    # depths=[3,3,15,3]=24 block × (a1,a2,a3) = 72 個の RFA depthwise
    check(len(mods) == 72, f"[{mode}] RFA depthwise が 72 個 実際 {len(mods)}")

    model.zero_grad()
    y.pow(2).mean().backward()
    # RF/重みパラメータへ勾配が流れたか
    grad_params = []
    for m in mods:
        for pname in (
            "log_dilation",
            "log_sigma",
            "coeff",
            "weight_h",
            "weight_v",
            "weight",
        ):
            p = getattr(m, pname, None)
            if p is not None and p.grad is not None and float(p.grad.abs().sum()) > 0:
                grad_params.append(f"{type(m).__name__}.{pname}")
                break
    check(
        len(grad_params) > 0,
        f"[{mode}] RF/重みパラメータへ勾配が流れた: {len(grad_params)}/{len(mods)}",
    )

    # RF を大きくしても出力 shape 不変
    for m in mods:
        _enlarge_rf(m)
    y2 = model(x)
    check(y2.shape == y.shape, f"[{mode}] RF 拡大後も出力 shape {tuple(y2.shape)} 不変")
    return model


def main():
    device = "cpu"
    torch.manual_seed(0)
    print("=" * 56)
    print("  スモークテスト: 学習可能 dilation RFA + ERF 正則化")
    print("=" * 56)

    # ── 1. モデル構築 & forward ──
    print("\n[1] モデル構築 & forward")
    model = UniConvNet_UNet_13CH(num_classes=13).to(device)
    x = torch.randn(1, 3, 64, 64, device=device)
    y = model(x)
    check(y.shape == (1, 13, 64, 64), f"出力 shape {tuple(y.shape)} == (1,13,64,64)")

    # ── 2. DilatedDWConv の存在と学習可能性 ──
    print("\n[2] DilatedDWConv")
    mods = collect_dilated_modules(model)
    check(len(mods) > 0, f"DilatedDWConv が {len(mods)} 個ある")
    check(all(m.log_dilation.requires_grad for m in mods), "全 log_dilation が学習可能")

    # ── 3. init dilation == 1 ──
    print("\n[3] 初期 dilation")
    dils = [float(m.current_dilation().detach()) for m in mods]
    check(
        all(abs(d - 1.0) < 1e-6 for d in dils),
        f"初期 dilation が全て 1.0 (dense conv と一致): 例 {dils[:3]}",
    )

    # ── 4. ERF 正則化損失と勾配 ──
    print("\n[4] ERF 正則化損失 (stage 別 target = [2,3,4,6])")
    loss, diag = erf_reg_loss(model, [2.0, 3.0, 4.0, 6.0])
    check(torch.isfinite(loss), f"損失が有限: {float(loss):.4f}")
    model.zero_grad()
    loss.backward()
    grads = [m.log_dilation.grad for m in mods]
    n_grad = sum(1 for g in grads if g is not None and float(g.abs()) > 0)
    check(n_grad > 0, f"log_dilation に勾配が流れた: {n_grad}/{len(mods)} 個が非ゼロ")

    # ── 5. stage 別診断 ──
    print("\n[5] stage 別 RF 広がり")
    by_stage = collect_dilated_by_stage(model)
    check(len(by_stage) == 4, f"stage 数 = {len(by_stage)}")
    print(f"     per_stage_spread = {diag['per_stage_spread']}")
    print(
        f"     mean_dilation = {diag['mean_dilation']:.3f}, "
        f"mean_spread = {diag['mean_spread']:.3f}"
    )

    # ── 6. FLOPs / パラメータ数 ──
    print("\n[6] FLOPs / パラメータ数")
    total, trainable = count_parameters(model)
    flops = count_flops(model, torch.zeros(1, 3, 64, 64, device=device))
    check(flops > 0, f"FLOPs 計算 OK: {human_readable(flops)}")
    print(
        f"     総パラメータ: {human_readable(total)}  "
        f"学習可能: {human_readable(trainable)}"
    )

    # ── 7. dilation を大きくしても出力 shape 不変 (スパース性の確認) ──
    print("\n[7] dilation 拡大後も出力 shape 不変 (タップ数 k^2 固定)")
    with torch.no_grad():
        for m in mods:
            m.log_dilation.fill_(float(torch.log(torch.tensor(3.0))))  # dilation≈3
    y2 = model(x)
    check(y2.shape == y.shape, f"dilation≈3 でも出力 shape {tuple(y2.shape)} 不変")

    # ── 8. 新機構 (機構A separable / 機構B spectral / 機構C spectral_mix) の共通スモーク ──
    print("\n[8] dw_mode 別スモーク (dense / spectral / spectral_mix / separable)")
    smoke_mode("dense")
    smoke_mode(
        "spectral", {"spectral_init_sigma": [8, 6, 4, 2], "spectral_init_gamma": 0.5}
    )
    smoke_mode(
        "spectral_mix",
        {
            "spectral_init_sigma": [8, 6, 4, 2],
            "spectral_init_gamma": 0.5,
            "spectral_num_gaussians": 3,
        },
    )
    if SeparableDWConv is not None:
        smoke_mode("separable")
        smoke_mode("separable", {"separable_rank": 2})  # 機構A rank-2 (境界回収狙い)
    else:
        print("  (separable_dw 未導入のためスキップ)")
    if GaussianDerivativeDW is not None:
        gd_kw = {
            "spectral_init_sigma": [4, 3, 2, 1.5],
            "spectral_max_sigma": [5, 4, 3, 2],
        }
        smoke_mode("gauss_deriv", {**gd_kw, "gauss_deriv_order": 1})
        smoke_mode("gauss_deriv", {**gd_kw, "gauss_deriv_order": 2})  # エッジ/リッジ
    else:
        print("  (gaussian_derivative_dw 未導入のためスキップ)")
    # gauss_pyramid (多スケール純ガウス, SpectralDW 派生)。学習可能σ版は smoke_mode で
    # log_sigma に勾配が流れることまで確認 (純ガウスは gate 無しなので σ に勾配が届く)。
    gp_kw = {
        "spectral_init_sigma": [2, 2, 2, 2],
        "spectral_max_sigma": [8, 6, 4, 3],
        "gauss_pyramid_growth": 1.6,
    }
    smoke_mode("gauss_pyramid", gp_kw)
    # freeze_scale=True は depthwise に勾配パラメータが無くなる(=固定スケール空間)ので、
    # smoke_mode の勾配アサートは適用外。build+forward と log_sigma 凍結だけ確認する。
    print("\n===== dw_mode = gauss_pyramid (freeze_scale=True) =====")
    from models.spectral_dw import SpectralDW as _SpecDW

    torch.manual_seed(0)
    mdl_fz = UniConvNet_UNet_13CH(
        num_classes=13, dw_mode="gauss_pyramid", gauss_freeze_scale=True, **gp_kw
    )
    xf = torch.randn(1, 3, 64, 64)
    yf = mdl_fz(xf)
    check(yf.shape == (1, 13, 64, 64), f"[gp-freeze] forward 出力 shape {tuple(yf.shape)}")
    spec_mods = [m for m in mdl_fz.modules() if isinstance(m, _SpecDW)]
    check(len(spec_mods) == 72, f"[gp-freeze] 純ガウス枝 72 個 実際 {len(spec_mods)}")
    check(
        all(not m.log_sigma.requires_grad for m in spec_mods),
        "[gp-freeze] 全 log_sigma が凍結 (requires_grad=False)",
    )

    # ── 9. FLOPs 比較 (dense vs spectral vs spectral_mix vs separable / gauss_deriv) ──
    print("\n[9] FLOPs 比較 (機構ごとの省コスト)")
    for label, mode, kw in [
        ("dense", "dense", {}),
        ("spectral", "spectral", {"spectral_init_sigma": [8, 6, 4, 2]}),
        ("spectral_mix", "spectral_mix", {"spectral_init_sigma": [8, 6, 4, 2]}),
        ("separable_r1", "separable", {"separable_rank": 1}),
        ("separable_r2", "separable", {"separable_rank": 2}),
        (
            "gauss_deriv_n2",
            "gauss_deriv",
            {"spectral_max_sigma": [8, 6, 4, 2], "gauss_deriv_order": 2},
        ),
        (
            "gauss_pyramid",
            "gauss_pyramid",
            {"spectral_init_sigma": [2, 2, 2, 2], "spectral_max_sigma": [8, 6, 4, 3]},
        ),
    ]:
        if mode == "gauss_deriv" and GaussianDerivativeDW is None:
            continue
        if mode == "separable" and SeparableDWConv is None:
            continue
        torch.manual_seed(0)
        mdl = UniConvNet_UNet_13CH(num_classes=13, dw_mode=mode, **kw)
        fl = count_flops(mdl, torch.zeros(1, 3, 256, 256))
        tot, _ = count_parameters(mdl)
        print(
            f"     {label:12s} params={human_readable(tot):>9}  FLOPs@256={human_readable(fl):>9}"
        )

    print("\n" + "=" * 56)
    print("  ✅ 全項目 PASS")
    print("=" * 56)


if __name__ == "__main__":
    main()
