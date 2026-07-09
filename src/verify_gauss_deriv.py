# coding:utf-8
"""GaussianDerivativeDW の (1) 事前学習転移 (基底射影), (2) eval 往復 (ckpt_utils 判別),
(3) 基底が事前学習カーネルをどれだけ捉えるか (制約の"損失") を一括検証する使い捨てスクリプト。

  docker ... python3 src/verify_gauss_deriv.py
"""
import os
import sys
import tempfile

import torch

sys.path.append(os.path.dirname(__file__))
from ckpt_utils import build_model_from_checkpoint
from model_uniconvnet_unet import UniConvNet_UNet_13CH
from models.gaussian_derivative_dw import (
    GaussianDerivativeDW,
    load_dense_into_gaussian_derivative,
)

PRE = "uniconvnet_t_1k_224_ema.pth"


def strip(sd, pfx):
    if sd and all(k.startswith(pfx) for k in sd):
        return {k[len(pfx):]: v for k, v in sd.items()}
    return sd


def main():
    torch.manual_seed(0)
    order = 2
    init_sigma = [3.0, 3.0, 2.0, 1.5]
    max_sigma = [6.0, 5.0, 4.0, 3.0]
    model = UniConvNet_UNet_13CH(
        num_classes=13,
        dw_mode="gauss_deriv",
        gauss_deriv_order=order,
        spectral_init_sigma=init_sigma,
        spectral_max_sigma=max_sigma,
    )

    # ── 1. 事前学習転移 (密カーネル → ガウス微分基底へ最小二乗射影) ──
    ck = torch.load(PRE, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = strip(strip(sd, "module."), "backbone.")
    n = load_dense_into_gaussian_derivative(model.backbone, sd, verbose=True)
    print(f"[1] 転移した GaussianDerivativeDW: {n} 個")

    # ── 基底が事前学習カーネルをどれだけ捉えるか (captured energy) ──
    #   各モジュールで、密カーネルを K×K に埋め込んだ Wemb に対し、再構成 H A Hᵀ の
    #   相対誤差と captured = 1 - ||Wemb - recon||² / ||Wemb||² を測る。
    caps = {0: [], 1: [], 2: [], 3: []}
    name2stage = {}
    for si, stage in enumerate(model.backbone.stages):
        for nm, m in stage.named_modules():
            if isinstance(m, GaussianDerivativeDW):
                name2stage[id(m)] = si
    for name, m in model.backbone.named_modules():
        if not isinstance(m, GaussianDerivativeDW):
            continue
        w = sd.get(name + ".weight")
        if w is None:
            continue
        C, _, K0, _ = w.shape
        K = m.kernel_size
        W0 = w.reshape(C, K0, K0).float()
        Wemb = W0.new_zeros(C, K, K)
        off = (K - K0) // 2
        Wemb[:, off:off + K0, off:off + K0] = W0
        recon = m.effective_kernel().reshape(C, K, K).float()
        num = (Wemb - recon).pow(2).flatten(1).sum(1)
        den = Wemb.pow(2).flatten(1).sum(1) + 1e-12
        cap = (1.0 - num / den).mean().item()
        st = name2stage.get(id(m), 0)
        caps[st].append(cap)
    print("[1b] 基底 captured energy (1=完全再現) — stage 別平均:")
    for st in range(4):
        if caps[st]:
            vals = torch.tensor(caps[st])
            print(f"      stage{st}: mean={vals.mean():.3f}  min={vals.min():.3f}  "
                  f"(n={len(caps[st])} branch, K={ [m.kernel_size for _,m in model.backbone.named_modules() if isinstance(m,GaussianDerivativeDW) and name2stage.get(id(m))==st][:1] })")

    # ── 2. forward / backward が通る ──
    x = torch.randn(1, 3, 96, 96)
    y = model(x)
    y.pow(2).mean().backward()
    gcoeff = sum(
        1 for m in model.modules()
        if isinstance(m, GaussianDerivativeDW) and m.coeff.grad is not None
        and float(m.coeff.grad.abs().sum()) > 0
    )
    gsig = sum(
        1 for m in model.modules()
        if isinstance(m, GaussianDerivativeDW) and m.log_sigma.grad is not None
        and float(m.log_sigma.grad.abs().sum()) > 0
    )
    print(f"[2] forward OK shape={tuple(y.shape)} | coeff grad {gcoeff}/72, log_sigma grad {gsig}/72")

    # ── 3. eval 往復: save → build_model_from_checkpoint で機構自動判別 → 出力一致 ──
    model.eval()
    with torch.no_grad():
        y1 = model(x)
    with tempfile.TemporaryDirectory() as d:
        wpath = os.path.join(d, "ck.pth")
        torch.save(model.state_dict(), wpath)
        import json
        with open(os.path.join(d, "run_config.json"), "w") as f:
            json.dump({"args": {
                "dw_mode": "gauss_deriv", "gauss_deriv_order": order,
                "spectral_init_sigma": init_sigma, "spectral_max_sigma": max_sigma,
            }}, f)
        m2 = build_model_from_checkpoint(wpath, num_classes=13, device="cpu", verbose=True)
        with torch.no_grad():
            y2 = m2(x)
    diff = (y1 - y2).abs().max().item()
    print(f"[3] eval 往復 max|Δ| = {diff:.2e}  {'OK' if diff < 1e-4 else 'MISMATCH!'}")

    print("\n✅ verify_gauss_deriv 完了" if diff < 1e-4 else "\n❌ 不一致あり")


if __name__ == "__main__":
    main()
