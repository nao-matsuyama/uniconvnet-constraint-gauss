# coding:utf-8
"""
RF プローブ実験 — 「広域文脈が必須なタスクで、受容野(RF)が効くか／学習可能
dilation がタスク駆動で RF を広げて解けるか」を最小コストで検証する。

課題: 序数バーのラベリング
  入力 (1,H,W): 黒背景に K 本の縦バー(位置ランダム, 見た目は全部同じ)
  正解 (H,W)  : 各バーを「左から何番目か」(1..K) でラベル, 背景=0
  → 見た目では序数が分からないので「左端から数える」しかない
  → バーが右にあるほど大きい RF が必要。RF 不足だと右のバーで失敗する。

3 モード比較:
  small     … 通常 depthwise 3x3 (dilation 1) → RF 小
  large     … 固定 dilation の depthwise → RF 大
  learnable … DilatedDWConv (dilation を学習) → タスク駆動で RF が伸びるか

出力: 序数ごとの精度表 + グラフ (acc vs 序数)。
  small が右の序数で落ち、learnable が小→大へ dilation を伸ばして解ければ、
  「RF が要るタスクでは学習可能 dilation が適応する」を示せる。

使い方:
  python3 probe/ordinal_probe.py --mode small     --out-dir /workspace/probe_results/small
  python3 probe/ordinal_probe.py --mode large     --out-dir /workspace/probe_results/large
  python3 probe/ordinal_probe.py --mode learnable  --out-dir /workspace/probe_results/learnable
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models.content_adaptive_dw import ContentAdaptiveDW
from models.dilated_dw import DilatedDWConv
from models.spectral_gaussian_dw import SpectralGaussianDW


# ──────────────────────────────────────────────
# 合成データ生成 (オンザフライ)
# ──────────────────────────────────────────────
def gen_batch(B, H, W, K, bar_w=5, device="cpu"):
    img = torch.zeros(B, 1, H, W, device=device)
    mask = torch.zeros(B, H, W, dtype=torch.long, device=device)
    for b in range(B):
        # K 本のバー中心 x を昇順ランダム配置 (重ならないよう間隔確保)
        margin = bar_w + 2
        xs = np.sort(
            np.random.choice(np.arange(margin, W - margin), size=K, replace=False)
        )
        # 近すぎる配置を弾く簡易リトライ
        tries = 0
        while np.min(np.diff(xs)) < bar_w * 2 and tries < 20:
            xs = np.sort(
                np.random.choice(np.arange(margin, W - margin), size=K, replace=False)
            )
            tries += 1
        for j, x in enumerate(xs):  # j=0..K-1 → ラベル j+1
            x0, x1 = x - bar_w // 2, x - bar_w // 2 + bar_w
            img[b, 0, :, x0:x1] = 1.0
            mask[b, :, x0:x1] = j + 1
    return img, mask


# ──────────────────────────────────────────────
# プローブモデル (downsample 無し → RF が透明)
#   RF ≈ 1 + n_blocks * (k-1) * dilation
# ──────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, ch, mode, k=3, dil=1, init_sigma=1.0):
        super().__init__()
        self.mode = mode
        if mode == "learnable":
            # DilatedDWConv は (B,C,H,W) channels_first を直接取る
            self.dw = DilatedDWConv(
                ch, kernel_size=k, init_dilation=1.0, max_dilation=64.0
            )
        elif mode == "adaptive":
            self.dw = ContentAdaptiveDW(ch, kernel_size=k, dilations=(1, 4, 16, 32))
        elif mode == "spectral":
            # 周波数ガウス。スクラッチ学習なので init_gamma=1 で spectral 枝を最初から有効化。
            # init_sigma を大きくすると鶏卵谷を回避できるか検証できる。
            self.dw = SpectralGaussianDW(
                ch, kernel_size=k, init_sigma=init_sigma, init_gamma=1.0, max_sigma=64.0
            )
        else:
            d = dil if mode == "large" else 1
            self.dw = nn.Conv2d(
                ch, ch, k, padding=(k - 1) // 2 * d, dilation=d, groups=ch
            )
        self.pw = nn.Conv2d(ch, ch, 1)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = self.act(self.pw(self.dw(x)))
        return self.norm(x + y)


class ProbeNet(nn.Module):
    def __init__(
        self, num_classes, mode, ch=32, n_blocks=6, k=3, dil=8, init_sigma=1.0
    ):
        super().__init__()
        self.stem = nn.Conv2d(1, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            [Block(ch, mode, k, dil, init_sigma) for _ in range(n_blocks)]
        )
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

    def mean_dilation(self):
        ds = [
            float(m.current_dilation().detach())
            for m in self.modules()
            if isinstance(m, DilatedDWConv)
        ]
        if ds:
            return float(np.mean(ds))
        # spectral: per-channel σ (px) の平均。dilation と同じ「RFの広さ」スロットで表示。
        ss = [
            m.mean_sigma() for m in self.modules() if isinstance(m, SpectralGaussianDW)
        ]
        if ss:
            return float(np.mean(ss))
        # adaptive: 直近 forward の期待 dilation 平均
        cs = [
            m.mean_expected_dilation()
            for m in self.modules()
            if isinstance(m, ContentAdaptiveDW)
        ]
        cs = [c for c in cs if c == c]  # NaN 除外
        return float(np.mean(cs)) if cs else 1.0

    @torch.no_grad()
    def dilation_map(self, x):
        """adaptive ブロックの期待 dilation マップを全ブロック平均で返す (H,W)。"""
        self.eval()
        h = self.stem(x)
        maps = []
        for blk in self.blocks:
            if isinstance(blk.dw, ContentAdaptiveDW):
                maps.append(blk.dw.expected_dilation(h)[0, 0])
            h = blk(h)
        if not maps:
            return None
        return torch.stack(maps).mean(0).cpu().numpy()

    def erf_reg(self, target):
        """RF拡大の駆動力。DilatedDWConv / SpectralGaussianDW の rf_spread を
        target へ寄せる正則化 (自己完結)。spectral では σ(px) を target へ押し上げる。"""
        mods = [
            m
            for m in self.modules()
            if isinstance(m, (DilatedDWConv, SpectralGaussianDW))
        ]
        if not mods:
            return torch.zeros((), device=self.head.weight.device)
        sp = torch.stack([m.rf_spread() for m in mods])
        return ((sp - target) ** 2).mean()


# ──────────────────────────────────────────────
# 評価: 序数ごとのバー画素精度
# ──────────────────────────────────────────────
@torch.no_grad()
def eval_per_ordinal(model, H, W, K, device, n_batches=20, B=16):
    model.eval()
    correct = np.zeros(K + 1)
    total = np.zeros(K + 1)
    for _ in range(n_batches):
        img, mask = gen_batch(B, H, W, K, device=device)
        pred = model(img).argmax(1)
        for j in range(1, K + 1):
            m = mask == j
            total[j] += m.sum().item()
            correct[j] += ((pred == j) & m).sum().item()
    acc = correct / np.maximum(total, 1)
    return acc  # acc[1..K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["small", "large", "learnable", "adaptive", "spectral"],
        default="small",
    )
    ap.add_argument("--H", type=int, default=48)
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--dil", type=int, default=8, help="large モードの固定 dilation")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--erf-reg-weight",
        type=float,
        default=0.0,
        help="learnable/spectral モードで rf_spread(σ) を target へ寄せる正則化係数",
    )
    ap.add_argument(
        "--erf-target",
        type=float,
        default=24.0,
        help="目標 rf_spread (大きいほど dilation を伸ばす)",
    )
    ap.add_argument(
        "--init-sigma",
        type=float,
        default=1.0,
        help="spectral モードの初期 σ(px)。大きくすると鶏卵谷を回避できるか検証",
    )
    ap.add_argument("--out-dir", default="/workspace/probe_results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}  mode={args.mode}")

    # 理論 RF (small/large の目安)
    base_d = args.dil if args.mode == "large" else 1
    rf = 1 + args.n_blocks * (3 - 1) * base_d
    print(f"理論RF(初期) ≈ {rf}px  (W={args.W} を覆うには RF≳W が必要)")
    if args.mode == "spectral":
        print("  (spectral は σ=1px から学習開始。σ が伸びれば RF≈数σ に拡大する)")

    model = ProbeNet(
        args.K + 1,
        args.mode,
        args.ch,
        args.n_blocks,
        dil=args.dil,
        init_sigma=args.init_sigma,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.erf_reg_weight > 0:
        print(
            f"🎯 ERF 正則化 ON  λ={args.erf_reg_weight}  target_spread={args.erf_target}"
        )

    model.train()
    for step in range(1, args.steps + 1):
        img, mask = gen_batch(args.batch, args.H, args.W, args.K, device=device)
        opt.zero_grad()
        loss = F.cross_entropy(model(img), mask)
        if args.erf_reg_weight > 0 and args.mode in ("learnable", "spectral"):
            loss = loss + args.erf_reg_weight * model.erf_reg(args.erf_target)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == 1:
            md = model.mean_dilation()
            print(f"  step {step:4d}  loss {loss.item():.4f}  mean_dilation {md:.2f}")

    acc = eval_per_ordinal(model, args.H, args.W, args.K, device)
    overall = float(np.mean(acc[1:]))
    print("\n序数ごとのバー画素精度:")
    for j in range(1, args.K + 1):
        print(f"  ordinal {j}: {acc[j]:.3f}")
    print(f"  平均: {overall:.3f}   最終 mean_dilation: {model.mean_dilation():.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    # CSV
    with open(os.path.join(args.out_dir, "per_ordinal_acc.csv"), "w") as f:
        f.write("ordinal,accuracy\n")
        for j in range(1, args.K + 1):
            f.write(f"{j},{acc[j]:.4f}\n")
        f.write(f"mean,{overall:.4f}\n")
        f.write(f"mean_dilation,{model.mean_dilation():.4f}\n")

    # 図
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, args.K + 1), acc[1:], marker="o")
    plt.ylim(0, 1.02)
    plt.xlabel("ordinal (1=leftmost … K=rightmost, 右ほど大RF必須)")
    plt.ylabel("bar pixel accuracy")
    plt.title(
        f"RF probe [{args.mode}]  mean={overall:.3f}  "
        f"mean_dil={model.mean_dilation():.2f}"
    )
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(args.out_dir, "per_ordinal_acc.png"), dpi=130, bbox_inches="tight"
    )
    plt.close()

    # サンプル可視化 (adaptive は dilation マップも)
    img, mask = gen_batch(1, args.H, args.W, args.K, device=device)
    pred = model(img).argmax(1)[0].cpu().numpy()
    dmap = model.dilation_map(img) if args.mode == "adaptive" else None

    if dmap is not None:
        Hh, Ww = dmap.shape
        bg = dmap[mask[0].cpu().numpy() == 0].mean()
        bar = dmap[mask[0].cpu().numpy() > 0].mean()
        l3 = dmap[:, : Ww // 3].mean()
        r3 = dmap[:, 2 * Ww // 3 :].mean()
        print(
            f"\n[dilation マップ] mean={dmap.mean():.1f} std={dmap.std():.1f}  "
            f"背景={bg:.1f} バー={bar:.1f}  左1/3={l3:.1f} 右1/3={r3:.1f}"
        )
        print("  → std が大きい/右>左/バー≠背景 なら『場所依存の適応』成立")

    nrow = 4 if dmap is not None else 3
    fig, ax = plt.subplots(nrow, 1, figsize=(10, 1.6 * nrow))
    ax[0].imshow(img[0, 0].cpu(), cmap="gray")
    ax[0].set_title("input")
    ax[0].axis("off")
    ax[1].imshow(mask[0].cpu(), cmap="tab10", vmin=0, vmax=args.K)
    ax[1].set_title("GT ordinal")
    ax[1].axis("off")
    ax[2].imshow(pred, cmap="tab10", vmin=0, vmax=args.K)
    ax[2].set_title("prediction")
    ax[2].axis("off")
    if dmap is not None:
        im = ax[3].imshow(dmap, cmap="viridis")
        ax[3].set_title("expected dilation map (large=wide RF)")
        ax[3].axis("off")
        plt.colorbar(im, ax=ax[3], fraction=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "sample.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # adaptive: 列ごとの期待 dilation (左→右で増えるか = "適切な場所依存拡大"の検証)
    if dmap is not None:
        col = dmap.mean(axis=0)  # (W,) 各列の平均期待 dilation
        plt.figure(figsize=(8, 4))
        plt.plot(col)
        plt.xlabel("x 位置 (左=序数小, 右=序数大=大RF必須)")
        plt.ylabel("expected dilation (列平均)")
        plt.title("Content-adaptive dilation vs position (右ほど大きければ適応成功)")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(args.out_dir, "dilation_vs_position.png"),
            dpi=130,
            bbox_inches="tight",
        )
        plt.close()

    print(f"\n保存: {args.out_dir}")


if __name__ == "__main__":
    main()
