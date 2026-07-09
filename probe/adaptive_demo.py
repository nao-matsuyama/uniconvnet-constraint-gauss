# coding:utf-8
"""
コンテンツ適応デモ — 「適切な受容野(RF)＝場所依存」を最も直感的に見せる最小実験。

課題: マーカー距離ビン分類
  入力 (1,H,W): 黒背景に 1 個の明るいマーカー(小さな正方形)をランダム配置
  正解 (H,W)  : 各画素を「マーカーからの距離ビン」(0..K-1) でラベル
                (0=最も近いリング … K-1=最も遠いリング)
  → ある画素のラベルを当てるには「マーカーが見える」必要がある。
    マーカーから遠い画素ほど、それを知覚するのに大きい RF が要る。
    = 必要 RF は距離に比例して増える(原理的に場所依存)。

3 モード比較:
  small     … 通常 depthwise 3x3 (dilation 1) → RF 小。遠ビンで失敗するはず。
  large     … 固定大 dilation depthwise → RF 大。全域解けるが常に広域を計算(無駄)。
  adaptive  … ContentAdaptiveDW (画素ごと dilation ゲート) → 近=小 RF / 遠=大 RF を
              自分で割り当てる。large 並みの精度で、かつ「距離に応じた dilation」を示す。

決定的な図 dilation_vs_distance.png:
  横軸=マーカーからの距離, 縦軸=期待 dilation E[d](x)。
  これが右肩上がりなら「ゲートが遠方ほど大 RF を割り当てる＝場所依存の適切な拡大」が成立。
  (序数バー課題では位置が単調でなかったが、距離ビン課題では必要 RF が距離に厳密比例するため
   単調性が原理的に保証される。)

使い方:
  python3 probe/adaptive_demo.py --mode small    --out-dir /workspace/probe_results/demo_small
  python3 probe/adaptive_demo.py --mode large    --out-dir /workspace/probe_results/demo_large
  python3 probe/adaptive_demo.py --mode adaptive --out-dir /workspace/probe_results/demo_adaptive
  # CPU スモーク確認: --steps 60 --H 64 --W 64 を付ける
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
from models.spectral_gaussian_dw import SpectralGaussianDW


# ──────────────────────────────────────────────
# 合成データ生成 (オンザフライ)
#   img:  (B,1,H,W)  1 個の明るいマーカー
#   mask: (B,H,W)    マーカーからの距離ビン 0..K-1
#   dist: (B,H,W)    マーカー中心からのユークリッド距離 (解析用)
# ──────────────────────────────────────────────
def gen_batch(B, H, W, K, marker=7, device="cpu"):
    img = torch.zeros(B, 1, H, W, device=device)
    mask = torch.zeros(B, H, W, dtype=torch.long, device=device)
    dist = torch.zeros(B, H, W, device=device)

    ys = torch.arange(H, device=device).view(H, 1).float()
    xs = torch.arange(W, device=device).view(1, W).float()
    maxd = float(np.hypot(H, W))  # 距離の理論最大 (ビン幅の基準)
    ring = maxd / K

    margin = marker
    for b in range(B):
        my = int(np.random.randint(margin, H - margin))
        mx = int(np.random.randint(margin, W - margin))
        y0, y1 = my - marker // 2, my - marker // 2 + marker
        x0, x1 = mx - marker // 2, mx - marker // 2 + marker
        img[b, 0, y0:y1, x0:x1] = 1.0

        d = torch.sqrt((ys - my) ** 2 + (xs - mx) ** 2)  # (H,W)
        dist[b] = d
        mask[b] = torch.clamp((d / ring).long(), 0, K - 1)
    return img, mask, dist


# ──────────────────────────────────────────────
# プローブモデル (downsample 無し → RF が透明)
#   RF ≈ 1 + n_blocks * (k-1) * dilation
# ──────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, ch, mode, k=3, dil=16, dilations=(1, 4, 16, 32)):
        super().__init__()
        self.mode = mode
        if mode == "adaptive":
            self.dw = ContentAdaptiveDW(ch, kernel_size=k, dilations=dilations)
        elif mode == "spectral":
            # 周波数ガウス。スクラッチ学習なので init_gamma=1 で spectral 枝を有効化。
            self.dw = SpectralGaussianDW(
                ch, kernel_size=k, init_sigma=1.0, init_gamma=1.0, max_sigma=64.0
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
        self,
        num_classes,
        mode,
        ch=32,
        n_blocks=6,
        k=3,
        dil=16,
        dilations=(1, 4, 16, 32),
    ):
        super().__init__()
        self.stem = nn.Conv2d(1, ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            [Block(ch, mode, k, dil, dilations) for _ in range(n_blocks)]
        )
        self.head = nn.Conv2d(ch, num_classes, 1)

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)

    def mean_dilation(self):
        # spectral: per-channel σ(px) 平均を同じ「RFの広さ」スロットで返す。
        ss = [
            m.mean_sigma() for m in self.modules() if isinstance(m, SpectralGaussianDW)
        ]
        if ss:
            return float(np.mean(ss))
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


# ──────────────────────────────────────────────
# 評価: 距離ビンごとの画素精度
# ──────────────────────────────────────────────
@torch.no_grad()
def eval_per_bin(model, H, W, K, marker, device, n_batches=20, B=16):
    model.eval()
    correct = np.zeros(K)
    total = np.zeros(K)
    for _ in range(n_batches):
        img, mask, _ = gen_batch(B, H, W, K, marker, device=device)
        pred = model(img).argmax(1)
        for j in range(K):
            m = mask == j
            total[j] += m.sum().item()
            correct[j] += ((pred == j) & m).sum().item()
    return correct / np.maximum(total, 1)  # acc[0..K-1]


# ──────────────────────────────────────────────
# 決定的図: 距離 → 期待 dilation の集計 (adaptive 専用)
# ──────────────────────────────────────────────
@torch.no_grad()
def dilation_vs_distance(model, H, W, K, marker, device, n_samples=24, nbins=40):
    maxd = float(np.hypot(H, W))
    edges = np.linspace(0, maxd, nbins + 1)
    sum_d = np.zeros(nbins)
    cnt = np.zeros(nbins)
    for _ in range(n_samples):
        img, _, dist = gen_batch(1, H, W, K, marker, device=device)
        dmap = model.dilation_map(img)  # (H,W)
        if dmap is None:
            return None, None
        d_flat = dist[0].cpu().numpy().ravel()
        e_flat = dmap.ravel()
        idx = np.clip(np.digitize(d_flat, edges) - 1, 0, nbins - 1)
        np.add.at(sum_d, idx, e_flat)
        np.add.at(cnt, idx, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_d = sum_d / np.maximum(cnt, 1)
    valid = cnt > 0
    return centers[valid], mean_d[valid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["small", "large", "adaptive", "spectral"],
        default="adaptive",
    )
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--K", type=int, default=5, help="距離ビン数")
    ap.add_argument("--marker", type=int, default=7)
    ap.add_argument("--ch", type=int, default=32)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--dil", type=int, default=16, help="large モードの固定 dilation")
    ap.add_argument(
        "--dilations",
        type=int,
        nargs="+",
        default=[1, 4, 16, 32],
        help="adaptive の dilation 枝",
    )
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="/workspace/probe_results/demo")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイス: {device}  mode={args.mode}")

    base_d = args.dil if args.mode == "large" else 1
    rf = 1 + args.n_blocks * (3 - 1) * base_d
    maxd = float(np.hypot(args.H, args.W))
    print(
        f"理論RF(small/large) ≈ {rf}px   最遠ビンの距離 ≈ {maxd:.0f}px "
        f"(これを覆う RF が必要)"
    )

    model = ProbeNet(
        args.K,
        args.mode,
        args.ch,
        args.n_blocks,
        dil=args.dil,
        dilations=tuple(args.dilations),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        img, mask, _ = gen_batch(
            args.batch, args.H, args.W, args.K, args.marker, device=device
        )
        opt.zero_grad()
        loss = F.cross_entropy(model(img), mask)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == 1:
            print(
                f"  step {step:4d}  loss {loss.item():.4f}  "
                f"mean_dilation {model.mean_dilation():.2f}"
            )

    acc = eval_per_bin(model, args.H, args.W, args.K, args.marker, device)
    overall = float(np.mean(acc))
    print("\n距離ビンごとの画素精度 (0=近 … K-1=遠):")
    for j in range(args.K):
        print(
            f"  bin {j} ({'近' if j == 0 else '遠' if j == args.K - 1 else '中'}): {acc[j]:.3f}"
        )
    print(f"  平均: {overall:.3f}   最終 mean_dilation: {model.mean_dilation():.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "per_bin_acc.csv"), "w") as f:
        f.write("bin,accuracy\n")
        for j in range(args.K):
            f.write(f"{j},{acc[j]:.4f}\n")
        f.write(f"mean,{overall:.4f}\n")
        f.write(f"mean_dilation,{model.mean_dilation():.4f}\n")

    # 図1: 距離ビンごとの精度
    plt.figure(figsize=(7, 5))
    plt.plot(range(args.K), acc, marker="o")
    plt.ylim(0, 1.02)
    plt.xlabel("distance bin (0=近 … K-1=遠=大RF必須)")
    plt.ylabel("pixel accuracy")
    plt.title(
        f"distance-bin probe [{args.mode}]  mean={overall:.3f}  "
        f"mean_dil={model.mean_dilation():.2f}"
    )
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(args.out_dir, "per_bin_acc.png"), dpi=130, bbox_inches="tight"
    )
    plt.close()

    # 図2: サンプル可視化
    img, mask, dist = gen_batch(1, args.H, args.W, args.K, args.marker, device=device)
    pred = model(img).argmax(1)[0].cpu().numpy()
    dmap = model.dilation_map(img) if args.mode == "adaptive" else None

    nrow = 4 if dmap is not None else 3
    fig, ax = plt.subplots(1, nrow, figsize=(3.2 * nrow, 3.4))
    ax[0].imshow(img[0, 0].cpu(), cmap="gray")
    ax[0].set_title("input (marker)")
    ax[0].axis("off")
    ax[1].imshow(mask[0].cpu(), cmap="viridis", vmin=0, vmax=args.K - 1)
    ax[1].set_title("GT dist-bin")
    ax[1].axis("off")
    ax[2].imshow(pred, cmap="viridis", vmin=0, vmax=args.K - 1)
    ax[2].set_title("prediction")
    ax[2].axis("off")
    if dmap is not None:
        im = ax[3].imshow(dmap, cmap="magma")
        ax[3].set_title("expected dilation\n(明=大RF)")
        ax[3].axis("off")
        plt.colorbar(im, ax=ax[3], fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "sample.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # 図3 (決定的): 距離 → 期待 dilation
    if dmap is not None:
        xc, yc = dilation_vs_distance(
            model, args.H, args.W, args.K, args.marker, device
        )
        if xc is not None:
            # 単調性の目安: Spearman 相関
            from numpy import argsort

            rx = argsort(argsort(xc))
            ry = argsort(argsort(yc))
            n = len(xc)
            spearman = (
                1 - 6 * np.sum((rx - ry) ** 2) / (n * (n**2 - 1))
                if n > 1
                else float("nan")
            )
            print(
                f"\n[距離 vs 期待dilation] Spearman ρ = {spearman:+.3f}  "
                f"(+1 に近いほど『遠いほど大 dilation』= 場所依存適応)"
            )
            plt.figure(figsize=(8, 5))
            plt.plot(xc, yc, marker="o")
            plt.xlabel("マーカーからの距離 [px] (右ほど大RF必須)")
            plt.ylabel("expected dilation E[d](x) (距離ビン平均)")
            plt.title(
                f"Content-adaptive dilation vs distance  "
                f"(Spearman ρ={spearman:+.2f}, 右肩上がり=適応成功)"
            )
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(args.out_dir, "dilation_vs_distance.png"),
                dpi=130,
                bbox_inches="tight",
            )
            plt.close()

    print(f"\n保存: {args.out_dir}")


if __name__ == "__main__":
    main()
