"""
推論結果の可視化スクリプト

バリデーションセット全体で推論 → Dice スコアで並べ、
  ・ベスト   (Dice 最大)
  ・平均的   (Dice 中央値付近)
  ・ワースト (Dice 最小)
の 3 サンプルを 3 行 × 4 列で保存する。

  列: 入力画像 | GT マスク | 予測マスク | オーバーレイ

使い方:
  python3 src/visualize_predictions.py --weights <pth> --data-dir <dir>
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(__file__))
from dataset_scinti import ScintiMultiClassDataset
from model_uniconvnet_unet import UniConvNet_UNet_13CH
from run_meta import write_manifest

# 13 クラス用カラーパレット (class 0 = 背景 = 黒)
_COLORS = (
    np.array(
        [
            [0, 0, 0],  #  0 background
            [214, 39, 40],  #  1
            [255, 127, 14],  #  2
            [255, 215, 0],  #  3
            [44, 160, 44],  #  4
            [23, 190, 207],  #  5
            [31, 119, 180],  #  6
            [148, 103, 189],  #  7
            [227, 119, 194],  #  8
            [140, 86, 75],  #  9
            [188, 189, 34],  # 10
            [127, 127, 127],  # 11
            [174, 199, 232],  # 12
        ],
        dtype=np.float32,
    )
    / 255.0
)

CLASS_NAMES = [
    "Background",
    "Class 01",
    "Class 02",
    "Class 03",
    "Class 04",
    "Class 05",
    "Class 06",
    "Class 07",
    "Class 08",
    "Class 09",
    "Class 10",
    "Class 11",
    "Class 12",
]


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────
def mask_to_rgb(mask_np):
    """(H, W) int → (H, W, 3) float RGB"""
    rgb = _COLORS[mask_np.clip(0, 12)]
    return rgb


def overlay(image_np, mask_np, alpha=0.5):
    """
    image_np : (H, W) float [0, 1]
    mask_np  : (H, W) int
    return   : (H, W, 3) float
    """
    img_rgb = np.stack([image_np] * 3, axis=-1)
    mask_rgb = mask_to_rgb(mask_np)
    fg = (mask_np > 0)[..., None]
    blended = img_rgb * (1 - alpha * fg) + mask_rgb * (alpha * fg)
    return np.clip(blended, 0, 1)


def sample_dice(pred, gt, num_classes=13):
    """1サンプルのクラス別 Dice（背景除く）と平均を返す。"""
    per_class = []
    for c in range(1, num_classes):
        p = (pred == c).astype(np.float32)
        m = (gt == c).astype(np.float32)
        inter = (p * m).sum()
        union = p.sum() + m.sum()
        per_class.append((2 * inter + 1e-5) / (union + 1e-5))
    return float(np.mean(per_class)), per_class


# ─────────────────────────────────────────────
# 推論
# ─────────────────────────────────────────────
def run_inference(model, val_dataset, device):
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    records = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(device)
            pred = torch.argmax(model(images), dim=1)[0].cpu().numpy()  # (H, W)
            img_np = images[0, 0].cpu().numpy()  # (H, W) grayscale
            gt_np = masks[0].numpy()  # (H, W)

            dice_mean, dice_cls = sample_dice(pred, gt_np)

            global_idx = val_dataset.indices[i]
            fname = os.path.basename(val_dataset.dataset.image_files[global_idx])

            records.append(
                {
                    "dice_mean": dice_mean,
                    "dice_cls": dice_cls,
                    "image": img_np,
                    "gt": gt_np,
                    "pred": pred,
                    "fname": fname,
                }
            )

    records.sort(key=lambda r: r["dice_mean"])
    return records


# ─────────────────────────────────────────────
# サンプル選択
# ─────────────────────────────────────────────
def pick_samples(records):
    n = len(records)
    return [
        ("Worst", records[0]),
        ("Average", records[n // 2]),
        ("Best", records[-1]),
    ]


# ─────────────────────────────────────────────
# 描画
# ─────────────────────────────────────────────
def plot_results(cases, out_dir):
    n_rows = len(cases)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(22, 6 * n_rows),
        gridspec_kw={"wspace": 0.03, "hspace": 0.35},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_headers = [
        "Input Image",
        "Ground Truth",
        "Prediction",
        "Overlay (pred on image)",
    ]
    for j, h in enumerate(col_headers):
        axes[0, j].set_title(h, fontsize=11, fontweight="bold", pad=6)

    for row, (case_name, rec) in enumerate(cases):
        img = rec["image"]
        gt = rec["gt"]
        pred = rec["pred"]
        fname = rec["fname"]
        dmean = rec["dice_mean"]
        dcls = rec["dice_cls"]

        ims = [
            (img, "gray", {"vmin": 0, "vmax": 1}),
            (mask_to_rgb(gt), None, {}),
            (mask_to_rgb(pred), None, {}),
            (overlay(img, pred, 0.45), None, {}),
        ]
        for j, (data, cmap, kwargs) in enumerate(ims):
            axes[row, j].imshow(data, cmap=cmap, **kwargs)
            axes[row, j].axis("off")

        # 行ラベル
        row_label = f"{case_name}\n{fname}\nmean Dice = {dmean:.4f}"
        axes[row, 0].set_ylabel(
            row_label, fontsize=9, rotation=0, labelpad=80, va="center"
        )
        axes[row, 0].yaxis.set_label_coords(-0.35, 0.5)

        # クラス別 Dice を予測パネル下に小テキスト
        dice_lines = [
            "  ".join(f"c{c+1:02d}:{dcls[c]:.2f}" for c in range(0, 6)),
            "  ".join(f"c{c+1:02d}:{dcls[c]:.2f}" for c in range(6, 12)),
        ]
        axes[row, 2].text(
            0.5,
            -0.04,
            "\n".join(dice_lines),
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
            family="monospace",
        )

    # 凡例
    patches = [
        mpatches.Patch(color=_COLORS[c], label=CLASS_NAMES[c]) for c in range(1, 13)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=6,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.suptitle(
        "Inference Results — UniConvNet-T U-Net\n"
        "Worst / Average / Best (by mean bone Dice)",
        fontsize=13,
        y=1.01,
    )

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "prediction_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"保存: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 全サンプルの推論マスクをフォルダに保存
# ─────────────────────────────────────────────
def save_all_predictions(records, out_dir):
    """
    全バリデーションサンプルについて:
      panels/{rank}_dice{d}_{fname}.png  … 入力|GT|予測|オーバーレイ の4枚組
      masks/{fname}_pred.png             … 予測マスク(カラー)単体
    を保存する。records は Dice 昇順 (worst→best) で並んでいる前提。
    """
    panel_dir = os.path.join(out_dir, "panels")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(panel_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    n = len(records)
    for rank, rec in enumerate(records):
        img, gt, pred = rec["image"], rec["gt"], rec["pred"]
        fname = os.path.splitext(rec["fname"])[0]
        dmean = rec["dice_mean"]

        # ① 4 パネル (rank 000 = 最悪, 末尾 = 最良 でソートされる)
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={"wspace": 0.02})
        panels = [
            (img, "gray", {"vmin": 0, "vmax": 1}, "Input"),
            (mask_to_rgb(gt), None, {}, "Ground Truth"),
            (mask_to_rgb(pred), None, {}, "Prediction"),
            (overlay(img, pred, 0.45), None, {}, "Overlay"),
        ]
        for ax, (data, cmap, kw, title) in zip(axes, panels):
            ax.imshow(data, cmap=cmap, **kw)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{rec['fname']}   mean Dice = {dmean:.4f}", fontsize=11)
        panel_path = os.path.join(panel_dir, f"{rank:03d}_dice{dmean:.4f}_{fname}.png")
        plt.savefig(panel_path, dpi=120, bbox_inches="tight")
        plt.close()

        # ② 予測マスク(カラー)単体
        plt.figure(figsize=(5, 5))
        plt.imshow(mask_to_rgb(pred))
        plt.axis("off")
        mask_path = os.path.join(mask_dir, f"{fname}_pred.png")
        plt.savefig(mask_path, dpi=120, bbox_inches="tight", pad_inches=0)
        plt.close()

        if (rank + 1) % 10 == 0 or rank == n - 1:
            print(f"  保存 {rank + 1}/{n}")

    print(f"全サンプル保存完了:\n  パネル: {panel_dir}\n  マスク: {mask_dir}")


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default="/workspace/pred_results")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="全バリデーションサンプルの推論マスク/パネルを保存する",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    write_manifest(
        args.out_dir,
        "visualize_predictions",
        {"data_dir": args.data_dir, "save_all": args.save_all},
        models={"model": args.weights},
    )

    full_dataset = (
        ScintiMultiClassDataset(data_dir=args.data_dir)
        if args.data_dir
        else ScintiMultiClassDataset()
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"バリデーションサンプル数: {val_size}")

    if args.weights and os.path.exists(args.weights):
        # separable/spectral/adaptive も dw_mode 自動判別で構築・load。
        from ckpt_utils import build_model_from_checkpoint

        model = build_model_from_checkpoint(
            args.weights, num_classes=13, device=device, eval_mode=False
        )
        print(f"重みロード: {args.weights}")
    else:
        print("重みなし（ランダム初期化）")
        model = UniConvNet_UNet_13CH(num_classes=13).to(device)

    print("推論中...")
    records = run_inference(model, val_dataset, device)

    print("\nDice スコア分布:")
    dices = [r["dice_mean"] for r in records]
    print(
        f"  最小: {min(dices):.4f}  中央: {np.median(dices):.4f}  最大: {max(dices):.4f}"
    )

    cases = pick_samples(records)
    for name, rec in cases:
        print(f"  {name:8s}: {rec['fname']}  Dice={rec['dice_mean']:.4f}")

    plot_results(cases, args.out_dir)

    if args.save_all:
        print("\n全サンプルの推論マスクを保存中...")
        save_all_predictions(records, os.path.join(args.out_dir, "all"))

    print(f"\n完了。結果フォルダ: {args.out_dir}")


if __name__ == "__main__":
    main()
