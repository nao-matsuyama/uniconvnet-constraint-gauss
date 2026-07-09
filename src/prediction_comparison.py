"""
複数モデルの推論マスクを横並びにする比較図 prediction_comparison.png を作る。

visualize_predictions.py が単一モデル (Worst/Avg/Best × 入力|GT|予測|Overlay) なのに対し、
こちらは **同一サンプルに対する複数機構の予測を1枚**に並べる:

  行 = 選んだサンプル (既定: 第1モデル基準の Worst/Average/Best)
  列 = 入力 | GT | <model1 予測> | <model2 予測> | ...   (各予測セル下に mean Dice)

対象機構は build_model_from_checkpoint が dw_mode を自動判別するので混在可:
  baseline(dense=DilatedDWConv d=1) / dilation(学習 dilation) / spectral / separable。

使い方:
  python3 src/prediction_comparison.py \
    --weights <base.pth> <spectral.pth> <dilation.pth> <separable.pth> \
    --labels baseline spectral dilation separable \
    --out-dir /workspace/pred_compare
  # 難症例に絞るなら --worst 5 (第1モデル基準の worst 5 行)
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split

sys.path.append(os.path.dirname(__file__))
from ckpt_utils import build_model_from_checkpoint
from dataset_scinti import ScintiMultiClassDataset
from run_meta import write_manifest
from visualize_predictions import _COLORS, CLASS_NAMES, mask_to_rgb, sample_dice


def _labels_from_weights(weights, labels):
    if labels and len(labels) == len(weights):
        return list(labels)
    out = []
    for w in weights:
        # experiments/run_xxxx_<tag>/best_uniconvnet_unet.pth → run フォルダ名
        out.append(os.path.basename(os.path.dirname(os.path.abspath(w))))
    return out


@torch.no_grad()
def predict_one(model, img_tensor, device):
    """img_tensor (3,H,W) → 予測ラベルマップ (H,W) int。"""
    logits = model(img_tensor.unsqueeze(0).to(device))
    return torch.argmax(logits, dim=1)[0].cpu().numpy()


@torch.no_grad()
def dice_over_val(model, val_dataset, device):
    """val 全サンプルの mean Dice を配列で返す (サンプル選択用, 第1モデル基準)。"""
    model.eval()
    dices = []
    for i in range(len(val_dataset)):
        img, gt = val_dataset[i]
        pred = predict_one(model, img, device)
        d, _ = sample_dice(pred, gt.numpy())
        dices.append(d)
    return np.array(dices)


def pick_positions(scores, worst_n):
    """行に使う val 内 position のリストと表示名を返す。scores は小さいほど難症例。"""
    order = np.argsort(scores)  # 昇順 = worst→best
    if worst_n and worst_n > 0:
        pos = [int(i) for i in order[:worst_n]]
        names = [f"Worst #{k+1}" for k in range(len(pos))]
        return pos, names
    n = len(scores)
    pos = [int(order[0]), int(order[n // 2]), int(order[-1])]
    names = ["Worst", "Average", "Best"]
    return pos, names


def plot_grid(rows, labels, out_dir, fname="prediction_comparison.png"):
    """rows: list of dict(name, fname, img, gt, preds[list], dices[list])。"""
    n_rows = len(rows)
    n_cols = 2 + len(labels)  # 入力 + GT + 各モデル
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.6 * n_rows),
        gridspec_kw={"wspace": 0.03, "hspace": 0.28},
    )
    axes = np.atleast_2d(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    headers = ["Input", "Ground Truth"] + list(labels)
    for j, h in enumerate(headers):
        axes[0, j].set_title(h, fontsize=11, fontweight="bold", pad=6)

    for r, row in enumerate(rows):
        img, gt = row["img"], row["gt"]
        panels = [(img, "gray", {"vmin": 0, "vmax": 1}), (mask_to_rgb(gt), None, {})]
        panels += [(mask_to_rgb(p), None, {}) for p in row["preds"]]
        for j, (data, cmap, kw) in enumerate(panels):
            axes[r, j].imshow(data, cmap=cmap, **kw)
            axes[r, j].axis("off")
        # 各予測セル下に mean Dice (第1モデル基準の best を太字強調)
        best_k = int(np.argmax(row["dices"])) if row["dices"] else -1
        for k, d in enumerate(row["dices"]):
            axes[r, 2 + k].text(
                0.5,
                -0.06,
                f"Dice {d:.4f}" + ("  *best" if k == best_k else ""),
                transform=axes[r, 2 + k].transAxes,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold" if k == best_k else "normal",
                family="monospace",
            )
        row_label = f"{row['name']}\n{row['fname']}"
        axes[r, 0].set_ylabel(row_label, fontsize=9, rotation=0, va="center")
        axes[r, 0].yaxis.set_label_coords(-0.35, 0.5)

    patches = [
        mpatches.Patch(color=_COLORS[c], label=CLASS_NAMES[c]) for c in range(1, 13)
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=6,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.03),
    )
    plt.suptitle(
        "Prediction comparison across mechanisms (same samples)\n"
        "Dice below each cell = mean bone Dice  (*best = best in row)",
        fontsize=13,
        y=1.01,
    )
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default="/workspace/pred_compare")
    parser.add_argument(
        "--worst",
        type=int,
        default=0,
        help="0(既定)=Worst/Average/Best の3行。N>0 で worst N 行(難症例のみ)。",
    )
    parser.add_argument(
        "--select-by",
        choices=["mean", "first"],
        default="mean",
        help="サンプル選択の基準。mean(既定)=全モデルの mean Dice 平均が最悪"
        "(=どのモデルでも当てにくい難症例)。first=第1モデル基準のみ(速い)。",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    labels = _labels_from_weights(args.weights, args.labels)

    write_manifest(
        args.out_dir,
        "prediction_comparison",
        {"data_dir": args.data_dir, "worst": args.worst},
        models={l: w for l, w in zip(labels, args.weights)},
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

    # モデルを全て構築 (dw_mode 自動判別)
    models = []
    for w, l in zip(args.weights, labels):
        print(f"  ロード [{l}]: {w}")
        models.append(
            build_model_from_checkpoint(
                w, num_classes=13, device=device, eval_mode=True
            )
        )

    # サンプル選択の基準スコアを計算 (小さいほど難症例)。
    if args.select_by == "mean":
        print(
            f"全 {len(models)} モデルで val 全サンプルの Dice を計算(mean で難症例選択)..."
        )
        dmat = np.stack([dice_over_val(m, val_dataset, device) for m in models])
        scores = dmat.mean(axis=0)
    else:
        print("第1モデルで val 全サンプルの Dice を計算しサンプル選択...")
        scores = dice_over_val(models[0], val_dataset, device)
    positions, names = pick_positions(scores, args.worst)

    rows = []
    for pos, name in zip(positions, names):
        img, gt = val_dataset[pos]
        gt_np = gt.numpy()
        img_np = img[0].numpy()
        global_idx = val_dataset.indices[pos]
        fname = os.path.basename(val_dataset.dataset.image_files[global_idx])
        preds, dvals = [], []
        for m in models:
            p = predict_one(m, img, device)
            preds.append(p)
            dvals.append(sample_dice(p, gt_np)[0])
        rows.append(
            {
                "name": name,
                "fname": fname,
                "img": img_np,
                "gt": gt_np,
                "preds": preds,
                "dices": dvals,
            }
        )
        print(
            f"  {name:10s} {fname}  "
            + "  ".join(f"{l}={d:.4f}" for l, d in zip(labels, dvals))
        )

    plot_grid(rows, labels, args.out_dir)
    print(f"\n完了。結果フォルダ: {args.out_dir}")


if __name__ == "__main__":
    main()
