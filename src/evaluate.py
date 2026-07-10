# coding:utf-8
"""
検証セットの定量評価 — 全指標テーブル + worst（悪いデータ）デシル集計。

これまでは smoothing 付き mean Dice だけだったが、metrics.py を使って
  重なり系: dice / iou / precision / recall / specificity
  境界系  : hd95 / hd / assd / nsd@tau
を per-sample × per-class で算出する。空クラスは NaN 除外（甘い 1.0 を数えない）。

出力（--out-dir 指定時に CSV/TXT も保存）:
  per_class table   … 部位ごとの各指標 mean（背景除く 12 部位）
  overall           … マクロ平均（部位平均）＋ Dice/IoU は micro（プール）も併記
  worst summary     … per-sample mean Dice で昇順 → worst 10%/25% 群の各指標 mean
  per_sample.csv    … サンプルごとの mean 指標（worst 解析・compare 用）

使い方:
  python3 src/evaluate.py \
    --weights /workspace/experiments/run_spectral/best_uniconvnet_unet.pth \
    --data-dir /workspace/scinti_segmentation \
    --out-dir /workspace/eval_results/run_spectral
  # --no-boundary で境界系をスキップ（高速）
"""

import os
import sys

import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(__file__))
import eval_report as R
import metrics as M
from ckpt_utils import build_model_from_checkpoint
from dataset_scinti import ScintiMultiClassDataset

NUM_CLASSES = 13


def build_val_loader(data_dir, batch_size, num_workers, view="both"):
    full = (
        ScintiMultiClassDataset(data_dir=data_dir, view=view)
        if data_dir
        else ScintiMultiClassDataset(view=view)
    )
    train_size = int(0.8 * len(full))
    val_size = len(full) - train_size
    _, val = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return val, loader


def evaluate_model(
    weight_path=None,
    data_dir=None,
    batch_size=16,
    num_workers=2,
    out_dir=None,
    boundary=True,
    spacing=(1.0, 1.0),
    nsd_taus=(1.0, 2.0, 3.0),
    worst_fracs=(0.1, 0.25),
    view="both",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"評価デバイス: {device}")

    if weight_path is None:
        weight_path = "experiments/best_uniconvnet_unet.pth"
    if not os.path.exists(weight_path):
        print(f"重みファイルが見つかりません: {weight_path}")
        return
    print(f"重みを読み込み中: {weight_path}")
    model = build_model_from_checkpoint(
        weight_path, num_classes=NUM_CLASSES, device=device
    )

    val, loader = build_val_loader(data_dir, batch_size, num_workers, view=view)
    print(f"バリデーションサンプル数: {len(val)}")

    names = M.metric_list(nsd_taus=nsd_taus, boundary=boundary)

    sample_results = []  # [(fname, M.sample_metrics(...))]
    print("🔍 評価を開始します...")
    with torch.no_grad():
        gi = 0
        for images, masks in loader:
            images = images.to(device)
            preds = torch.argmax(model(images), dim=1).cpu().numpy()  # (B,H,W)
            gts = masks.numpy()
            for b in range(preds.shape[0]):
                res = M.sample_metrics(
                    preds[b],
                    gts[b],
                    NUM_CLASSES,
                    spacing=spacing,
                    nsd_taus=nsd_taus,
                    boundary=boundary,
                )
                gidx = val.indices[gi]
                fname = os.path.basename(val.dataset.image_files[gidx])
                sample_results.append((fname, res))
                gi += 1

    per_class_mean, micro, sample_means = R.aggregate(
        sample_results, names, NUM_CLASSES
    )
    title = f"定量評価レポート  weights={os.path.basename(weight_path)}  n={len(val)}"
    report = R.build_report(
        per_class_mean,
        micro,
        sample_means,
        names,
        title,
        worst_fracs=worst_fracs,
        num_classes=NUM_CLASSES,
    )
    print("\n" + report)
    if out_dir:
        R.write_outputs(
            out_dir, report, per_class_mean, sample_means, names, NUM_CLASSES
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="学習済み重みの .pth パス")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--no-boundary", action="store_true", help="境界系(HD95/ASSD/NSD)をスキップ"
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="画素間隔 (y x)。省略時は px 単位",
    )
    parser.add_argument("--nsd-taus", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--worst-fracs", type=float, nargs="+", default=[0.1, 0.25])
    parser.add_argument(
        "--view",
        choices=["both", "anterior", "posterior"],
        default="both",
        help="評価するビュー。学習時と揃えること (anterior 学習なら anterior で評価)。",
    )
    args = parser.parse_args()

    evaluate_model(
        weight_path=args.weights,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        boundary=not args.no_boundary,
        spacing=tuple(args.spacing),
        nsd_taus=tuple(args.nsd_taus),
        worst_fracs=tuple(args.worst_fracs),
        view=args.view,
    )
