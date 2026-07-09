# coding:utf-8
"""
モデル非依存スコアラ — 保存済み予測ラベルマップを metrics.py で採点する。

UniConvNet も U-Net も DeepLabV3+ も SegFormer も、最終的に「seed42 の val 分割に対する
予測ラベルマップ (H,W, 値=0..12)」さえ吐けば、ここで同一の指標・同一の集計で採点できる。
これが SOTA 比較の統一ハーネスの口：実装手段（smp / torchvision / 自前）を問わない。

予測の受け渡し形式（--pred-dir 内）
  方式A: <fname>.npy        … 各 .mhd と同じ basename の int ラベルマップ (H,W)
  方式B: predictions.npz    … キー=fname(拡張子なしでも可), 値=ラベルマップ
GT は dataset_scinti の seed42 val 分割から取得（学習スクリプトと同じ分割規約）。

予測の生成側は別途用意する（各ベースラインの推論スクリプトが val を回して上記を保存）。
そのため学習・推論の実装方式を後で決めても、採点はこのスクリプトに一本化される。

使い方:
  python3 src/evaluate_predictions.py \
    --pred-dir /workspace/preds/unet_baseline \
    --data-dir /workspace/scinti_segmentation \
    --label SOTA-UNet \
    --out-dir /workspace/eval_results/unet_baseline
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import random_split

sys.path.append(os.path.dirname(__file__))
import eval_report as R
import metrics as M
from dataset_scinti import ScintiMultiClassDataset

NUM_CLASSES = 13


def get_val_split(data_dir):
    """学習・他評価スクリプトと同一の seed42 val 分割を返す（Subset）。"""
    full = (
        ScintiMultiClassDataset(data_dir=data_dir)
        if data_dir
        else ScintiMultiClassDataset()
    )
    train_size = int(0.8 * len(full))
    val_size = len(full) - train_size
    _, val = random_split(
        full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    return val


def _stem(name):
    return os.path.splitext(os.path.basename(name))[0]


def load_predictions(pred_dir):
    """pred_dir から {stem: ndarray(H,W)} を読み込む（方式A/B 両対応）。"""
    npz = os.path.join(pred_dir, "predictions.npz")
    if os.path.exists(npz):
        data = np.load(npz)
        return {_stem(k): data[k] for k in data.files}
    out = {}
    for f in os.listdir(pred_dir):
        if f.endswith(".npy"):
            out[_stem(f)] = np.load(os.path.join(pred_dir, f))
    if not out:
        raise FileNotFoundError(
            f"予測が見つかりません: {pred_dir}\n"
            "  <fname>.npy 群か predictions.npz を置いてください。"
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, help="予測ラベルマップの保存先")
    parser.add_argument(
        "--data-dir", default=None, help="GT(scinti_segmentation)の場所"
    )
    parser.add_argument("--label", default="model", help="レポートに出すモデル名")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--spacing", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--nsd-taus", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--worst-fracs", type=float, nargs="+", default=[0.1, 0.25])
    args = parser.parse_args()

    boundary = not args.no_boundary
    spacing = tuple(args.spacing)
    nsd_taus = tuple(args.nsd_taus)
    names = M.metric_list(nsd_taus=nsd_taus, boundary=boundary)

    val = get_val_split(args.data_dir)
    preds = load_predictions(args.pred_dir)
    print(f"val サンプル数: {len(val)}  予測ファイル数: {len(preds)}")

    sample_results = []
    missing = []
    for i in range(len(val)):
        gidx = val.indices[i]
        fname = os.path.basename(val.dataset.image_files[gidx])
        stem = _stem(fname)
        if stem not in preds:
            missing.append(fname)
            continue
        gt = val[i][1].numpy() if hasattr(val[i][1], "numpy") else np.asarray(val[i][1])
        pred = np.asarray(preds[stem])
        if pred.shape != gt.shape:
            raise ValueError(
                f"形状不一致 {fname}: pred {pred.shape} vs gt {gt.shape}。"
                "予測は GT と同じ (H,W) で保存してください。"
            )
        res = M.sample_metrics(
            pred,
            gt,
            NUM_CLASSES,
            spacing=spacing,
            nsd_taus=nsd_taus,
            boundary=boundary,
        )
        sample_results.append((fname, res))

    if missing:
        print(
            f"⚠️ 予測が無い val サンプル {len(missing)} 件はスキップ（例: {missing[:3]}）"
        )
    if not sample_results:
        raise RuntimeError(
            "採点できたサンプルが 0 件。fname の対応を確認してください。"
        )

    per_class_mean, micro, sample_means = R.aggregate(
        sample_results, names, NUM_CLASSES
    )
    title = (
        f"定量評価レポート（保存済み予測）  model={args.label}  n={len(sample_results)}"
    )
    report = R.build_report(
        per_class_mean,
        micro,
        sample_means,
        names,
        title,
        worst_fracs=tuple(args.worst_fracs),
        num_classes=NUM_CLASSES,
    )
    print("\n" + report)
    if args.out_dir:
        R.write_outputs(
            args.out_dir, report, per_class_mean, sample_means, names, NUM_CLASSES
        )


if __name__ == "__main__":
    main()
