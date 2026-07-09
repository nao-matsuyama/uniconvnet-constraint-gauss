# coding:utf-8
"""
評価レポートの共有ロジック — per-sample 指標の集計・表組み・CSV 出力。

evaluate.py（UniConvNet を推論して採点）と evaluate_predictions.py（任意モデルの
保存済み予測を採点）が同一フォーマットのレポートを出せるよう、集計と出力をここに集約する。

入力は「per-sample の M.sample_metrics 出力」のリスト。これによりモデル実装に依存しない
（UniConvNet も U-Net も SegFormer も、予測ラベルマップさえ作れば同じ採点に乗る）。
"""

import csv
import os

import numpy as np

import metrics as M


def aggregate(sample_results, names, num_classes=13):
    """sample_results: [(fname, M.sample_metrics(...) の dict), ...] を集計。

    返り値:
      per_class_mean: {metric: np.ndarray[C]}   背景除く各部位の nanmean
      micro: {"dice", "iou"}                    counts プールの micro 指標
      sample_means: [(fname, {metric: per-sample mean}), ...]
    """
    C = num_classes - 1
    acc = {m: [[] for _ in range(C)] for m in names}
    tot = {k: np.zeros(C, dtype=np.int64) for k in ("tp", "fp", "fn", "tn")}
    sample_means = []
    for fname, res in sample_results:
        for m in names:
            arr = res["per_class"][m]
            for c in range(C):
                if not np.isnan(arr[c]):
                    acc[m][c].append(arr[c])
        for k in tot:
            tot[k] += res["counts"][k]
        sample_means.append((fname, res["mean"]))

    per_class_mean = {
        m: np.array(
            [float(np.mean(acc[m][c])) if acc[m][c] else np.nan for c in range(C)]
        )
        for m in names
    }
    tp, fp, fn = tot["tp"].sum(), tot["fp"].sum(), tot["fn"].sum()
    micro = {
        "dice": float((2 * tp) / max(2 * tp + fp + fn, 1)),
        "iou": float(tp / max(tp + fp + fn, 1)),
    }
    return per_class_mean, micro, sample_means


def build_report(
    per_class_mean,
    micro,
    sample_means,
    names,
    title,
    worst_fracs=(0.1, 0.25),
    num_classes=13,
):
    """整形済みレポート文字列を返す。"""
    C = num_classes - 1
    n = len(sample_means)
    width = 22 + 11 * len(names)
    lines = ["=" * width, title, "=" * width]
    header = f"{'部位':<22}" + "".join(f"{m:>11}" for m in names)
    lines.append(header)
    lines.append("-" * len(header))
    for c in range(C):
        row = f"{M.CLASS_NAMES[c + 1]:<22}"
        for m in names:
            v = per_class_mean[m][c]
            row += "        nan" if np.isnan(v) else f"{v:>11.4f}"
        lines.append(row)
    lines.append("-" * len(header))
    macro = f"{'マクロ平均(部位平均)':<22}"
    for m in names:
        macro += f"{float(np.nanmean(per_class_mean[m])):>11.4f}"
    lines.append(macro)
    lines.append(f"  micro(pooled)  Dice={micro['dice']:.4f}  IoU={micro['iou']:.4f}")

    # worst デシル（per-sample mean Dice 昇順）
    order = sorted(range(n), key=lambda i: _safe(sample_means[i][1].get("dice")))
    lines.append("")
    lines.append(f"[ worst 解析: per-sample mean Dice 昇順, n={n} ]")
    wh = f"{'group':<16}" + "".join(f"{m:>11}" for m in names)
    lines.append(wh)
    lines.append("-" * len(wh))
    groups = [("overall", order)]
    for fr in worst_fracs:
        k = max(1, int(round(n * fr)))
        groups.append((f"worst{int(fr * 100)}%(n={k})", order[:k]))
    for gname, idxs in groups:
        row = f"{gname:<16}"
        for m in names:
            vals = [sample_means[i][1].get(m) for i in idxs]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            row += f"{(np.mean(vals) if vals else float('nan')):>11.4f}"
        lines.append(row)
    lines.append("=" * width)
    return "\n".join(lines)


def write_outputs(out_dir, report, per_class_mean, sample_means, names, num_classes=13):
    """report.txt / per_class.csv / per_sample.csv を保存。"""
    C = num_classes - 1
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(
        os.path.join(out_dir, "per_class.csv"), "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["class"] + names)
        for c in range(C):
            w.writerow(
                [M.CLASS_NAMES[c + 1]] + [f"{per_class_mean[m][c]:.6f}" for m in names]
            )
        w.writerow(
            ["macro_mean"]
            + [f"{float(np.nanmean(per_class_mean[m])):.6f}" for m in names]
        )
    with open(
        os.path.join(out_dir, "per_sample.csv"), "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["fname"] + names)
        for fname, mean in sample_means:
            w.writerow([fname] + [f"{_safe(mean.get(m)):.6f}" for m in names])
    print(f"保存: {out_dir}/report.txt, per_class.csv, per_sample.csv")


def _safe(v):
    return float("nan") if v is None else float(v)
