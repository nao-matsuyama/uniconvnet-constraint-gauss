# coding:utf-8
"""
複数モデルの per-sample 比較 — 「RF 拡大が悪いデータ（worst）を救うか」を全指標で検証する。

同じ val 分割 (seed 42) で各モデルを推論し、サンプルごとに全指標（dice/iou/precision/
recall/specificity + 境界系 hd95/hd/assd/nsd@tau）を対応付ける（ペア比較）。狙いは
「平均では差が出なくても、局所が曖昧な最悪例では広域文脈=RF 拡大が効く」という仮説の検証。
特に境界系（hd95/assd）は worst 群で差が出やすい。

ベースライン（先頭モデル）の per-sample mean Dice 昇順でサンプルを並べ、worst k% 群を
固定して全モデルを同じ群で比較する（ペア比較）。各指標について Wilcoxon 符号順位検定も実施。
境界系は「小さいほど良い」ので改善方向の符号を自動で反転して判定する。

出力 (--out-dir):
  per_sample_metrics.csv   … サンプル×モデル×指標の mean（ベースライン Dice 昇順）
  summary.txt              … overall / worst 群の各指標 mean ＋ Δ・Wilcoxon
  per_class_on_worst.csv   … worst 群でのクラス別 mean（--plot-metric の指標）
  comparison.png           … (左)昇順 per-sample 折れ線 / (右)集計バー（--plot-metric）

使い方:
  python3 src/compare_models.py \
    --weights /workspace/experiments/run_baseline/best_uniconvnet_unet.pth \
              /workspace/experiments/run_spectral/best_uniconvnet_unet.pth \
    --labels baseline spectral \
    --data-dir /workspace/scinti_segmentation \
    --out-dir /workspace/compare_results/spectral_vs_baseline
  # --no-boundary で境界系をスキップ（高速）。--plot-metric hd95 で図の指標を変更。
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(__file__))
import metrics as M
from ckpt_utils import build_model_from_checkpoint
from dataset_scinti import ScintiMultiClassDataset
from run_meta import write_manifest

NUM_CLASSES = 13


def get_val_dataset(data_dir, view="both"):
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
    return val


def infer_metrics(weights, val_dataset, device, boundary, spacing, nsd_taus):
    """weights を読み込み、val 全サンプルの per-sample 指標を fname キーで返す。

    返り値: {fname: {"mean": {metric: val}, "per_class": {metric: np.ndarray}}}
    spectral/adaptive チェックポイントも自動判別して正しく構築する。
    """
    model = build_model_from_checkpoint(weights, num_classes=NUM_CLASSES, device=device)

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    out = {}
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images = images.to(device)
            pred = torch.argmax(model(images), dim=1)[0].cpu().numpy()
            gt = masks[0].numpy()
            res = M.sample_metrics(
                pred,
                gt,
                NUM_CLASSES,
                spacing=spacing,
                nsd_taus=nsd_taus,
                boundary=boundary,
            )
            gidx = val_dataset.indices[i]
            fname = os.path.basename(val_dataset.dataset.image_files[gidx])
            out[fname] = {"mean": res["mean"], "per_class": res["per_class"]}
    return out


def _vals(results, label, fnames, metric):
    """指定モデル・群・指標の per-sample 値（NaN 除外）。"""
    out = []
    for fn in fnames:
        v = results[label][fn]["mean"].get(metric)
        if v is not None and not np.isnan(v):
            out.append(v)
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="比較するモデルの .pth を複数。先頭をベースラインとして昇順ソート",
    )
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default="/workspace/compare_results")
    parser.add_argument("--worst-fracs", type=float, nargs="+", default=[0.1, 0.25])
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--spacing", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--nsd-taus", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument(
        "--plot-metric", default="dice", help="図と per_class_on_worst の指標"
    )
    parser.add_argument(
        "--view",
        choices=["both", "anterior", "posterior"],
        default="both",
        help="比較するビュー。学習時と揃えること (anterior 学習同士なら anterior)。",
    )
    args = parser.parse_args()

    boundary = not args.no_boundary
    spacing = tuple(args.spacing)
    nsd_taus = tuple(args.nsd_taus)
    names = M.metric_list(nsd_taus=nsd_taus, boundary=boundary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    if args.labels and len(args.labels) == len(args.weights):
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.dirname(w)) for w in args.weights]
    print("比較モデル:")
    for l, w in zip(labels, args.weights):
        print(f"  {l}: {w}")

    os.makedirs(args.out_dir, exist_ok=True)
    write_manifest(
        args.out_dir,
        "compare_models",
        {
            "data_dir": args.data_dir,
            "worst_fracs": args.worst_fracs,
            "boundary": boundary,
            "metrics": names,
        },
        models=dict(zip(labels, args.weights)),
    )

    val = get_val_dataset(args.data_dir, view=args.view)
    print(f"バリデーションサンプル数: {len(val)}")

    results = {}
    for l, w in zip(labels, args.weights):
        print(f"推論中: {l}")
        results[l] = infer_metrics(w, val, device, boundary, spacing, nsd_taus)

    # 全モデル共通の fname
    fnames = set(results[labels[0]])
    for l in labels[1:]:
        fnames &= set(results[l])
    fnames = list(fnames)

    base = labels[0]
    # ベースラインの mean Dice 昇順（worst 先頭）
    order = sorted(fnames, key=lambda f: _safe(results[base][f]["mean"].get("dice")))
    n = len(order)

    # 群の定義（ベースラインの worst k% で固定）
    worst_sets = {"overall": order}
    for fr in args.worst_fracs:
        k = max(1, int(round(n * fr)))
        worst_sets[f"worst{int(fr * 100)}%"] = order[:k]

    # ① per_sample_metrics.csv
    csv_path = os.path.join(args.out_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["rank", "fname"]
        for m in names:
            head += [f"{l}:{m}" for l in labels]
        w.writerow(head)
        for rank, fn in enumerate(order):
            row = [rank, fn]
            for m in names:
                row += [f"{_safe(results[l][fn]['mean'].get(m)):.4f}" for l in labels]
            w.writerow(row)
    print(f"保存: {csv_path}")

    # ② summary.txt — 各指標 × 各群 × 各モデル + Δ/Wilcoxon
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    lines = [
        "=" * 78,
        " モデル比較サマリ（per-sample, 全指標）",
        "=" * 78,
        f" val n={n}  ベースライン={base}  群はベースラインの mean Dice 昇順で固定",
        "",
    ]
    for m in names:
        arrow = "↓小さいほど良い" if not M.higher_is_better(m) else "↑大きいほど良い"
        lines.append(f"[{m}]  ({arrow})")
        header = f"  {'group':<16}" + "".join(f"{l:>16}" for l in labels)
        lines.append(header)
        for gname, gfn in worst_sets.items():
            row = f"  {gname:<16}"
            for l in labels:
                v = _vals(results, l, gfn, m)
                row += f"{(v.mean() if len(v) else float('nan')):>16.4f}"
            lines.append(row)
        # worst 最小 frac 群でベースライン比 Δ ＋ Wilcoxon
        fr0 = min(args.worst_fracs)
        wf = worst_sets[f"worst{int(fr0 * 100)}%"]
        base_pairs = {fn: results[base][fn]["mean"].get(m) for fn in wf}
        for l in labels:
            if l == base:
                continue
            # ペア（両方 NaN でない fname）
            diffs = []
            for fn in wf:
                a = results[l][fn]["mean"].get(m)
                b = base_pairs[fn]
                if (
                    a is not None
                    and b is not None
                    and not np.isnan(a)
                    and not np.isnan(b)
                ):
                    diffs.append(a - b)
            diffs = np.array(diffs)
            if len(diffs) == 0:
                lines.append(f"    {l} vs {base}: ペアなし")
                continue
            d = float(diffs.mean())
            improved = (d > 0) if M.higher_is_better(m) else (d < 0)
            mark = "改善" if d != 0 and improved else ("悪化" if d != 0 else "→")
            pstr = ""
            if wilcoxon is not None and not np.allclose(diffs, 0):
                try:
                    _, p = wilcoxon(diffs)
                    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
                    pstr = f"  p={p:.3g} {sig}"
                except Exception:
                    pstr = ""
            lines.append(
                f"    {l} vs {base} @worst{int(fr0*100)}%: Δ={d:+.4f} {mark}{pstr} (n={len(diffs)})"
            )
        lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # ③ per_class_on_worst.csv（--plot-metric 指標, worst 最小 frac 群）
    pm = args.plot_metric
    fr0 = min(args.worst_fracs)
    wf = worst_sets[f"worst{int(fr0 * 100)}%"]
    pc_path = os.path.join(args.out_dir, "per_class_on_worst.csv")
    with open(pc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"class ({pm})"] + labels)
        for c in range(NUM_CLASSES - 1):
            row = [M.CLASS_NAMES[c + 1]]
            for l in labels:
                vals = [results[l][fn]["per_class"][pm][c] for fn in wf]
                vals = [v for v in vals if not np.isnan(v)]
                row.append(f"{(np.mean(vals) if vals else float('nan')):.4f}")
            w.writerow(row)
    print(f"保存: {pc_path}")

    # ④ 図（--plot-metric）
    _plot(results, labels, order, worst_sets, args.worst_fracs, pm, args.out_dir)
    print(f"\n完了。結果フォルダ: {args.out_dir}")


def _plot(results, labels, order, worst_sets, worst_fracs, pm, out_dir):
    n = len(order)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(n)
    for l in labels:
        axes[0].plot(
            x,
            [_safe(results[l][fn]["mean"].get(pm)) for fn in order],
            marker=".",
            ms=4,
            label=l,
        )
    axes[0].set_title(f"Per-sample {pm} (sorted by baseline Dice, worst→best)")
    axes[0].set_xlabel("sample rank (0 = baseline worst)")
    axes[0].set_ylabel(pm)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    groups = ["overall"] + [f"worst{int(fr * 100)}%" for fr in worst_fracs]
    width = 0.8 / len(labels)
    all_vals = {}
    for j, l in enumerate(labels):
        vals = []
        for g in groups:
            v = _vals(results, l, worst_sets[g], pm)
            vals.append(v.mean() if len(v) else np.nan)
        all_vals[l] = vals
        bars = axes[1].bar(np.arange(len(groups)) + j * width, vals, width, label=l)
        for rect, v in zip(bars, vals):
            if not np.isnan(v):
                axes[1].text(
                    rect.get_x() + rect.get_width() / 2,
                    v,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
    flat = [v for vs in all_vals.values() for v in vs if not np.isnan(v)]
    if flat:
        lo, hi = min(flat), max(flat)
        pad = (hi - lo) * 0.15 + 1e-4
        axes[1].set_ylim(lo - pad, hi + pad * 3)
    axes[1].set_xticks(np.arange(len(groups)) + width * (len(labels) - 1) / 2)
    axes[1].set_xticklabels(groups)
    axes[1].set_title(f"{pm}: overall vs baseline-worst groups (y zoomed)")
    axes[1].set_ylabel(pm)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Model comparison — does RF expansion help worst cases? ({pm})", fontsize=13
    )
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {fig_path}")


def _safe(v):
    return float("nan") if v is None else float(v)


if __name__ == "__main__":
    main()
