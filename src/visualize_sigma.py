# coding:utf-8
"""
visualize_sigma — SpectralDW の学習可能 σ が「ちゃんとフィットしたか」を可視化する。

σ は per-channel の学習パラメータ (log_sigma, 72 モジュール分) だが、lr が小さいと
init 値で凍結して「学習可能 RF」が名ばかりになる。本スクリプトはチェックポイントの
log_sigma を直接読み、stage ごとに:
  * 学習後の実効 σ (=clamp(exp(log_sigma), .., max_sigma)) の分布
  * init σ からの log 空間移動量 |Δlog σ| (どれだけ動いたか = フィットの健全性)
を図 + コンソール表で出す。複数チェックポイントを並べて比較できる
(例: init-frozen 版 vs σ高lr 版)。

使い方:
  python3 src/visualize_sigma.py --weights <spec.pth> [<spec2.pth> ...] \
      --labels frozen high_lr --out-dir /workspace/sigma_results

読み方:
  * |Δlog σ| の中央値が ~0 (< 0.05) → σ 凍結 (動いていない = フィット失敗)。
  * σ を 2倍にするには Δlog=0.69 必要。0.69 の参照線を超える帯があれば大きく動いた証拠。
  * σ 分布が init 一点に集中 (std~0) → per-channel 分化なし。広がれば分化して学習。
"""

import argparse
import collections
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

LOG_2X = 0.6931  # σ を 2 倍にするのに必要な |Δlog|
NOISE = 0.05  # これ未満の |Δlog| は実質「動いていない」ノイズ帯


def _to4(v):
    if not isinstance(v, (list, tuple)):
        return [float(v)] * 4
    v = [float(x) for x in v]
    return (v + [v[-1]] * 4)[:4]


def _read_cfg(weights, key, default):
    """checkpoint と同じフォルダの run_config.json から学習時引数を読む。"""
    cfg = os.path.join(os.path.dirname(os.path.abspath(weights)), "run_config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                args = json.load(f).get("args", {})
            if key in args and args[key] is not None:
                return args[key]
        except Exception:
            pass
    return default


def collect_sigma(weights):
    """checkpoint から stage 別の σ / |Δlog| を集める。

    Returns: dict{stage: {sigma, dlog, init, max, n_mod}}, 総 log_sigma モジュール数
    """
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    init = _to4(_read_cfg(weights, "spectral_init_sigma", [1.0]))
    mx = _to4(_read_cfg(weights, "spectral_max_sigma", [32.0, 24.0, 12.0, 6.0]))

    keys = [k for k in sd if k.endswith("log_sigma")]
    acc = collections.defaultdict(lambda: {"sigma": [], "dlog": [], "n": 0})
    for k in keys:
        m = re.search(r"stages\.(\d+)\.", k)
        st = int(m.group(1)) if m else 0
        raw = sd[k].float()
        sigma = torch.exp(raw).clamp(1e-3, mx[st]).numpy()
        dlog = (raw - float(np.log(max(init[st], 1e-6)))).abs().numpy()
        acc[st]["sigma"].append(sigma)
        acc[st]["dlog"].append(dlog)
        acc[st]["n"] += 1

    out = {}
    for st, d in acc.items():
        out[st] = {
            "sigma": np.concatenate(d["sigma"]),
            "dlog": np.concatenate(d["dlog"]),
            "init": init[st],
            "max": mx[st],
            "n_mod": d["n"],
        }
    return out, len(keys)


def print_table(label, data, n_keys):
    print(f"\n[{label}] log_sigma モジュール数: {n_keys}")
    print(
        f"  {'stage':<6}{'init':>6}{'sigma mean':>12}{'std':>8}{'min':>7}{'max':>7}"
        f"{'|dlog|med':>11}{'moved%':>9}{'verdict':>10}"
    )
    for st in sorted(data):
        d = data[st]
        s, dl = d["sigma"], d["dlog"]
        moved = float((dl > NOISE).mean()) * 100
        med = float(np.median(dl))
        verdict = "FROZEN" if med < NOISE else "moved"
        print(
            f"  {st:<6}{d['init']:>6.1f}{s.mean():>12.3f}{s.std():>8.3f}"
            f"{s.min():>7.2f}{s.max():>7.2f}{med:>11.4f}{moved:>8.0f}%{verdict:>10}"
        )


def _boxes(ax, series, stages, labels, colors, ylabel, title):
    """stage を x 軸, label ごとにオフセットして箱ひげを並べる。"""
    L = len(labels)
    width = 0.8 / max(L, 1)
    for li, lab in enumerate(labels):
        data = [series[li].get(st, np.array([np.nan])) for st in stages]
        pos = [i + (li - (L - 1) / 2.0) * width for i in range(len(stages))]
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for box in bp["boxes"]:
            box.set(facecolor=colors[li], alpha=0.6)
        for med in bp["medians"]:
            med.set(color="black", lw=1.2)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([f"stage{st}" for st in stages])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out-dir", default="/workspace/sigma_results")
    args = ap.parse_args()

    if args.labels and len(args.labels) == len(args.weights):
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.dirname(w)) for w in args.weights]

    per_model = []
    for lab, w in zip(labels, args.weights):
        data, n = collect_sigma(w)
        if n == 0:
            print(
                f"[{lab}] log_sigma を持たない (spectral でない) チェックポイント → スキップ"
            )
            per_model.append((lab, {}))
            continue
        print_table(lab, data, n)
        per_model.append((lab, data))

    valid = [(l, d) for l, d in per_model if d]
    if not valid:
        print("\nσ を持つチェックポイントがありません。終了。")
        return

    labels = [l for l, _ in valid]
    stages = sorted({st for _, d in valid for st in d})
    sigma_series = [{st: d[st]["sigma"] for st in d} for _, d in valid]
    dlog_series = [{st: d[st]["dlog"] for st in d} for _, d in valid]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))

    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    _boxes(
        axes[0],
        sigma_series,
        stages,
        labels,
        colors,
        "learned sigma (feature px)",
        "Learned sigma per stage (init = X)",
    )
    # init σ を各 stage に X 印で重ねる (最初のモデルの init を使用)
    d0 = valid[0][1]
    for i, st in enumerate(stages):
        axes[0].scatter(
            [i],
            [d0[st]["init"]],
            marker="x",
            color="red",
            zorder=5,
            s=60,
            label="init" if i == 0 else None,
        )
        axes[0].hlines(d0[st]["max"], i - 0.4, i + 0.4, color="gray", ls=":", lw=1)

    _boxes(
        axes[1],
        dlog_series,
        stages,
        labels,
        colors,
        "|Δlog sigma| from init",
        "How far sigma moved from init",
    )
    axes[1].axhline(LOG_2X, color="green", ls="--", lw=1, label="Δlog=0.69 (2x sigma)")
    axes[1].axhline(NOISE, color="red", ls=":", lw=1, label="0.05 (frozen floor)")
    axes[1].legend(fontsize=8)

    # 凡例 (label の色)
    handles = [
        plt.Line2D([0], [0], marker="s", ls="", color=colors[i], alpha=0.6, label=l)
        for i, l in enumerate(labels)
    ]
    axes[0].legend(
        handles=handles
        + [plt.Line2D([0], [0], marker="x", ls="", color="red", label="init")],
        fontsize=8,
    )

    plt.suptitle("SpectralDW sigma fitting check", fontsize=13)
    plt.tight_layout()
    path = os.path.join(args.out_dir, "sigma_fit.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n保存: {path}")
    print(
        "読み方: |Δlog| の箱が赤点線(0.05)付近 = σ凍結 / 緑破線(0.69)超え = σが2倍以上動いた。"
    )


if __name__ == "__main__":
    main()
