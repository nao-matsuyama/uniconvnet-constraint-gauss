# coding:utf-8
"""
per_sample_dice.csv から「全モデルペア × 全 worst 群」の Wilcoxon 符号順位検定を計算。

compare_models.py は「各モデル vs baseline」しか検定しないため(3モデルなら
baseline-erf0.01, baseline-erf0.1 の 2 ペア × 2 群 = 4 通り)、モデル間の総当り
(erf0.01 vs erf0.1) を補完し、全 3ペア × 2群 = 6 通りを揃える。

帰無仮説 H0: ペア差 Δ = (モデルB の Dice − モデルA の Dice) の母中央値 = 0
             (= その worst 群で A と B に系統的な差がない)。両側検定。

worst 群は「ソート基準モデル(既定=先頭列)の Dice が低い上位 k%」で固定し、
同一サンプル集合上で全ペアを比較する(対応のある比較)。

使い方:
  python3 scripts/pairwise_wilcoxon.py <per_sample_dice.csv>
  python3 scripts/pairwise_wilcoxon.py <csv> --fracs 0.1 0.25 --sort-by baseline
CSV 形式(compare_models 出力): rank,fname,<model1>,<model2>,...
"""

import argparse
import csv
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="compare_models の per_sample_dice.csv")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.25])
    ap.add_argument("--sort-by", default=None,
                    help="worst 群の定義に使うモデル列 (既定=先頭のモデル列)")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    cols = [c for c in rows[0].keys() if c not in ("rank", "fname")]
    n = len(rows)
    key = args.sort_by or cols[0]
    rows.sort(key=lambda r: float(r[key]))   # worst(低Dice)を先頭へ

    print(f"CSV: {args.csv}")
    print(f"n={n}  models={cols}  worst 基準列={key}")
    print("H0: ペア差(B−A)の母中央値=0 / 両側 Wilcoxon 符号順位検定")

    for fr in args.fracs:
        k = max(1, round(n * fr))
        sub = rows[:k]
        print(f"\n===== worst {int(fr*100)}% (n={k}) =====")
        print(f"  {'pair (B vs A)':<26}{'Δmean':>10}{'Δmedian':>10}{'p':>12}  有意")
        print("  " + "-" * 60)
        for a, b in combinations(cols, 2):
            xa = np.array([float(r[a]) for r in sub])
            xb = np.array([float(r[b]) for r in sub])
            d = xb - xa
            if np.allclose(d, 0):
                print(f"  {b+' vs '+a:<26}{'0':>10}{'0':>10}{'n/a':>12}  差0")
                continue
            _, p = wilcoxon(d)  # 両側, H0: median(d)=0
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
            print(f"  {b+' vs '+a:<26}{d.mean():>+10.4f}{np.median(d):>+10.4f}{p:>12.3g}  {sig}")


if __name__ == "__main__":
    main()
