#!/usr/bin/env bash
# ============================================================
# 一括評価スクリプト (コンテナ内で実行)
#
#   bash scripts/eval_all.sh
#
# 下の MODELS (label=path) を編集するだけで、複数モデルを同じ条件で
# まとめて評価する。結果は /workspace/eval_results/<日時>/ に集約。
#
# 環境変数で上書き可:
#   GPU=1 bash scripts/eval_all.sh        … 使う GPU
#   NSAMPLES=100 bash scripts/eval_all.sh … ERF σ のサンプル数 (速くしたい時)
#   FULL=1 bash scripts/eval_all.sh       … 各モデルの推論可視化/ERF図も生成 (重い)
# ============================================================
set -uo pipefail
cd "$(dirname "$0")/.."   # repo ルートへ

GPU="${GPU:-0}"
DATA="${DATA:-/workspace/scinti_segmentation}"
NSAMPLES="${NSAMPLES:-300}"
FULL="${FULL:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="/workspace/eval_results/${STAMP}"
mkdir -p "$ROOT"

# --- 比較対象モデル (label=path) ---
# 引数で label=path を渡せる。渡さなければ下のデフォルトを使う。
#   bash scripts/eval_all.sh A=/path/best.pth B=/path/best.pth ...
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(
    "baseline=/workspace/experiments/run_20260627_211529_baseline_noerf/best_uniconvnet_unet.pth"
    "erf0.1_old=/workspace/experiments/run_20260627_192001_erf0.1_t3.6-8/best_uniconvnet_unet.pth"
    "erf0.1_fmap=/workspace/experiments/run_20260628_015719_erf0.1_fmap-aware/best_uniconvnet_unet.pth"
  )
fi

LABELS=(); WEIGHTS=()
for m in "${MODELS[@]}"; do LABELS+=("${m%%=*}"); WEIGHTS+=("${m#*=}"); done

export CUDA_VISIBLE_DEVICES="$GPU"
run(){ echo -e "\n\033[1;36m# $*\033[0m"; "$@"; }

echo "================ 一括評価 ================"
echo " 日時   : $STAMP"
echo " GPU    : $GPU"
echo " 出力   : $ROOT"
echo " モデル :"; for m in "${MODELS[@]}"; do echo "   $m"; done
echo "=========================================="

# 1) ERF σ & AGD(kurtosis) 比較
run python3 src/erf_sigma_table.py \
  --weights "${WEIGHTS[@]}" --labels "${LABELS[@]}" \
  --input-size 512 --n-samples "$NSAMPLES" --out-dir "$ROOT/sigma"

# 2) ワースト群 全指標比較 (worst10/25% + Wilcoxon + per-class, 境界系hd95/assd含む)
#    NOBOUNDARY=1 で重なり系のみ(高速, 図はdice)。境界ありなら図はhd95。
if [ "${NOBOUNDARY:-0}" = "1" ]; then CMP_FLAGS="--no-boundary"; else CMP_FLAGS="--plot-metric hd95"; fi
run python3 src/compare_models.py \
  --weights "${WEIGHTS[@]}" --labels "${LABELS[@]}" \
  --data-dir "$DATA" --out-dir "$ROOT/worst_compare" $CMP_FLAGS

# 3) 計算コスト (params/FLOPs/throughput/省コスト比)
run python3 src/benchmark.py \
  --weights "${WEIGHTS[@]}" --labels "${LABELS[@]}" \
  --input-size 256 --batch-size 8 --out-dir "$ROOT/benchmark"

# 4) 各モデルの全指標(dice/iou/precision/recall/specificity+境界hd95/assd/nsd)
#    + worstデシル。--out-dir に report.txt/per_class.csv/per_sample.csv も保存。
#    境界指標が重い時は NOBOUNDARY=1 で重なり系のみに。
EVAL_FLAGS=""; [ "${NOBOUNDARY:-0}" = "1" ] && EVAL_FLAGS="--no-boundary"
for i in "${!LABELS[@]}"; do
  l="${LABELS[$i]}"; w="${WEIGHTS[$i]}"
  run python3 src/evaluate.py --weights "$w" --data-dir "$DATA" \
    --batch-size 16 --num-workers 2 --out-dir "$ROOT/eval_${l}" $EVAL_FLAGS \
    2>&1 | tee "$ROOT/eval_${l}.txt"
  run python3 src/model_stats.py --weights "$w" --input-size 256 \
    2>&1 | tee "$ROOT/stats_${l}.txt"
done

# 5) (FULL=1) 各モデルの推論可視化(全マスク) と ERF 図
if [ "$FULL" = "1" ]; then
  for i in "${!LABELS[@]}"; do
    l="${LABELS[$i]}"; w="${WEIGHTS[$i]}"
    run python3 src/visualize_predictions.py --weights "$w" --data-dir "$DATA" \
      --out-dir "$ROOT/pred_${l}" --save-all
    run python3 src/visualize_erf.py --weights "$w" --part all \
      --input-size 512 --n-samples 100 --out-dir "$ROOT/erf_${l}"
  done
fi

echo -e "\n================ 完了 ================"
echo " 結果: $ROOT"
echo "  sigma/erf_sigma_compare.png      … σ & AGD 比較図"
echo "  worst_compare/summary.txt        … ワースト群 Dice + 有意性"
echo "  worst_compare/per_class_on_worst.csv"
echo "  benchmark/benchmark.csv          … コスト"
echo "  eval_<label>.txt / stats_<label>.txt"
echo " 各フォルダに manifest.txt (条件) 同梱"
echo "======================================"
