#!/usr/bin/env bash
# ==============================================================
# notify_run.sh — 任意コマンドを実行し、完了時に Discord/Slack へ通知する汎用ラッパ。
#
# これを通して実験を起動すれば、**どの実験スクリプトでも(将来追加した新規ファイルでも)**
# スクリプト側を一切いじらずに完了通知が届く。成否・終了コード・所要時間・GPU・ログ末尾・
# 最新の run フォルダを本文に載せる。
#
# 使い方(コンテナ内 or ホスト、どちらでも):
#   scripts/notify_run.sh python3 src/train.py --dw-mode spectral_mix --tag mix_v1
#   scripts/notify_run.sh make train-inner
#
# 事前準備(1回だけ):
#   echo 'https://discord.com/api/webhooks/XXXX/YYYY' > scripts/.notify_webhook
#   (または export NOTIFY_WEBHOOK_URL=... 。scripts/.notify_webhook は .gitignore 済み)
#
# 終了コードは実行したコマンドのものをそのまま返す(パイプライン連結でも壊れない)。
# ==============================================================
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: notify_run.sh <command> [args...]" >&2
    exit 2
fi

# リポジトリルート(このスクリプトの親の親)を基準にする。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CMD_STR="$*"
HOST="$(hostname 2>/dev/null || echo unknown)"
GPU="${CUDA_VISIBLE_DEVICES:-unset}"
LOG="$(mktemp 2>/dev/null || echo /tmp/notify_run.$$.log)"
START=$SECONDS

# 子ラッパ由来の二重通知を防ぐフラグ(train.py 内蔵通知はこれを見て自分は送らない)。
export NOTIFY_WRAPPED=1

# 実行(標準出力/エラーを画面とログの両方へ)。PIPESTATUS で本体の終了コードを取得。
"$@" 2>&1 | tee "$LOG"
CODE=${PIPESTATUS[0]}

DUR=$((SECONDS - START))
DUR_STR="$((DUR / 3600))h$(((DUR % 3600) / 60))m$((DUR % 60))s"

# 最新の run フォルダ(あれば)。実験成果の場所を通知に添える。
RUN_DIR="$(ls -dt "$REPO_ROOT"/experiments/run_* 2>/dev/null | head -1 || true)"
RUN_LINE=""
[ -n "$RUN_DIR" ] && RUN_LINE="run: $(basename "$RUN_DIR")"

# ログ末尾(最後の15行)を code block で添付。
TAIL="$(tail -n 15 "$LOG" 2>/dev/null)"

if [ "$CODE" -eq 0 ]; then
    STATUS=ok
    TITLE="実験完了 (exit 0)"
else
    STATUS=fail
    TITLE="実験失敗 (exit $CODE)"
fi

BODY="$(printf 'cmd: %s\nhost: %s | GPU: %s | 所要: %s\n%s\n\n```\n%s\n```' \
    "$CMD_STR" "$HOST" "$GPU" "$DUR_STR" "$RUN_LINE" "$TAIL")"

python3 "$SCRIPT_DIR/notify.py" --status "$STATUS" --title "$TITLE" "$BODY" || true

rm -f "$LOG" 2>/dev/null || true
exit "$CODE"
