#!/usr/bin/env python3
# coding:utf-8
"""
notify.py — 実験完了を Discord(または Slack)Webhook に通知する軽量ヘルパ。

設計方針:
  * **標準ライブラリのみ**(urllib)。コンテナ内(requests 未導入でも)で動く。
  * **ベストエフォート**: Webhook 未設定・送信失敗でも決して例外で落とさない
    (学習の最後で通知に失敗しても学習結果は無事、が最優先)。
  * **秘密をリポジトリに置かない**: Webhook URL は次の優先順で解決する。
      1. 環境変数 NOTIFY_WEBHOOK_URL
      2. 環境変数 NOTIFY_WEBHOOK_FILE が指すファイル
      3. このファイルと同じ scripts/.notify_webhook(.gitignore 済み)
  * Discord / Slack を URL ドメインで自動判別(payload 形式が違うため)。

使い方(CLI):
    python3 scripts/notify.py "本文"                 # 本文を送る
    python3 scripts/notify.py --status ok --title "run done" "本文"
    echo "本文" | python3 scripts/notify.py --status fail --title "run FAILED"

使い方(import):
    from notify import notify
    notify("本文", title="run done", status="ok")
"""

import argparse
import json
import os
import sys
import urllib.request

# Discord の content 上限は 2000 字。安全側に丸める。
_MAX_LEN = 1900


def resolve_webhook():
    """Webhook URL を環境変数/設定ファイルから解決。無ければ None(=通知しない)。"""
    url = os.environ.get("NOTIFY_WEBHOOK_URL", "").strip()
    if url:
        return url
    path = os.environ.get("NOTIFY_WEBHOOK_FILE", "").strip()
    if not path:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".notify_webhook"
        )
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
        except OSError:
            return None
    return None


def _build_payload(url, text):
    """URL のドメインから Discord / Slack を判別して payload を組む。"""
    if "hooks.slack.com" in url:
        return {"text": text}  # Slack Incoming Webhook
    # 既定は Discord(discord.com / discordapp.com)。
    return {"content": text}


def _emoji(status):
    return {"ok": "✅", "fail": "❌"}.get(status, "🔔")


def notify(message, title=None, status=None, timeout=10):
    """Webhook に message を送る。成功で True、未設定/失敗で False(例外は投げない)。"""
    url = resolve_webhook()
    if not url:
        # 未設定は「通知しない」正常系。CI/ローカルで静かに no-op。
        return False

    head = f"{_emoji(status)} {title}".strip() if title else _emoji(status)
    text = f"{head}\n{message}" if message else head
    if len(text) > _MAX_LEN:
        text = text[: _MAX_LEN - 3] + "..."

    data = json.dumps(_build_payload(url, text)).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Discord は 204、Slack は 200 を返す。2xx を成功とみなす。
            return 200 <= resp.status < 300
    except Exception as e:  # noqa: BLE001  ネット断/URL不正でも学習は落とさない
        print(f"[notify] 送信失敗(無視して続行): {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Discord/Slack Webhook 通知")
    ap.add_argument("message", nargs="?", default=None, help="本文(無ければ stdin)")
    ap.add_argument("--title", default=None, help="見出し")
    ap.add_argument(
        "--status", choices=["ok", "fail"], default=None, help="✅/❌ の絵文字"
    )
    args = ap.parse_args()

    msg = args.message
    if msg is None and not sys.stdin.isatty():
        msg = sys.stdin.read()
    msg = (msg or "").rstrip()

    sent = notify(msg, title=args.title, status=args.status)
    if not sent and not resolve_webhook():
        print(
            "[notify] Webhook 未設定のため送信しませんでした。"
            "scripts/.notify_webhook に URL を書くか NOTIFY_WEBHOOK_URL を設定してください。",
            file=sys.stderr,
        )
    # 通知失敗で呼び出し元(ラッパ)を落とさないため常に 0 で返す。
    sys.exit(0)


if __name__ == "__main__":
    main()
