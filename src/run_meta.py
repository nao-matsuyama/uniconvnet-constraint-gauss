# coding:utf-8
"""
解析・可視化の出力フォルダに「何の条件で作った結果か」を記録する共通ヘルパー。

各ツール(erf_sigma_table / compare_models / visualize_* )が out-dir に
  manifest.json … 機械可読 (tool, 日時, git commit, 全パラメータ, モデル一覧)
  manifest.txt  … 人間可読 (同内容を整形)
を残す。これでフォルダ単体で「日付・パラメータ・対象モデル」が分かる。
"""

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def _now():
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo("Asia/Tokyo"))
    except Exception:
        pass
    return datetime.now(timezone(timedelta(hours=9)))


def _git_commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(out_dir, tool, params, models=None):
    """
    Args:
        out_dir : 出力フォルダ
        tool    : ツール名 (例 "erf_sigma_table")
        params  : dict — このツールの全パラメータ
        models  : dict {label: weights_path} or None — 対象モデル一覧
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = _now().strftime("%Y-%m-%d %H:%M:%S %Z")

    meta = {
        "tool": tool,
        "timestamp": ts,
        "git_commit": _git_commit(),
        "out_dir": out_dir,
        "params": params,
        "models": models or {},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    lines = [
        "=" * 60,
        " 解析マニフェスト (この結果がどの条件で作られたか)",
        "=" * 60,
        f" tool       : {tool}",
        f" date       : {ts}",
        f" git commit : {_git_commit()}",
        f" out_dir    : {out_dir}",
        "",
        "[ パラメータ ]",
    ]
    for k, v in params.items():
        lines.append(f"  {k:<16}: {v}")
    if models:
        lines += ["", "[ 対象モデル ]"]
        for label, path in models.items():
            lines.append(f"  {label:<12}: {path}")
    lines += ["=" * 60, ""]

    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"📝 manifest を保存: {os.path.join(out_dir, 'manifest.txt')}")
