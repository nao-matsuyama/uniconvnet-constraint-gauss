# coding:utf-8
"""
list_runs — experiments/ 配下の学習 run を一覧表にする (実験管理用)。

各 run フォルダの run_config.json(条件) と run_info.txt(結果) を読み、
  folder / date / dw_mode / 主要config / best_bone_dice / epochs / ckpt有無
を1行にまとめて表示する。手作業の台帳管理を不要にする。

使い方:
  python3 src/list_runs.py                       # experiments/ を dice 降順
  python3 src/list_runs.py --root experiments --sort date
  python3 src/list_runs.py --filter spectral     # dw_mode に spectral を含む run のみ
  python3 src/list_runs.py --csv runs.csv        # CSV 出力も
"""

import argparse
import csv
import glob
import json
import os
import re


def parse_run(d):
    args, ts, commit = {}, "", ""
    cfg = os.path.join(d, "run_config.json")
    if os.path.exists(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                c = json.load(f)
            args = c.get("args", {})
            ts = c.get("timestamp", "")
            commit = c.get("git_commit", "")
        except Exception:
            pass

    best, epochs, stopped = None, None, None
    info = os.path.join(d, "run_info.txt")
    if os.path.exists(info):
        with open(info, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        m = re.search(r"best_bone_dice\s*:\s*([\d.]+)", txt)
        best = float(m.group(1)) if m else None
        m = re.search(r"epochs_run\s*:\s*(\d+)", txt)
        epochs = int(m.group(1)) if m else None
        m = re.search(r"early_stopped\s*:\s*(\w+)", txt)
        stopped = m.group(1) if m else None

    has_ckpt = os.path.exists(os.path.join(d, "best_uniconvnet_unet.pth"))
    return args, ts, commit, best, epochs, stopped, has_ckpt


def mode_str(args):
    dw = args.get("dw_mode", "dense")
    if dw == "dense":
        if args.get("spectral_dw"):
            return "spec_gauss*"  # legacy SpectralGaussianDW
        if args.get("adaptive_dw"):
            return "adaptive*"  # legacy ContentAdaptiveDW
    return dw


def cfg_str(args):
    dw = mode_str(args)
    if dw.startswith("spec"):
        p = [f"σ={args.get('spectral_init_sigma')}"]
        slr = args.get("spectral_sigma_lr")
        if slr:
            p.append(f"σlr={slr}")
        p.append(f"γ={args.get('spectral_init_gamma')}")
        p.append(f"local={args.get('spectral_use_local_branch')}")
        p.append(f"α={args.get('spectral_alpha')}")
        return " ".join(str(x) for x in p)
    if dw.startswith("adaptive"):
        return f"dil={args.get('adaptive_dilations')}"
    if dw == "separable":
        return "rank1-SVD"
    # dense / dilated
    erf = args.get("erf_reg_weight")
    if erf:
        return f"erf λ={erf} target={args.get('erf_target_spread')}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--sort", choices=["date", "dice", "name"], default="dice")
    ap.add_argument(
        "--filter", default=None, help="dw_mode/config にこの文字列を含む run のみ"
    )
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    dirs = [d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)]
    rows = []
    for d in dirs:
        a, ts, commit, best, epochs, stopped, ckpt = parse_run(d)
        if not a and best is None:
            continue  # run らしくない
        rows.append(
            {
                "folder": os.path.basename(d),
                "date": ts[:16],
                "mode": mode_str(a),
                "config": cfg_str(a),
                "best_dice": best,
                "epochs": epochs,
                "stopped": stopped,
                "ckpt": "Y" if ckpt else "-",
                "commit": commit,
                "path": os.path.join(d, "best_uniconvnet_unet.pth"),
            }
        )

    if args.filter:
        f = args.filter.lower()
        rows = [r for r in rows if f in r["mode"].lower() or f in r["config"].lower()]

    if args.sort == "dice":
        rows.sort(key=lambda r: (r["best_dice"] is None, -(r["best_dice"] or 0)))
    elif args.sort == "date":
        rows.sort(key=lambda r: r["date"])
    else:
        rows.sort(key=lambda r: r["folder"])

    print(
        f"\n{'folder':<34}{'date':<17}{'mode':<13}{'best':>7}{'ep':>4}{'ck':>3}  config"
    )
    print("-" * 120)
    for r in rows:
        bd = f"{r['best_dice']:.4f}" if r["best_dice"] is not None else "  -  "
        ep = str(r["epochs"]) if r["epochs"] is not None else "-"
        print(
            f"{r['folder']:<34}{r['date']:<17}{r['mode']:<13}{bd:>7}{ep:>4}"
            f"{r['ckpt']:>3}  {r['config']}"
        )
    print("-" * 120)
    print(
        f"{len(rows)} runs  (* = legacy flag経由)  ckpt Y = best_uniconvnet_unet.pth あり"
    )
    print(
        "凡例: spectral config = σ(init) σlr(専用lr) γ(init_gamma) local(use_local_branch) α(alpha)"
    )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=list(rows[0].keys()) if rows else ["folder"]
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV 保存: {args.csv}")


if __name__ == "__main__":
    main()
