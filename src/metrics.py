# coding:utf-8
"""
セグメンテーション評価指標の共有モジュール（per-sample × per-class）。

これまで評価は smoothing 付き Dice 一本だった（evaluate.py / compare_models.py /
visualize_predictions.sample_dice）。smoothing(+1e-5) は「GT も予測も空」のクラスを
~1.0 と数えてしまい、スコアを甘く見せる。worst（悪いデータ）解析では致命的なので、
ここでは smoothing を使わず、測れないケースは NaN（＝平均から除外）にする。

含む指標
  重なり系（counts から算出, 単位なし 0–1）:
    dice        … 2TP / (2TP+FP+FN)
    iou         … TP / (TP+FP+FN)             （Jaccard）
    precision   … TP / (TP+FP)                （PPV; 予測が空なら NaN）
    recall      … TP / (TP+FN)                （= sensitivity; GT が空なら NaN）
    specificity … TN / (TN+FP)
  境界系（surface distance, scipy EDT, 単位=px or spacing）:
    hd95        … 95 パーセンタイル対称 Hausdorff
    hd          … 最大対称 Hausdorff
    assd        … 平均対称表面距離
    nsd@tau     … Normalized Surface Dice（許容 tau 内に入る表面点の割合）

空クラスの規約（per-class, 1 サンプル）
    GT 空 & 予測 空 : 全指標 NaN（測れない → 平均から除外）
    GT 空 & 予測有  : 偽陽性。dice/iou=0, precision=0, recall=NaN,
                      境界系=ペナルティ（hd*=対角長, assd=対角長, nsd=0）
    GT 有 & 予測 空 : 見逃し。dice/iou=0, recall=0, precision=NaN,
                      境界系=ペナルティ
    GT 有 & 予測有  : 通常計算

注意: 重なり系の dataset 集計は2通り（per-image 平均 / dataset プール）あるので、
呼び出し側が選べるよう counts（tp/fp/fn/tn）も返す。境界系は per-image のみ。
"""

import numpy as np

try:
    from scipy.ndimage import (
        binary_erosion,
        distance_transform_edt,
        generate_binary_structure,
    )

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

# 背景(0)を除く骨 12 部位の名前（legacy/evaluate_metrics.py と同じ並び）
CLASS_NAMES = [
    "0: Background",
    "1: Head",
    "2: Cervical Sp.",
    "3: Thoracic Sp.",
    "4: Lumbar Sp.",
    "5: Ribs/Thorax",
    "6: Scapula/Clavicle",
    "7: Pelvis",
    "8: R-Arm",
    "9: L-Arm",
    "10: R-Leg",
    "11: L-Leg",
    "12: Sternum",
]

OVERLAP_METRICS = ("dice", "iou", "precision", "recall", "specificity")
BOUNDARY_METRICS = ("hd95", "hd", "assd")  # nsd は tau 付きで動的に足す


# ─────────────────────────────────────────────
# counts と重なり系
# ─────────────────────────────────────────────
def binary_counts(pred_c, gt_c):
    """1 クラスの二値マスクから (tp, fp, fn, tn) を返す。"""
    pred_c = pred_c.astype(bool)
    gt_c = gt_c.astype(bool)
    tp = int(np.count_nonzero(pred_c & gt_c))
    fp = int(np.count_nonzero(pred_c & ~gt_c))
    fn = int(np.count_nonzero(~pred_c & gt_c))
    tn = int(pred_c.size - tp - fp - fn)
    return tp, fp, fn, tn


def overlap_from_counts(tp, fp, fn, tn):
    """counts から重なり系指標を算出。測れない（分母 0）ものは NaN。"""
    nan = float("nan")
    gt_pos = tp + fn
    pred_pos = tp + fp

    if gt_pos == 0 and pred_pos == 0:
        # GT も予測も空 → 何も測れない
        return {m: nan for m in OVERLAP_METRICS}

    dice = (2.0 * tp) / (pred_pos + gt_pos) if (pred_pos + gt_pos) > 0 else nan
    union = tp + fp + fn
    iou = tp / union if union > 0 else nan
    precision = tp / pred_pos if pred_pos > 0 else nan  # 予測空なら測れない
    recall = tp / gt_pos if gt_pos > 0 else nan  # GT 空なら測れない
    spec_den = tn + fp
    specificity = tn / spec_den if spec_den > 0 else nan
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


# ─────────────────────────────────────────────
# 境界系（surface distance）
# ─────────────────────────────────────────────
def _surface(mask):
    """二値マスクの境界画素（mask かつ 4 近傍に背景を持つ画素）を返す。"""
    conn = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=conn, border_value=0)
    return mask & ~eroded


def _directed_surface_distances(surf_a, surf_b, spacing):
    """surf_a の各境界点から surf_b までの最短距離（spacing 単位）の配列を返す。"""
    if not surf_b.any():
        return np.array([])
    # distance_transform_edt は「最も近い背景(0)までの距離」を返すので ~surf_b を渡す
    dt = distance_transform_edt(~surf_b, sampling=spacing)
    return dt[surf_a]


def surface_distance_metrics(
    pred_c, gt_c, spacing=(1.0, 1.0), nsd_taus=(1.0, 2.0, 3.0)
):
    """1 クラスの境界系指標（hd95/hd/assd/nsd@tau）を dict で返す。

    片側だけ空のときは対角長をペナルティにする（hd*=diag, assd=diag, nsd=0）。
    両側空のときは全 NaN（測れない）。
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("境界系指標には scipy が必要です。")

    pred_c = pred_c.astype(bool)
    gt_c = gt_c.astype(bool)
    out = {"hd95": float("nan"), "hd": float("nan"), "assd": float("nan")}
    for t in nsd_taus:
        out[f"nsd@{_fmt_tau(t)}"] = float("nan")

    p_any, g_any = pred_c.any(), gt_c.any()
    if not p_any and not g_any:
        return out  # 両側空 → 測れない

    # 画像対角長（spacing 考慮）をペナルティに使う
    diag = float(np.sqrt(sum((s * d) ** 2 for s, d in zip(spacing, pred_c.shape))))
    if not p_any or not g_any:
        out["hd95"] = diag
        out["hd"] = diag
        out["assd"] = diag
        for t in nsd_taus:
            out[f"nsd@{_fmt_tau(t)}"] = 0.0
        return out

    surf_p = _surface(pred_c)
    surf_g = _surface(gt_c)
    # 面積1pxなど erosion で表面が消える退行ケースは元マスクを表面とみなす
    if not surf_p.any():
        surf_p = pred_c
    if not surf_g.any():
        surf_g = gt_c

    d_pg = _directed_surface_distances(surf_p, surf_g, spacing)  # pred→gt
    d_gp = _directed_surface_distances(surf_g, surf_p, spacing)  # gt→pred
    alld = np.concatenate([d_pg, d_gp])

    out["assd"] = float(alld.mean())
    out["hd"] = float(max(d_pg.max(), d_gp.max()))
    out["hd95"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    n = len(alld)
    for t in nsd_taus:
        out[f"nsd@{_fmt_tau(t)}"] = float(np.count_nonzero(alld <= t) / n)
    return out


def _fmt_tau(t):
    return f"{t:g}"


# ─────────────────────────────────────────────
# 1 サンプルまとめ
# ─────────────────────────────────────────────
def sample_metrics(
    pred,
    gt,
    num_classes=13,
    spacing=(1.0, 1.0),
    nsd_taus=(1.0, 2.0, 3.0),
    boundary=True,
):
    """1 サンプル（pred, gt は (H,W) のクラス index 配列）の per-class 指標を返す。

    返り値: dict
      "per_class": {metric_name: np.ndarray(shape=[num_classes-1])}  背景を除く 1..C-1
      "counts":    {"tp"/"fp"/"fn"/"tn": np.ndarray(shape=[num_classes-1], int)}
      "mean":      {metric_name: nanmean over present classes}
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    metric_names = list(OVERLAP_METRICS)
    if boundary:
        metric_names += list(BOUNDARY_METRICS) + [
            f"nsd@{_fmt_tau(t)}" for t in nsd_taus
        ]

    C = num_classes - 1  # 背景除く
    per_class = {m: np.full(C, np.nan, dtype=float) for m in metric_names}
    counts = {k: np.zeros(C, dtype=np.int64) for k in ("tp", "fp", "fn", "tn")}

    for idx, c in enumerate(range(1, num_classes)):
        pred_c = pred == c
        gt_c = gt == c
        tp, fp, fn, tn = binary_counts(pred_c, gt_c)
        counts["tp"][idx], counts["fp"][idx] = tp, fp
        counts["fn"][idx], counts["tn"][idx] = fn, tn

        ov = overlap_from_counts(tp, fp, fn, tn)
        for m in OVERLAP_METRICS:
            per_class[m][idx] = ov[m]

        if boundary:
            sd = surface_distance_metrics(pred_c, gt_c, spacing, nsd_taus)
            for m in sd:
                per_class[m][idx] = sd[m]

    mean = {
        m: (
            float(np.nanmean(per_class[m]))
            if np.any(~np.isnan(per_class[m]))
            else float("nan")
        )
        for m in metric_names
    }
    return {"per_class": per_class, "counts": counts, "mean": mean}


def metric_list(nsd_taus=(1.0, 2.0, 3.0), boundary=True):
    """この設定で出力される指標名の並びを返す（表ヘッダ用）。"""
    names = list(OVERLAP_METRICS)
    if boundary:
        names += list(BOUNDARY_METRICS) + [f"nsd@{_fmt_tau(t)}" for t in nsd_taus]
    return names


# 境界系は「小さいほど良い」、重なり系は「大きいほど良い」
LOWER_IS_BETTER = set(BOUNDARY_METRICS)


def higher_is_better(metric):
    base = metric.split("@")[0]
    return base not in LOWER_IS_BETTER
