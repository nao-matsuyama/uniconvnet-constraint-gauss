import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset_scinti import ScintiBoneDataset
from model_uniconvnet_unet import UniConvNet_UNet

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


def calculate_metrics(pred, target, num_classes=13):
    batch_metrics = {}
    for c in range(num_classes):
        p_mask = pred == c
        t_mask = target == c
        intersection = (p_mask & t_mask).sum().item()
        union = p_mask.sum().item() + t_mask.sum().item()
        iou_denominator = (p_mask | t_mask).sum().item()
        batch_metrics[c] = {
            "intersection": intersection,
            "union": union,
            "iou_denom": iou_denominator,
            "correct": intersection,
            "total": t_mask.sum().item(),
        }
    return batch_metrics


def evaluate_model():
    DATA_DIR = "/workspace/scinti_segmentation"
    NUM_CLASSES = 13
    BATCH_SIZE = 32

    # 自動的に一番新しい実験フォルダを探し出す
    exp_folders = sorted(glob.glob("/workspace/experiments/*_*/"))
    if not exp_folders:
        raise FileNotFoundError(
            "❌ 実験フォルダ（/workspace/experiments/）が見つかりません。先に学習を行ってください。"
        )

    LATEST_EXP_DIR = exp_folders[-1]  # 最新のフォルダを取得
    MODEL_PATH = os.path.join(LATEST_EXP_DIR, "best_model.pth")
    REPORT_SAVE_PATH = os.path.join(
        LATEST_EXP_DIR, "evaluation_report.txt"
    )  # 評価結果の保存先

    print(f"📂 評価対象の実験フォルダ: {LATEST_EXP_DIR}")
    print(f"📦 読み込むモデルの重み: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. データセットの準備
    dataset = ScintiBoneDataset(data_dir=DATA_DIR)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
    )

    # 2. モデルの準備とロード
    model = UniConvNet_UNet()
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d) and module.out_channels == 1:
            in_ch = module.in_channels
            model.final_conv = nn.Conv2d(
                in_ch,
                NUM_CLASSES,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
            )
            break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    total_counts = {
        c: {"intersection": 0, "union": 0, "iou_denom": 0, "correct": 0, "total": 0}
        for c in range(NUM_CLASSES)
    }

    # 3. 評価ループ
    print("🧪 未知の検証データに対する定量評価を開始します...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            batch_res = calculate_metrics(preds, masks, NUM_CLASSES)
            for c in range(NUM_CLASSES):
                for key in total_counts[c].keys():
                    total_counts[c][key] += batch_res[c][key]

    # 4. レポートの作成（ここから綺麗に整理しました✨）
    dice_list, iou_list, acc_list = [], [], []

    report_text = (
        "======================================================================\n"
    )
    report_text += f"📊 定量評価レポート (Target Folder: {os.path.basename(LATEST_EXP_DIR.strip('/'))})\n"
    report_text += (
        "======================================================================\n"
    )
    report_text += f"{'部 位 (Class)':<25} | {'Dice スコア':<12} | {'IoU スコア':<12} | {'Accuracy':<12}\n"
    report_text += (
        "----------------------------------------------------------------------\n"
    )

    # 各クラスのスコアを計算して一行ずつ追加
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        intersection = total_counts[c]["intersection"]
        union = total_counts[c]["union"]
        iou_denom = total_counts[c]["iou_denom"]
        correct = total_counts[c]["correct"]
        total = total_counts[c]["total"]

        dice = (2.0 * intersection) / union if union > 0 else 1.0
        iou = intersection / iou_denom if iou_denom > 0 else 1.0
        accuracy = correct / total if total > 0 else 1.0

        if c != 0:  # 背景(0)以外の数値を平均用にストック
            dice_list.append(dice)
            iou_list.append(iou)
            acc_list.append(accuracy)

        # 🟢 正しい並び順のフォーマット（左寄せ12文字の枠の中に、小数点4桁で表示）
        report_text += (
            f"{name:<25} | {dice:<12.4f} | {iou:<12.4f} | {accuracy:<12.4f}\n"
        )

    # ループが終わった後に、全体の平均（Mean）をドカンと一行追加
    report_text += (
        "======================================================================\n"
    )
    report_text += f"{'🔥 全部位の平均 (Mean) ※背景除外':<25} | {np.mean(dice_list):<12.4f} | {np.mean(iou_list):<12.4f} | {np.mean(acc_list):<12.4f}\n"
    report_text += (
        "======================================================================\n"
    )

    # 🌟 画面に結果を表示
    print(report_text)

    # 🌟 同じ実験フォルダ内にテキストとして保存！
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📝 評価レポートを実験フォルダ内に保存しました ➔ {REPORT_SAVE_PATH}\n")


if __name__ == "__main__":
    evaluate_model()
