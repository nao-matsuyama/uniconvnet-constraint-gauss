import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_scinti import ScintiBoneDataset
from model_uniconvnet_unet import UniConvNet_UNet


def evaluate_model():
    # 💡 評価は1基のGPU、またはCPUで十分動きます
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # パス設定（最新の実験結果のフォルダ、モデル名に変えてください）
    # ※ [ここに最新の実験フォルダ名を入れる]
    EXP_DIR = "/workspace/experiments/20260615_1612_PlanA_3Class_CE_plus_DiceLoss"
    MODEL_PATH = os.path.join(EXP_DIR, "best_model.pth")
    DATA_DIR = "/workspace/scinti_segmentation"
    NUM_CLASSES = 3

    # 1. データセットとローダーの準備 (評価なのでシャッフルは無し)
    dataset = ScintiBoneDataset(data_dir=DATA_DIR)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # 2. モデルの定義と重みのロード
    model = UniConvNet_UNet()

    # 最終層のクラス数を学習時と同じ3クラスに変更
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

    # 保存された重みを読み込む
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # 💡 DDP(マルチGPU)で保存したモデルは頭に「module.」がついているため、それを消す処理
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    print("🚀 評価を開始します...")

    total_dice_bone = 0.0
    total_dice_tumor = 0.0
    count = 0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.softmax(outputs, dim=1)
            pred_masks = torch.argmax(
                preds, dim=1
            )  # 一番確率の高いクラス(0, 1, 2)を取得

            # クラスごとのDiceスコアを計算 (1: 骨, 2: 病変)
            for c, label in [(1, "Bone"), (2, "Tumor")]:
                p = (pred_masks == c).float().reshape(-1)
                t = (masks == c).float().reshape(-1)
                intersection = (p * t).sum()
                union = p.sum() + t.sum()

                dice = (2.0 * intersection) / (union + 1e-5)
                if c == 1:
                    total_dice_bone += dice.item()
                if c == 2:
                    total_dice_tumor += dice.item()

            count += 1

    print("\n================ 評価結果 ================")
    print(f"📊 評価症例数: {count}")
    print(f"🦴 正常骨の平均 Dice スコア: {total_dice_bone / count:.4f}")
    print(f"🔥 病変領域の平均 Dice スコア: {total_dice_tumor / count:.4f}")
    print("==========================================")


if __name__ == "__main__":
    evaluate_model()
