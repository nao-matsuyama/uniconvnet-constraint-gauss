# coding:utf-8
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_scinti import ScintiMultiClassDataset
from model_uniconvnet_unet import UniConvNet_UNet_13CH


def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHT_PATH = "experiments/best_uniconvnet_unet.pth"

    dataset = ScintiMultiClassDataset()
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

    model = UniConvNet_UNet_13CH(num_classes=13)
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
        print(f"🎯 重み '{WEIGHT_PATH}' をロードしました。")
    model.to(device).eval()

    # 各クラス(1~12)のDiceを格納する辞書
    class_dices = {c: [] for c in range(1, 13)}

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)

            for c in range(1, 13):
                for b in range(images.size(0)):
                    pred_c = (preds[b] == c).float()
                    mask_c = (masks[b] == c).float()
                    inter = (pred_c * mask_c).sum()
                    uni = pred_c.sum() + mask_c.sum()
                    dice = (2.0 * inter + 1e-5) / (uni + 1e-5)
                    class_dices[c].append(dice.item())

    print("\n=================== 🦴 骨12部位別 Dice Score 評価 ===================")
    all_means = []
    for c in range(1, 13):
        m_dice = np.mean(class_dices[c])
        all_means.append(m_dice)
        print(f"  • クラス {c:02d} 平均 Dice : {m_dice:.4f}")
    print("--------------------------------------------------------------------")
    print(f" 🌟 全12部位の総合平均 Dice (Macro Mean): {np.mean(all_means):.4f}")
    print("====================================================================")


if __name__ == "__main__":
    validate()
