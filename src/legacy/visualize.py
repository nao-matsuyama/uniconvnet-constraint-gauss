# coding:utf-8
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from dataset_scinti import ScintiMultiClassDataset
from model_uniconvnet_unet import UniConvNet_UNet_13CH


def save_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "visualization_results"
    os.makedirs(out_dir, exist_ok=True)

    dataset = ScintiMultiClassDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UniConvNet_UNet_13CH(num_classes=13).to(device)
    if os.path.exists("experiments/best_uniconvnet_unet.pth"):
        model.load_state_dict(
            torch.load("experiments/best_uniconvnet_unet.pth", map_location="cpu")
        )
    model.eval()

    # 🎨 13クラス固定のカラフルなカラーマップを作成（バグ防止）
    cmap = plt.cm.get_cmap("tab20", 13)
    cmap_colors = cmap(np.arange(13))
    cmap_colors[0] = [0, 0, 0, 1]  # 背景(0)は完全な黒にする
    custom_cmap = matplotlib.colors.ListedColormap(cmap_colors)

    print("📸 13クラスマルチ分類の可視化を開始します...")
    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            if idx >= 5:
                break  # 最初の5例のみ出力

            outputs = model(images.to(device))
            ai_msk = torch.argmax(outputs, dim=1).cpu().numpy()[0]  # (H, W)
            true_msk = masks.cpu().numpy()[0]  # (H, W)
            img_np = images[0, 0].cpu().numpy()

            # 💾 1. ITK-SNAP用の13クラス一体型MHD書き出し
            sitk.WriteImage(
                sitk.GetImageFromArray(ai_msk.astype(np.uint8)),
                os.path.join(out_dir, f"case_{idx:03d}_13ch_pred.mhd"),
            )

            # 🖼️ 2. 3並列の比較画像生成 (元画像 | 人間GT | AI予測)
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # 左：元画像
            axes[0].imshow(img_np, cmap="gray")
            axes[0].set_title("1. Original Image")
            axes[0].axis("off")

            # 中央：人間が作成した正解（マルチクラス）
            # 💡 vmin/vmaxを0~12に固定することで、色が絶対に狂わなくなります
            axes[1].imshow(true_msk, cmap=custom_cmap, vmin=0, vmax=12)
            axes[1].set_title("2. True Mask (Ground Truth)")
            axes[1].axis("off")

            # 右：AIが予測した部位（マルチクラス）
            axes[2].imshow(ai_msk, cmap=custom_cmap, vmin=0, vmax=12)
            axes[2].set_title("3. AI Predicted Mask")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"case_{idx:03d}_multi_result.png"),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()
            print(f"   [Saved] case_{idx:03d}_multi_result.png")


if __name__ == "__main__":
    save_predictions()
