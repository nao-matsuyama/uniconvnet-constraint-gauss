import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset_scinti import ScintiMultiClassDataset
from model_uniconvnet_unet import UniConvNet_UNet_13CH


def visualize_results():
    # 1. デバイスの設定と重みのパス指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 今回は「最高精度を出した時点のパス」を直接指定します！
    latest_exp_dir = "experiments"
    weight_path = "experiments/best_uniconvnet_unet.pth"

    if not os.path.exists(weight_path):
        print(f"❌ 重みファイルが見つかりません: {weight_path}")
        return

    print(f"📂 学習済み重み '{weight_path}' を読み込んでいます...")

    # 2. データセットの準備（学習・評価時と全く同じ乱数シードを使用）
    full_dataset = ScintiMultiClassDataset()
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 画像ごとに評価するため batch_size=1 に設定
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 3. モデルの準備とロード
    model = UniConvNet_UNet_13CH(num_classes=13)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    results = []

    print("🔍 各画像のDiceスコアを計算中...")
    with torch.no_grad():
        for i, (image, mask) in enumerate(val_loader):
            img_tensor = image.to(device)
            mask_tensor = mask.to(device)

            output = model(img_tensor)
            pred = torch.argmax(output, dim=1)  # (1, H, W)

            # 画像内の12部位のDiceを計算して平均を取る
            dices = []
            for c in range(1, 13):
                pred_c = (pred == c).float()
                mask_c = (mask_tensor == c).float()
                inter = (pred_c * mask_c).sum()
                uni = pred_c.sum() + mask_c.sum()
                dice = (2.0 * inter + 1e-5) / (uni + 1e-5)
                dices.append(dice.item())

            mean_dice = np.mean(dices)

            # CPUに移動して保存
            results.append(
                {
                    "index": i,
                    "dice": mean_dice,
                    "image": image.squeeze().numpy(),  # (C, H, W) -> (H, W) or (C, H, W)
                    "gt": mask.squeeze().numpy(),  # (H, W)
                    "pred": pred.cpu().squeeze().numpy(),  # (H, W)
                }
            )

    # 4. Diceスコアでソートして、Best, Worst, Averageのインデックスを取得
    results.sort(key=lambda x: x["dice"])

    worst_case = results[0]  # 一番精度が低いもの
    best_case = results[-1]  # 一番精度が高いもの
    median_case = results[len(results) // 2]  # 真ん中（平均的）なもの

    cases_to_plot = [
        ("Worst Case", worst_case),
        ("Average Case", median_case),
        ("Best Case", best_case),
    ]

    # 5. 画像としてプロットして保存
    print("🎨 PNG画像を生成して保存しています...")
    # クラス分類用のカラーマップ（背景0〜クラス12まで見分けやすいように tab20 を使用）
    cmap = plt.get_cmap("tab20")

    for title, case in cases_to_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{title} - Mean Bone Dice: {case['dice']:.4f}", fontsize=16)

        # 元画像（1チャネルを想定して表示、もし3チャネルなら最初のチャネルを使用）
        img_display = case["image"]
        if img_display.ndim == 3:
            img_display = img_display[0]

        axes[0].imshow(img_display, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # 正解マスク (GT)
        axes[1].imshow(case["gt"], cmap=cmap, vmin=0, vmax=12)
        axes[1].set_title("Ground Truth (GT)")
        axes[1].axis("off")

        # 予測マスク
        axes[2].imshow(case["pred"], cmap=cmap, vmin=0, vmax=12)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        # 画像の保存
        save_filename = f"visualize_{title.replace(' ', '_').lower()}.png"
        save_path = os.path.join(latest_exp_dir, save_filename)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"✅ 保存完了: {save_path}")


if __name__ == "__main__":
    visualize_results()
