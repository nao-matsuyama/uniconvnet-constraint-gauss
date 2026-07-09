import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from model_uniconvnet_unet import UniConvNet_UNet_13CH

captured_map = None


def get_erf(model, target_layer_name, image_size=1024):
    global captured_map
    captured_map = None
    device = next(model.parameters()).device

    # 灰色のベース画像に微小なノイズを加えてDCN（可変形畳み込み）の挙動を安定させる
    base_image = torch.ones(1, 3, image_size, image_size, device=device) * 0.5
    noise = torch.randn(1, 3, image_size, image_size, device=device) * 0.05
    inputs = (base_image + noise).requires_grad_(True)

    def hook_fn(module, input, output):
        global captured_map
        if isinstance(output, torch.Tensor):
            captured_map = output
        elif isinstance(output, tuple) or isinstance(output, list):
            captured_map = output[0]

    # 指定されたレイヤーにフックを登録
    target_hook = None
    for name, layer in model.named_modules():
        if name == target_layer_name:
            target_hook = layer.register_forward_hook(hook_fn)
            break

    if target_hook is None:
        return None

    # 順伝播
    _ = model(inputs)
    target_hook.remove()

    if captured_map is None:
        return None

    features = captured_map
    # チャンネルの並び順が (B, H, W, C) の場合、(B, C, H, W) に変換
    if features.dim() == 4 and features.shape[1] == features.shape[2]:
        features = features.permute(0, 3, 1, 2)

    H, W = features.shape[2], features.shape[3]
    center_y, center_x = H // 2, W // 2

    # 中心ピクセルの全チャンネルの勾配の和をターゲットとする
    target = features[0, :, center_y, center_x].sum()

    model.zero_grad()
    target.backward()

    grad = inputs.grad[0]
    if grad is None:
        return None

    erf = grad.abs().mean(dim=0).cpu().numpy()
    return erf


def main():
    print("🚀 U-Net モデルの準備中...")
    model = UniConvNet_UNet_13CH(num_classes=13)

    weights_path = "experiments/best_uniconvnet_unet.pth"

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        load_info = model.load_state_dict(state_dict, strict=False)
        print("✅ 学習済み重みを完璧にロードしました！")
    else:
        print(f"❌ 重みファイルが見つかりません: {weights_path}")
        return

    model.cuda()
    model.eval()

    # ========================================================
    # 🎯 可視化するレイヤーの指定（解析ログに基づき修正完了）
    # ========================================================
    layers_to_visualize = {
        "Low (Encoder)": "backbone.stages.0",  # エンコーダー浅層
        "High (Bottleneck)": "backbone.stages.3",  # 最深部（ボトルネック）
        "Final (Decoder)": "final_head",  # 👈 'decoder' から 'final_head' に修正しました
    }

    num_samples = 100  # テスト用のサンプル数（動作確認後、綺麗にしたい場合は500等に増やしてください）
    results = {}

    # 1. 各層のERFを計算
    for title, layer_name in layers_to_visualize.items():
        print(f"\n🔄 [{title}] 層 '{layer_name}' の ERFを計算中...")

        # 最初の1回でテスト
        test_erf = get_erf(model, layer_name, 1024)
        if test_erf is None or test_erf.max() == 0:
            print(
                f"❌ エラー: '{layer_name}' という層から有効な勾配が計算できませんでした。"
            )
            return

        # 複数回サンプリングして平均化
        cumulative_erf = test_erf
        for i in range(num_samples - 1):
            res = get_erf(model, layer_name, 1024)
            if res is not None:
                cumulative_erf += res
            if (i + 2) % 20 == 0:
                print(f"  ... {i + 2}/{num_samples} 完了")

        final_erf = cumulative_erf / num_samples
        final_erf = final_erf / (final_erf.max() + 1e-7)

        # クロップ（中心部分を切り出し）
        crop_size = 600
        center = final_erf.shape[0] // 2
        erf_cropped = final_erf[
            center - crop_size // 2 : center + crop_size // 2,
            center - crop_size // 2 : center + crop_size // 2,
        ]
        results[title] = erf_cropped

    # 2. タイル状にプロット（2行 × 3列）
    print("\n🎨 結果をタイル状にプロットしています...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "ERF Progression in U-Net (Low $\\rightarrow$ High $\\rightarrow$ Decoder)",
        fontsize=18,
    )

    cmap_colors = [
        "#A1D99A",
        "#C7E8C1",
        "#756BB0",
        "#9C9AC8",
        "#BCBDDC",
        "#DBDAEC",
        "#636363",
        "#969696",
        "#BDBDBD",
        "#D9D9D9",
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_erf", cmap_colors)

    for col_idx, (title, erf_img) in enumerate(results.items()):
        # 上段：Linear Scale
        ax_lin = axes[0, col_idx]
        ax_lin.imshow(erf_img, cmap=custom_cmap)
        ax_lin.set_title(f"{title}\nLinear", fontsize=14)
        ax_lin.axis("off")

        # 下段：Log Scale
        ax_log = axes[1, col_idx]
        ax_log.imshow(np.log10(erf_img + 1e-7), cmap=custom_cmap)
        ax_log.set_title(f"{title}\nLog", fontsize=14)
        ax_log.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    save_path = "ERF_Tiled_Progression.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"✨ すべて完了しました！ タイル画像を保存しました: {save_path}")


if __name__ == "__main__":
    main()
