import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from model_uniconvnet_unet import UniConvNet_UNet_13CH

# 特徴マップをステージごとに自動収集するためのリスト
captured_maps = []


def get_erf_all_stages(model, image_size=1024):
    global captured_maps
    captured_maps = []
    device = next(model.parameters()).device

    # 灰色のベース画像に微小なノイズを加えてDCN（可変形畳み込み）の挙動を安定させる
    base_image = torch.ones(1, 3, image_size, image_size, device=device) * 0.5
    noise = torch.randn(1, 3, image_size, image_size, device=device) * 0.05
    inputs = (base_image + noise).requires_grad_(True)

    # フック関数：条件に合うすべての特徴マップを、通過した順番に記録する
    def hook_fn(module, input, output):
        global captured_maps
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            s1, s2, s3 = output.shape[1], output.shape[2], output.shape[3]
            # 縦横が正方形かつ一定以上のサイズを持つテンソルをすべて記録
            if (s2 > 7 and s3 > 7 and s2 == s3) or (s1 > 7 and s2 > 7 and s1 == s2):
                captured_maps.append(output)

    # ========================================================
    # 🎯 修正のコア：親コンテナを無視し、子を持たない末端の演算層だけを狙い撃ち
    # ========================================================
    hooks = []
    for name, layer in model.named_modules():
        # 子モジュールを持たない末端の層（Conv2dやLinear、DCNの演算層などウェイトを持つ層）だけを指定
        if len(list(layer.children())) == 0 and isinstance(
            layer, (torch.nn.Conv2d, torch.nn.Linear)
        ):
            hooks.append(layer.register_forward_hook(hook_fn))

    # 順伝播（これで親ブロックの重複がない、純粋な演算直後の特徴マップが順番に格納される）
    _ = model(inputs)

    # フックを即座に解除
    for h in hooks:
        h.remove()

    if len(captured_maps) == 0:
        return None, None, None

    # ========================================================
    # 🎯 全体の流れから【最初・真ん中・最後】の特徴マップを自動抽出
    # ========================================================
    # 1. Low Stage (エンコーダー初期：最初の方に通過したマップ)
    low_idx = int(len(captured_maps) * 0.05)  # 最初の5%付近
    # 2. High Stage (ボトルネック付近：中間地点で解像度が最も低いマップの直前など)
    mid_idx = int(len(captured_maps) * 0.45)  # 中盤の45%付近
    # 3. Final Stage (デコーダー最終盤：一番最後に通過したマップ)
    final_idx = len(captured_maps) - 1  # 最後の1個手前の最終特徴層

    selected_stages = {
        "Low (Encoder)": captured_maps[low_idx],
        "High (Bottleneck)": captured_maps[mid_idx],
        "Final (Decoder)": captured_maps[final_idx],
    }

    erf_results = {}

    # 各ステージの特徴マップごとに個別に逆伝播を行って勾配（受容野）を計算
    for title, features in selected_stages.items():
        # 次元の並び順を自動検知して (Batch, Channel, Height, Width) に強制統一
        if features.shape[1] == features.shape[2]:
            features = features.permute(0, 3, 1, 2)

        H, W = features.shape[2], features.shape[3]
        center_y, center_x = H // 2, W // 2

        # 中心ピクセルの全チャンネルの和をターゲットとする
        target = features[0, :, center_y, center_x].sum()

        model.zero_grad()
        if inputs.grad is not None:
            inputs.grad.zero_()

        target.backward(retain_graph=True)  # 複数回 backward するためグラフを保持

        grad = inputs.grad[0]
        if grad is not None:
            erf = grad.abs().mean(dim=0).cpu().numpy()
            erf_results[title] = erf
        else:
            erf_results[title] = np.zeros((image_size, image_size))

    return (
        erf_results["Low (Encoder)"],
        erf_results["High (Bottleneck)"],
        erf_results["Final (Decoder)"],
    )


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

    num_samples = 100  # テスト用のサンプル数（本番は500〜1000推奨）
    print(
        f"🔄 層の名前指定なしで自動トラッキング中... ({num_samples}回の平均を取ります)"
    )

    results = {"Low (Encoder)": 0, "High (Bottleneck)": 0, "Final (Decoder)": 0}

    for i in range(num_samples):
        low_erf, mid_erf, final_erf = get_erf_all_stages(model, image_size=1024)

        if low_erf is not None:
            results["Low (Encoder)"] += low_erf
            results["High (Bottleneck)"] += mid_erf
            results["Final (Decoder)"] += final_erf

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{num_samples} 完了")

    # 正規化とクロップ処理
    for title in results.keys():
        final_img = results[title] / num_samples
        if final_img.max() > 0:
            final_img = final_img / (final_img.max() + 1e-7)

        crop_size = 600
        center = final_img.shape[0] // 2
        results[title] = final_img[
            center - crop_size // 2 : center + crop_size // 2,
            center - crop_size // 2 : center + crop_size // 2,
        ]

    # 🎨 タイル状にプロット（2行 × 3列）
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

    for col_idx, title in enumerate(
        ["Low (Encoder)", "High (Bottleneck)", "Final (Decoder)"]
    ):
        erf_img = results[title]

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

    save_path = "ERF_Auto_Tiled_Progression.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"✨ すべて完了しました！ タイル画像を保存しました: {save_path}")


if __name__ == "__main__":
    main()
