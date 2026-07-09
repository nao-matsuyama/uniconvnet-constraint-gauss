import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

# 🌟 学習済みの U-Net モデルを読み込む
from model_uniconvnet_unet import UniConvNet_UNet_13CH

# グローバル変数で特徴マップを一時保管
captured_map = None


def get_erf(model, target_layer_name, image_size=1024):
    global captured_map
    captured_map = None
    device = next(model.parameters()).device

    # グレー画像+微小ノイズを入力してDCNなどの挙動を安定させる
    base_image = torch.ones(1, 3, image_size, image_size, device=device) * 0.5
    noise = torch.randn(1, 3, image_size, image_size, device=device) * 0.05
    inputs = (base_image + noise).requires_grad_(True)

    # フック関数：呼ばれたら無条件で保存する
    def hook_fn(module, input, output):
        global captured_map
        if isinstance(output, torch.Tensor):
            captured_map = output
        elif isinstance(output, tuple) or isinstance(output, list):
            captured_map = output[0]  # タプルで返ってくる層の対策

    # 🌟 指定した名前の層「だけ」を探してフックを仕掛ける
    target_hook = None
    for name, layer in model.named_modules():
        if name == target_layer_name:
            target_hook = layer.register_forward_hook(hook_fn)
            break

    if target_hook is None:
        return None  # 指定された層が見つからない場合

    # フォワードパス
    _ = model(inputs)

    # フック解除
    target_hook.remove()

    if captured_map is None:
        return np.zeros((image_size, image_size))

    features = captured_map

    # 次元を (B, C, H, W) に強制統一
    if features.dim() == 4 and features.shape[1] == features.shape[2]:
        features = features.permute(0, 3, 1, 2)

    H, W = features.shape[2], features.shape[3]
    center_y, center_x = H // 2, W // 2

    # 💡 中間層は解像度が小さいため、中心の1ピクセルのみをターゲットにする
    target = features[0, :, center_y, center_x].sum()

    model.zero_grad()
    target.backward()

    grad = inputs.grad[0]
    if grad is None:
        return np.zeros((image_size, image_size))

    erf = grad.abs().mean(dim=0).cpu().numpy()
    return erf


def main():
    print("🚀 U-Net モデルの準備中...")
    model = UniConvNet_UNet_13CH(num_classes=13)

    # 🔥 今回最高精度(Dice 0.9140)を出した重みを読み込む
    weights_path = "experiments/best_uniconvnet_unet.pth"

    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("✅ 最高精度の学習済み重みをロードしました！")
    else:
        print(f"❌ 重みファイルが見つかりません: {weights_path}")
        return

    model.cuda()
    model.eval()

    # ========================================================
    # 🔍 【超便利機能】モデル内のすべての層の名前をファイルに出力する
    # どの層を指定すればいいか迷った時は、このテキストファイルを確認してください！
    # ========================================================
    with open("model_layers_list.txt", "w") as f:
        for name, _ in model.named_modules():
            f.write(name + "\n")
    print("📝 モデルの全レイヤー名を 'model_layers_list.txt' に出力しました。")

    # ========================================================
    # 🎯 見たい中間層の名前を指定する（※ここを書き換えて遊びます）
    # ========================================================
    # 例として適当な名前を入れています。実際の名前は model_layers_list.txt を見て変更してください
    target_layer = "encoder.layers.0"  # 👈 ここを見たい層の名前に書き換える！

    print(f"🔄 層 [{target_layer}] の ERFを計算中...")

    # 💡 テスト用に100回に減らしています（本番の綺麗な図が欲しい時は1000にしてください）
    num_samples = 100

    cumulative_erf = 0
    for i in range(num_samples):
        erf_result = get_erf(model, target_layer_name=target_layer, image_size=1024)

        if erf_result is None:
            print(f"❌ エラー: '{target_layer}' という名前の層が見つかりません。")
            print(
                "💡 'model_layers_list.txt' を開いて、正しい層の名前を探してコピペしてください！"
            )
            return

        cumulative_erf += erf_result
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{num_samples} 完了")

    final_erf = cumulative_erf / num_samples

    if final_erf.max() == 0:
        print("❌ 勾配が計算できませんでした。")
        return

    final_erf = final_erf / final_erf.max()

    crop_size = 600
    center = final_erf.shape[0] // 2
    erf_cropped = final_erf[
        center - crop_size // 2 : center + crop_size // 2,
        center - crop_size // 2 : center + crop_size // 2,
    ]

    plt.figure(figsize=(10, 5))
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

    plt.subplot(1, 2, 1)
    plt.imshow(erf_cropped, cmap=custom_cmap)
    plt.title(f"ERF: {target_layer} (Linear)")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(erf_cropped + 1e-7), cmap=custom_cmap)
    plt.title(f"ERF: {target_layer} (Log)")
    plt.colorbar()

    plt.tight_layout()

    # 保存ファイル名に層の名前を入れる（スラッシュやドットはアンダースコアに変換）
    safe_layer_name = target_layer.replace(".", "_")
    save_path = f"ERF_{safe_layer_name}.png"

    plt.savefig(save_path)
    print(f"✨ 完了！ 画像を保存しました: {save_path}")


if __name__ == "__main__":
    main()
