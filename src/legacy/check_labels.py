# coding:utf-8
import os

import numpy as np
import torch
from PIL import Image

# train.py で使われているデータセットクラスをそのままインポート
from dataset_scinti import ScintiMultiClassDataset


def verify_and_save_labels():
    print("🔄 骨シンチデータセットの読み込みを開始します...")

    # 1. データセットの初期化
    try:
        dataset = ScintiMultiClassDataset()
        print(f"✅ データセットの初期化に成功しました。総データ数: {len(dataset)} 件")
    except Exception as e:
        print(f"❌ データセットの初期化に失敗しました。原因: {e}")
        return

    if len(dataset) == 0:
        print("❌ データセットが空っぽです！画像やラベルのパスを確認してください。")
        return

    # 2. 保存先フォルダの作成
    output_dir = "./check_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 骨シンチの13クラスを識別しやすいように色分けするカラーパレット (RGB)
    color_palette = [
        0,
        0,
        0,  # 0: 背景 (黒)
        255,
        0,
        0,  # 1: 部位1 (赤)
        0,
        255,
        0,  # 2: 部位2 (緑)
        0,
        0,
        255,  # 3: 部位3 (青)
        255,
        255,
        0,  # 4: 黄
        255,
        0,
        255,  # 5: 紫
        0,
        255,
        255,  # 6: 水色
        128,
        0,
        0,  # 7: 茶色
        0,
        128,
        0,  # 8: 深緑
        0,
        0,
        128,  # 9: 紺
        128,
        128,
        0,  # 10: オリーブ
        128,
        0,
        128,  # 11: 濃紫
        0,
        128,
        128,  # 12: 鴨の羽色
    ]
    # PILのPモード（パレット変換）に必要な768要素（256色分）になるよう残りを0で埋める
    color_palette += [0] * (768 - len(color_palette))

    # 3. 先頭から最大5枚を抽出してPNG変換
    num_to_check = min(5, len(dataset))
    print(f"📸 先頭 {num_to_check} 枚の正解マスクをPNGとして書き出します...")

    for i in range(num_to_check):
        try:
            # データセットから画像とマスクを取得
            image, mask = dataset[i]

            # マスクの型と形状を確認
            # mask が PyTorch テンソルの場合は numpy に変換
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)

            mask_np = mask_np.astype(np.uint8)

            print(f"\n--- サンプル [{i}] ---")
            print(f" 形状 (Shape): {mask_np.shape}")

            # 💡 もしデータセットが最初から one-hot [13, H, W] で返している場合の対策
            if len(mask_np.shape) == 3:
                if mask_np.shape[0] == 13:
                    mask_np = np.argmax(mask_np, axis=0).astype(np.uint8)
                else:
                    mask_np = mask_np[0]  # チャンネル次元が1の場合など

            # 含まれている数値をチェック（0しか入っていなければ背景のみ＝データ不良）
            unique_labels = np.unique(mask_np)
            print(f" 検出されたラベルID: {unique_labels}")

            # カラーマップ付きPNGとして保存
            img_mask = Image.fromarray(mask_np, mode="P")
            img_mask.putpalette(color_palette)

            save_path = os.path.join(output_dir, f"label_sample_{i}.png")
            img_mask.save(save_path)
            print(f" 💾 保存完了: {save_path}")

        except Exception as e:
            print(f" ❌ サンプル [{i}] の処理中にエラーが発生しました: {e}")

    print(
        f"\n✅ すべての確認用書き出しが完了しました！ `./check_outputs` フォルダを確認してください。"
    )


if __name__ == "__main__":
    verify_and_save_labels()
