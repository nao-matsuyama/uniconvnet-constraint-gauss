# coding:utf-8
import glob
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


# ファイル名末尾のビュー記号 (…_A.mhd = Anterior/正面, …_P.mhd = Posterior/背面)。
# 各症例は _A と _P の 2 ビューを持つ。view で読み込む側を絞れる。
_VIEW_ALIASES = {
    "both": "both",
    "all": "both",
    "anterior": "A",
    "a": "A",
    "front": "A",
    "正面": "A",
    "posterior": "P",
    "p": "P",
    "back": "P",
    "背面": "P",
}


def view_of(path):
    """ファイルパスからビュー記号 'A'/'P' を返す (末尾 _A/_P)。不明は '?'。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    tok = stem.rsplit("_", 1)[-1].upper()
    return tok if tok in ("A", "P") else "?"


class ScintiMultiClassDataset(Dataset):
    def __init__(self, data_dir="/workspace/scinti_segmentation", view="both"):
        self.data_dir = self._resolve_data_dir(data_dir)
        self.image_dir = os.path.join(self.data_dir, "image")
        self.bone_dir = os.path.join(self.data_dir, "bone")

        v = _VIEW_ALIASES.get(str(view).strip().lower())
        if v is None:
            raise ValueError(
                f"view は both/anterior/posterior のいずれか (指定: {view!r})"
            )
        self.view = v

        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.mhd")))
        self.bone_files = sorted(glob.glob(os.path.join(self.bone_dir, "*.mhd")))

        if len(self.image_files) == 0:
            raise FileNotFoundError(
                "❌ 画像ファイルが見つかりません: "
                f"{self.image_dir}\n"
                "   --data-dir で scinti_segmentation の親ディレクトリを指定してください。"
            )

        # ビューで絞る (both は無フィルタ)。A/P はファイル名末尾記号で判定。
        if self.view in ("A", "P"):
            n_before = len(self.image_files)
            self.image_files = [f for f in self.image_files if view_of(f) == self.view]
            self.bone_files = [f for f in self.bone_files if view_of(f) == self.view]
            label = "Anterior(正面)" if self.view == "A" else "Posterior(背面)"
            print(
                f"🧭 view={label}: {len(self.image_files)}/{n_before} 枚を選択"
                f" (残りは他ビューとして除外)"
            )
            if len(self.image_files) == 0:
                raise FileNotFoundError(
                    f"❌ view={label} に該当するファイルがありません。"
                    "ファイル名末尾の _A/_P 記号を確認してください。"
                )

        if len(self.image_files) != len(self.bone_files):
            print(
                f"⚠️ 警告: 画像数 ({len(self.image_files)}) と骨マスク数 ({len(self.bone_files)}) が一致しません。"
            )

    @staticmethod
    def _resolve_data_dir(data_dir):
        candidates = []
        if data_dir:
            candidates.append(Path(data_dir))

        env_dir = os.environ.get("SCINTI_DATA_DIR")
        if env_dir:
            candidates.append(Path(env_dir))

        repo_root = Path(__file__).resolve().parent.parent
        candidates.extend(
            [
                repo_root / "scinti_segmentation",
                repo_root / "data" / "scinti_segmentation",
                Path("/workspace/scinti_segmentation"),
                Path("/workspace/data/scinti_segmentation"),
            ]
        )

        for candidate in candidates:
            image_dir = candidate / "image"
            bone_dir = candidate / "bone"
            if image_dir.is_dir() and bone_dir.is_dir():
                return str(candidate)

        searched = "\n".join(f" - {p}" for p in candidates)
        raise FileNotFoundError(
            "❌ scinti_segmentation の配置先を見つけられませんでした。\n"
            "   次の候補を確認しました:\n"
            f"{searched}\n"
            "   環境変数 SCINTI_DATA_DIR か --data-dir を指定してください。"
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_np = sitk.GetArrayFromImage(sitk.ReadImage(self.image_files[idx])).astype(
            np.float32
        )
        # 💡 13クラス分類のインデックス(0~12)として扱うため int64 型で読み込み
        bone_np = sitk.GetArrayFromImage(sitk.ReadImage(self.bone_files[idx])).astype(
            np.int64
        )

        # 輝度正規化 (0.0 ~ 1.0)
        min_val, max_val = img_np.min(), img_np.max()
        if max_val - min_val > 0:
            img_np = (img_np - min_val) / (max_val - min_val)
        else:
            img_np = np.zeros_like(img_np)

        # UniConvNetへの入力用に3チャンネル化 (3, H, W)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).repeat(3, 1, 1)
        # マスクはチャンネル次元なしの (H, W)
        mask_tensor = torch.from_numpy(bone_np)

        return img_tensor, mask_tensor
