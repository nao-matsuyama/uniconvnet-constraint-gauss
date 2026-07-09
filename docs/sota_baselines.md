# SOTA ベースライン布陣と統一評価プロトコル

骨シンチ 12 部位（13 クラス）セグメンテーションで UniConvNet-T U-Net の優位/中立を
**公平かつ多指標で**示すための比較対象（SOTA）を確定する。参照: Sparse-MoE-SAM
(PMC12430776) の比較設計（U-Net / Attention U-Net / DeepLabV3+ / SegFormer / SAM 系）。

## 1. 確定した布陣（4 本 + 本命）

| 役割 | モデル | 文脈/RF の獲得方法 | 本研究での意味 |
|---|---|---|---|
| 下限基準 | **vanilla U-Net** | 局所 conv のみ | 文脈拡大なしの素の基準線 |
| skip 選別 | **Attention U-Net** | attention gate で skip を選別 | 文脈/選択機構の対抗馬 |
| atrous 文脈 | **DeepLabV3+** | ASPP（atrous=固定 dilation 多枝） | **dilation ベース RF 拡大の直接の対抗概念** |
| 大域 attention | **SegFormer (MiT-B*)** | self-attention で大域文脈 | RF=画像全体の極（Transformer 系） |
| 本命 | **UniConvNet-T U-Net (+SpectralGaussianDW)** | 周波数ガウス低域通過（学習可能 σ, FFT で RF 非依存コスト） | 本研究の提案機構 |

実装手段（segmentation_models_pytorch / torchvision / 自前）は **後で決定**。布陣のみ確定。
DeepLabV3+ は本研究の dilation 系 RF 拡大の最も近い既存手法なので必須。SegFormer は
「大域文脈の上限」を与え、骨シンチが本当に局所で解けるなら Transformer でも勝てないことの
傍証になる（負の結果側の補強）。

## 2. 公平性プロトコル（全モデル共通・厳守）

各ベースラインを下記で完全に揃える。揃っていない比較は SOTA 比較として無効とする。

- **データ分割**: `random_split(full, [0.8, 0.2], generator=torch.Generator().manual_seed(42))`
  （train.py / evaluate.py / compare_models.py / evaluate_predictions.py と同一）。
- **入力前処理**: grayscale .mhd を per-image min-max で [0,1] 正規化 → 3ch 複製
  （`dataset_scinti.ScintiMultiClassDataset.__getitem__` と同一: `unsqueeze(0).repeat(3,1,1)`）。
  ベースラインも同じ 3ch 入力・同じ正規化を使う（ImageNet 事前学習を使う場合のみ
  そのモデルの mean/std 正規化を追加してよいが、その旨を run_meta に明記）。
- **クラス数 / 損失**: 13 クラス、train.py と同じ損失（CE + Dice 等、要統一）。
- **学習予算**: epoch 数・optimizer・lr スケジュール・batch size・augmentation を統一
  （別紙の学習設定に固定。事前学習重みの有無は run ごとに run_meta に記録）。
- **採点**: 後述の `metrics.py` 一本。空クラスは NaN 除外（smoothing で 1.0 にしない）。

## 3. 統一評価ハーネス（実装非依存の採点口）

モデルの実装方式に依存せず、**予測ラベルマップを保存 → 共通スコアラで採点**する。

```
[各モデルの推論スクリプト]  seed42 の val を回す
        │  予測ラベルマップ (H,W, int 0..12) を保存
        ▼
  preds/<model>/  ──►  src/evaluate_predictions.py  ──►  report.txt / per_class.csv / per_sample.csv
        ▲                         │
        └── GT は seed42 val 分割から自動取得
```

### 予測の保存形式（`--pred-dir` 内、いずれか）
- **方式A**: `<image basename>.npy`（各 .mhd と同じ stem、int ラベルマップ (H,W)）
- **方式B**: `predictions.npz`（キー=fname stem、値=ラベルマップ (H,W)）

予測は GT と同じ (H,W) であること（リサイズしたモデルは元解像度へ戻して保存）。

### 採点コマンド（全モデル共通）
```bash
python3 src/evaluate_predictions.py \
  --pred-dir /workspace/preds/<model> \
  --data-dir /workspace/scinti_segmentation \
  --label <model> \
  --out-dir /workspace/eval_results/<model>
```
UniConvNet 自身は `src/evaluate.py --weights ...` で直接採点でき、出力フォーマットは
`evaluate_predictions.py` と共有（`eval_report.py`）なので完全に同一。

## 4. 指標一式（`src/metrics.py`）

| 種別 | 指標 | 向き | 意味 |
|---|---|---|---|
| 重なり | dice | ↑ | 2TP/(2TP+FP+FN) |
| 重なり | iou (Jaccard) | ↑ | TP/(TP+FP+FN) |
| 重なり | precision | ↑ | TP/(TP+FP)（予測空は NaN） |
| 重なり | recall | ↑ | TP/(TP+FN)（GT 空は NaN） |
| 重なり | specificity | ↑ | TN/(TN+FP) |
| 境界 | hd95 | ↓ | 95%タイル対称 Hausdorff |
| 境界 | hd | ↓ | 最大対称 Hausdorff |
| 境界 | assd | ↓ | 平均対称表面距離 |
| 境界 | nsd@τ | ↑ | 許容 τ 内の表面点割合（Normalized Surface Dice） |

集計: マクロ（部位平均, 主指標）＋ micro（プール, Dice/IoU）。空クラスは NaN 除外。

## 5. worst（悪いデータ）解析 — 本研究の主張軸

「平均では差が出ないが、局所が曖昧な最悪例で文脈/RF 拡大が効く」を検証する。
`compare_models.py` がベースラインの per-sample mean Dice 昇順で worst 10%/25% 群を固定し、
全指標で群別 mean ＋ ペア Wilcoxon を出す。境界系は worst 群で差が開きやすい
（合成検証でも worst40% で hd95 が 4.0→9.1、nsd@1 が 0.74→0.35 と Dice より鋭く分離）。

```bash
python3 src/compare_models.py \
  --weights .../run_baseline/best.pth .../run_spectral/best.pth \
  --labels baseline spectral --data-dir /workspace/scinti_segmentation \
  --out-dir /workspace/compare_results/spectral_vs_baseline --plot-metric hd95
```

## 6. 次の実装ステップ（布陣確定後）

1. 実装手段の決定（smp / torchvision / 自前）→ requirements.txt 更新。
2. 各ベースラインの学習スクリプト（プロトコル §2 厳守）。
3. 各ベースラインの推論→予測ダンプ（§3 形式）。UniConvNet 用 `dump_predictions.py` も用意し
   全モデルを `evaluate_predictions.py` の単一経路に通す。
4. 全モデルを評価→ per_class / worst の集計表を報告書へ。
