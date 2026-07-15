# CLAUDE.md

UniConvNet-T U-Net による骨シンチ13クラスセグメンテーションの研究リポジトリ
(`uniconvnet-t` から fork、2026-07-09)。**このフォルダの主眼は
「RFA(ConvMod) の受容野拡大に、ガウス性をより強い"構造的制約"として結ぶ」本命機構
`gauss_deriv`(ガウス微分/スケールスペース基底)の設計・検証**。

背景: 既存の周波数ガウス機構(機構B/C)はガウス性を `out=local(x)+gamma·spectral(x)` の
**加法枝**として後付けするため、gamma≈0 でネットに無効化されうる(骨シンチで実測)。かつ
純ガウス(0次)は境界をボカす。`gauss_deriv` はカーネル自体を `W=H(σ)·A·H(σ)ᵀ`
(ガウス微分基底)に閉じ込め、**受容野スケールを σ のみに縛る**(local枝/gammaなし=ハード制約)。
微分項(1次=エッジ, 2次=リッジ)で純ガウスの境界ボケを回避しつつ AGD を構造的に保つ。
比較対象として旧アークの効率化機構(dilated / separable / spectral / spectral_mix)も温存。

## プロジェクト種別

- **Python ML 研究リポジトリ**(package.json は無い)。PyTorch ベース。
- 依存: `torch`, `matplotlib`, `numpy`, `scipy`, `SimpleITK`, `timm`(`requirements.txt`)。
- 実行は基本 **Docker**(GPU=サーバー tsubaki / CPU 検証=ローカル)。ホスト直 python は不可(torch 無し)。

## ディレクトリ

```
src/
  models/
    UniConvNet.py            # 本体(ConvMod=RFA / Block / UniConvNet)。dw_mode の分岐はここ
    dcnv3_pytorch.py         # 純PyTorch版 DCNv3(コンパイル不要)
    dilated_dw.py            # 学習可能連続dilation depthwise
    separable_dw.py          # 機構A: 1×K・K×1 分離(rank-1 SVD 転移)
    spectral_dw.py           # 機構B: 周波数ガウス+動的切り出し(pad/quantile 改善ノブ)
    spectral_mixture_dw.py   # 機構C: 多ガウス混合(K本の周波数ガウス和)+ 分離rank-K + 振幅考慮切り出し
    gaussian_derivative_dw.py # ★本命: ガウス微分基底 W=H(σ)AH(σ)ᵀ。RF を σ のみに縛る(local枝/gammaなし)
    gaussian_pyramid_dw.py    # 多スケール純ガウス(scale-space pyramid)。SpectralDW派生。境界は pointwise の DoG で復元
    spectral_gaussian_dw.py  # 旧: フルサイズ周波数ガウス
    content_adaptive_dw.py   # 旧: コンテンツ適応 dilation
    erf_regularization.py    # ERF正則化損失(rf_spread を目標へ)
  model_uniconvnet_unet.py   # U-Net ラッパ UniConvNet_UNet_13CH
  ckpt_utils.py              # build_model_from_checkpoint(dw_mode 自動判別)。評価系の要
  train.py                   # 学習(dw_mode/spectral各種フラグ, σ専用param group)
  benchmark.py               # params/FLOPs/img-s/RFA省コスト比
  compare_models.py          # worst×境界(hd95/assd/nsd)の Wilcoxon 比較
  evaluate.py / eval_report.py / metrics.py   # 部位別/worst 全指標(smoothing無し, 空クラスNaN)
  erf_sigma_table.py         # 層別 ERF σ_moment & excess kurtosis 表
  visualize_erf.py           # 1層6面ERF図(--tag でモデル別)
  compare_erf.py             # 複数モデルの層別ERF断面+FFT重ね描き
  erf_gmm_fit.py             # ERF断面を N ガウス和でフィット(--cusp)。AGD=集約ガウス検証
  agd_probe.py               # 単層カーネルの AGD 比較(学習不要, ローカルCPU)。gauss-deriv(smooth) 含む
  verify_gauss_deriv.py      # gauss_deriv の転移/eval往復/captured energy を一括検証
  visualize_sigma.py         # SpectralDW の σ が学習で動いたか(凍結/moved)
  visualize_predictions.py   # 単一モデルの推定マスク(Worst/Avg/Best, --save-all)
  prediction_comparison.py   # 複数機構の予測を横並び比較(prediction_comparison.png)
  list_runs.py               # experiments/ の run 一覧表(dw_mode/best_dice/ckpt)
  smoke_test.py              # CPU構造テスト(全dw_mode)
  model_stats.py             # FLOPs/params(各機構のフック)
  dataset_scinti.py          # 骨シンチ dataloader(正規化+3ch化のみ)
  augment_scinti.py          # train用データ拡張(--aug: 左右swap付き水平反転+小アフィン+強度)
docs/
  EXPERIMENTS.md             # ★運用リファレンス(trainレシピ/評価/フォルダ規約/サーバー運用)
  report_agd_efficient_dw.md # ★2機構の実装・検証レポート(全アーク)
  sota_baselines.md / progress_report_*.md / report_erf_eval.md
experiments/run_<ts>[_<tag>]/  # 学習成果(best_uniconvnet_unet.pth / run_config.json / run_info.txt)
docker/ makefile requirements.txt run.ps1
uniconvnet_t_1k_224_ema*.pth   # ImageNet 事前学習
```

## dw_mode(RFA depthwise の機構切替)

`ConvMod`/`UniConvNet_UNet_13CH` に `dw_mode ∈ {dense(既定), separable, spectral, spectral_mix, gauss_deriv}`。
既定 dense は現状維持(事前学習そのまま)。`spectral_mix`(機構C)は `--spectral-num-gaussians K`(既定3)で
成分数を指定(K=1 で機構B spectral に一致)。**`separable`(機構A)は `--separable-rank R`(既定1)** で分離
rank を指定(R 本の 1×K・K×1 の和=タップ 2RK、rank-R SVD で転移)。評価系 `build_model_from_checkpoint` は
**checkpoint 同フォルダの `run_config.json` から dw_mode 等を自動判別**(機構C は log_sigma (C,K)、機構A は
weight_h (C,R,1,K)、gauss_deriv は coeff (C,M,M) から復元)するので、ckpt 移動時は run_config.json も一緒に扱う。

### ★ 本命 `gauss_deriv`(ガウス微分基底 — ガウス性を構造的制約に)

カーネルを **ガウス微分(スケールスペース)基底** に閉じ込める:
`w_c = Σ_{p,q≤N} a_{c,pq} ∂ₓ^p∂_y^q G_{σ_c}` = `W = H(σ_c)·A_c·H(σ_c)ᵀ`
(`H`=1次元エルミート×ガウス基底 (K,M), `A`=学習係数 (M,M), `M=N+1`)。
- **RF スケールは σ_c ただ一つが支配**。local枝/gamma を持たない=演算子全体が基底の内側
  =ガウス性がハード制約(機構B/C の「gamma で無効化」を構造的に封じる)。
- 微分項(N≥1: 1次=方向エッジ, 2次=リッジ)で純ガウス(N=0)の境界ボケを回避。
- AGD: 多項式×ガウスはガウス裾 → dilated のような高周波複製ローブが出ない(agd_probe で
  gauss-deriv(smooth)=dense と数値一致=確認済)。
- forward は `W=HAHᵀ` が rank≤M の事実で **M 本の 1×K・K×1 分離畳み込みの和**(タップ 2MK、FFT不使用=
  DataParallel 可)。事前学習は密カーネルを基底へ最小二乗射影(`load_dense_into_gaussian_derivative`)。
- ノブ: **`--gauss-deriv-order N`(既定2)** が制約↔表現力の唯一の軸(N=0 純ガウス〜N大 無制約)。
  σ(RF)は `--spectral-init-sigma`/`--spectral-max-sigma`(fmap px, stage別可)で指定、
  `--spectral-sigma-lr 1e-2` で σ を学習可能に(log_sigma 専用 param group)。K は max_sigma から自動
  (±3σ, 上限41)。

**推奨レシピ(coarse-to-fine)**: init σ は事前学習カーネルの実効スケール(≈1)に合わせ **小さく**、
max_sigma を大きく取り σ-lr で σ を育てる。最初から広い σ は崩壊/局所解(spectral アークの教訓)。
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode gauss_deriv \
  --gauss-deriv-order 2 --spectral-init-sigma 1.5 --spectral-max-sigma 8 6 4 3 \
  --spectral-sigma-lr 1e-2 --batch-size 16 --tag gaussderiv_n2
```

**転移の captured energy(検証済, 2026-07-09)**: 事前学習密カーネルをこの基底が捉える割合は
**N=2 で ~0.5-0.6 / N=1 で ~0.35**(σ=1 が最良、separable rank-1 の 0.80 よりずっとタイト)。
=制約が強く init は弱いwarm-start。精度は fine-tune + σ 学習で取り戻す設計(要実機検証)。
まず `baseline(dense)` vs `gauss_deriv N=2` を worst×境界(compare_models)で、次に **N=0 vs N=2** で
「微分項が境界を救うか」を切り分ける。効くとすれば骨シンチ(局所)より RF が効くタスク。

### `gauss_pyramid`(多スケール純ガウス + pointwise DoG, 2026-07-10)

RFA を「大小の学習カーネルで AGD を作る」から「最初から純ガウス」へ。a1/a2/a3 を σ1<σ2<σ3 の
**純ガウス**(`SpectralDW` を use_local_branch=False で派生 `GaussianPyramidDW`)にしてスケール空間
ピラミッドを張る。**カスケード=ガウス半群** `G_σ1*G_σ2=G_√(σ1²+σ2²)` で実効σが自然増大。
`--gauss-pyramid-growth`(既定1.6, 枝σ=init·growth^{0,1,2})で明示的に広げる。**枝σが stage の
max_sigma を超えると clamp 飽和で σ の勾配が消える**ので `init·growth² ≤ max_sigma`(例 init 1 growth
1.6 max 8 6 4 3 で全枝可動)。**境界は depthwise でなく pointwise(v-conv) の DoG で復元**(純ガウス+DoG,
local枝/gamma なし)。σ既定学習可(`--spectral-sigma-lr 1e-2`)/`--freeze-scale`で固定。**FFT系=単一GPU厳守**。
ckpt は run_config の dw_mode=gauss_pyramid で判別。model_stats/benchmark/erf/visualize_sigma は
isinstance(SpectralDW) で自動対応。推奨: `--dw-mode gauss_pyramid --spectral-init-sigma 1
--spectral-max-sigma 8 6 4 3 --gauss-pyramid-growth 1.6 --spectral-sigma-lr 1e-2 --view anterior
--batch-size 16`。焦点=pure-spec は境界悪化だったが **pyramid+pointwise-DoG で境界を保てるか**。

## よく使うコマンド

**ローカル CPU 検証(torch はコンテナ内のみ)**:
```bash
docker run --rm -v <repo>:/workspace -w /workspace -e CUDA_VISIBLE_DEVICES="" \
  uniconvnet-t-app:latest python3 src/smoke_test.py
```
Windows Git Bash では `MSYS_NO_PATHCONV=1` と `$(pwd -W)` を使う。

**サーバー(tsubaki, GPU)学習**(単一GPU厳守 / spectral は FFT で batch16):
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode spectral \
  --spectral-use-local-branch --spectral-init-gamma 0.5 --spectral-sigma-lr 1e-2 \
  --spectral-init-sigma 8 6 4 2 --spectral-alpha 2 --batch-size 16 --tag <name>
python3 src/list_runs.py            # run 一覧
```
詳細レシピ・評価ループは **`docs/EXPERIMENTS.md`** 参照。

**完了通知(Discord/Slack)**: 初回だけ `scripts/.notify_webhook` に Webhook URL を1行書けば、
`src/train.py` は完了/例外時に自動通知(`--no-notify` で抑止)。任意のスクリプト(将来の新規
ファイル含む)は `scripts/notify_run.sh <コマンド>` で包めばそのまま通知される。詳細は
`docs/EXPERIMENTS.md` §0.5。`scripts/.notify_webhook` は .gitignore 済み(秘密は push しない)。

## コード同期(重要な運用ルール)

- **ローカル(Win)で commit → push**、サーバーは **`git fetch origin && git checkout origin/main -- src/`**(src/ だけ取り込み)。
- サーバーで `git checkout main` / `git merge` / `git pull` は**禁止**(手編集 README/makefile と衝突し index が壊れる)。壊れたら `git reset --merge` → `git checkout origin/main -- src/`。
- 作業前に `git fetch`。並行して bonescinti(サーバー)が origin/main に直接 push することがあるので `python -m py_compile src/...` で HEAD 健全性を確認。

## コーディング規約

- **pre-commit で black + isort(profile=black)** が走る。commit 時に自動整形されるので、
  整形で commit がabortしたら **再度 `git add` して commit** する(これが正常フロー)。
- コメント/docstring は既存に倣い日本語。周囲のスタイル(命名・コメント密度)に合わせる。
- 新機構は `weight/bias` を既存 depthwise と同形・同キー(`a*.2.weight/bias`)にして
  事前学習転移を壊さない。新規パラメータ(log_sigma/gamma/gate 等)は strict=False で吸収。

## 単一の真実源(SSOT)

研究の詳細な経緯・数値結果・設計判断は **`docs/report_agd_efficient_dw.md`** と
運用は **`docs/EXPERIMENTS.md`** に集約。CLAUDE.md はそれらへの地図。
