# 実験運用リファレンス (dw_mode: dense / separable / spectral)

骨シンチ13クラス segmentation。RFA(ConvMod) の a1/a2/a3 depthwise を効率化する
機構A(separable)/機構B(spectral) の学習・評価・フォルダ管理をまとめる。
**run 一覧はいつでも `python3 src/list_runs.py` で自動生成**（手作業の台帳不要）。

---

## 0. TL;DR

```bash
# サーバーはコード同期 → 学習 → 一覧確認 → 評価
git fetch origin && git checkout origin/main -- src/     # main から src/ だけ取り込む
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode <mode> [...] --tag <name>
python3 src/list_runs.py                                 # 全 run を dice 降順で一覧
python3 src/list_runs.py --filter spectral               # spectral 系だけ
```

- **spectral(FFT)系は必ず単一GPU**（`CUDA_VISIBLE_DEVICES=0`）。DataParallel で崩壊する既知バグ。
- **spectral は `--batch-size 16`**（FFT でメモリを食う。大 batch は DCNv3 forward で OOM）。
- ローカル(Win)は commit→push、サーバーは `git checkout origin/main -- src/`。**サーバーで `git merge`/`git checkout main` は禁止**（手編集 README/makefile と衝突する）。

---

## 0.5 完了通知 (Discord/Slack Webhook)

学習が終わった/落ちたら Webhook に通知が届く。**初回だけ URL を1回設定**すれば以後ずっと有効。

```bash
# 初回のみ: Webhook URL を1行書く(.gitignore 済み → push されない)。
cp scripts/.notify_webhook.example scripts/.notify_webhook
echo 'https://discord.com/api/webhooks/XXXX/YYYY' > scripts/.notify_webhook
```

- **train.py は自前で通知**する(tag/機構/best dice/保存先を本文に)。上の TL;DR の
  `python3 src/train.py ...` をそのまま回すだけで完了/例外時に届く。`--no-notify` で抑止。
- **どの実験スクリプトでも(将来追加する新規ファイルでも)通知させたい**場合は、コマンドを
  汎用ラッパ `scripts/notify_run.sh` で包むだけ。スクリプト側の改修は不要:
  ```bash
  scripts/notify_run.sh python3 src/train.py --dw-mode spectral_mix --tag mix_v1
  scripts/notify_run.sh make train-inner        # make ターゲットでも可
  scripts/notify_run.sh python3 src/新しい実験.py ...   # 新規ファイルもそのまま
  ```
  ラッパは終了コード・所要時間・GPU・ログ末尾15行・最新 run フォルダを載せ、コマンドの
  終了コードをそのまま返す。ラッパ経由時は `NOTIFY_WRAPPED=1` が立ち train.py 内蔵通知は
  黙る(二重送信しない)。
- 通知は**ベストエフォート**: Webhook 未設定・送信失敗でも学習は落ちない(標準ライブラリのみ、
  requests 不要)。コンテナから discord.com への外向き HTTPS が塞がれている環境では、ホスト側で
  `scripts/notify_run.sh docker compose exec app python3 src/train.py ...` のように**外側を包む**。

---

## 1. 機構 (dw_mode) と train レシピ

| dw_mode | 機構 | 何をするか |
|---|---|---|
| `dense` (既定) | DilatedDWConv(dilation=1) | 素の密 depthwise。事前学習そのまま転移 |
| `separable` | 機構A: 1×K・K×1 分離 | K²→2K に削減。密カーネルを rank-1 SVD で分離初期化 |
| `spectral` | 機構B: 周波数ガウス+動的切り出し | per-channel σ の低域通過。σ大で帯域を η=α/σ に切り詰め |

### レシピ（コピペ可・すべて単一GPU）

```bash
# ① baseline (dense, 対照)
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode dense --tag baseline

# ② separable (機構A, SVD warm-start)。FFT無しなので batch 32 可
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode separable --batch-size 32 --tag separable

# ③ spectral pure (機構B, local枝オフ)。純ガウス低域通過。σは凍結する点に注意
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode spectral \
  --spectral-init-sigma 8 6 4 2 --spectral-alpha 2 --batch-size 16 --tag spec_pure

# ④ spectral 学習σ (推奨: 鋭いlocal + gamma>0 + σ高lr で σ が実学習)
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --dw-mode spectral \
  --spectral-use-local-branch --spectral-init-gamma 0.5 --spectral-sigma-lr 1e-2 \
  --spectral-init-sigma 8 6 4 2 --spectral-alpha 2 --batch-size 16 --tag spec_learnsig
```

### spectral の主なノブ

| フラグ | 意味 | 効き所 |
|---|---|---|
| `--spectral-init-sigma` (stage別) | σ 初期値 (feature px) | RF幅。狙い値を直接 init |
| `--spectral-sigma-lr` | σ専用lr (既定0=凍結) | **1e-2 でσが実学習**(WD0で分離) |
| `--spectral-init-gamma` | spectral枝ゲート初期 | 0=identity(σ勾配ゼロ), >0でσ学習可 |
| `--spectral-use-local-branch` | 鋭いlocal枝併用 | 境界維持・AGD回復 |
| `--spectral-alpha` | 切り出し比 η=α/σ | 2で無損失, 小で安いがリンギング |
| `--spectral-max-sigma` (stage別) | σ上限 | 既定 32/24/12/6 |

**σ が動く条件**(重要): ①σが勾配を受ける経路=`--spectral-use-local-branch`+`init-gamma>0` か pure-spec、
かつ ②`--spectral-sigma-lr>0`。gamma=0 の identity init は spec をゲートしσ勾配=0。
学習後は **必ず `visualize_sigma.py` で σ が動いたか確認**。

---

## 2. フォルダ & 命名

```
experiments/run_<timestamp>[_<tag>]/
  ├── best_uniconvnet_unet.pth   # ベスト重み (評価はこれを指す)
  ├── run_config.json            # 機械可読な学習条件 (args 全部)。評価系が dw_mode 等を自動判別
  └── run_info.txt               # 人間可読サマリ (best_bone_dice, epochs 等)
```

- **`--tag` を必ず付ける**と自己説明的なフォルダ名になる（例 `run_..._spec_learnsig`）。
- 評価系スクリプトは `best_uniconvnet_unet.pth` と同じフォルダの `run_config.json` から
  dw_mode / use_local_branch / alpha を読む → **ckpt は run フォルダごと扱う**(単体コピー時は run_config.json も一緒に)。

結果フォルダ(gitignore):
```
bench_results/  compare_results/  erf_results/  pred_results/  sigma_results/
```

### run 一覧 (台帳の代わり)
```bash
python3 src/list_runs.py                 # dice 降順
python3 src/list_runs.py --sort date     # 日付順
python3 src/list_runs.py --filter spectral --csv runs.csv
```

---

## 3. 評価スクリプト

| スクリプト | 何を出すか | 代表コマンド |
|---|---|---|
| `benchmark.py` | params/FLOPs/**img-s**/RFA省コスト比 | `--weights $A $B --labels a b --input-size 256` |
| `compare_models.py` | 全指標×worst10/25%×Wilcoxon (境界hd95/assd/nsd含む) | `--weights $A $B --worst-fracs 0.1 0.25` |
| `evaluate.py` | 1モデルの部位別/worst 全指標表 | `--weights $A` |
| `erf_sigma_table.py` | 層別 ERF σ_moment & **excess kurtosis(AGD)** | `--weights $A $B --input-size 512 --n-samples 50` |
| `visualize_sigma.py` | **σが凍結/学習したか** (stage別分布+\|Δlog\|) | `--weights $A $B --labels frozen learned` |
| `visualize_predictions.py` | 推定マスク (Worst/Avg/Best, `--save-all`で全件) | `--weights $A` |
| `visualize_erf.py` | 1層のERF可視化(FFT断面+ガウス包絡) | `--weights $A --part encoder` |
| `agd_probe.py` | 単層カーネルの AGD 比較(dilated/separable/spectral, 学習不要) | `--sigma 6 --dilation 4` |
| `list_runs.py` | run 一覧表 | `--sort dice` |
| `smoke_test.py` | CPU構造テスト(全dw_mode) | (引数なし) |

すべて `build_model_from_checkpoint` 経由で **dw_mode を自動判別**するので、追加フラグ不要。

---

## 4. 標準の検証ループ (機構を1本回したら)

```bash
BASE=experiments/run_..._baseline/best_uniconvnet_unet.pth
NEW=experiments/run_..._<tag>/best_uniconvnet_unet.pth

# (spectralなら) σ が動いたか
python3 src/visualize_sigma.py --weights $BASE $NEW --labels base new --out-dir sigma_results/<tag>
# コスト (img/s が本当に下がったか)
python3 src/benchmark.py       --weights $BASE $NEW --labels base new --input-size 256 --out-dir bench_results/<tag>
# 精度×worst×境界
python3 src/compare_models.py  --weights $BASE $NEW --labels base new --worst-fracs 0.1 0.25 --out-dir compare_results/<tag>
# AGD (層別 kurtosis)
python3 src/erf_sigma_table.py --weights $BASE $NEW --labels base new --input-size 512 --n-samples 50 --out-dir erf_results/<tag>
```

判断の軸: **①img/s(実速度) が真実**(RFA比x○はMAC上の理論値)、②worst×境界(hd95/assd)で劣化がないか、
③erf kurtosis で AGD が保たれているか、④visualize_sigma で σ が学習されているか。

---

## 5. サーバー運用 (tsubaki, rootless Docker)

```bash
# 起動 (--user は付けない)
docker run --rm -it --gpus '"device=0"' --security-opt seccomp=unconfined --shm-size=2g \
  -v $(pwd):/workspace \
  -v /autohome/bonescinti/nao/scinti_segmentation:/workspace/scinti_segmentation:ro \
  bonescinti/uniconvnet-t:cuda121-torch231-v1 bash

# コード同期 (src/ だけ。README/makefile 等の手編集を守る)
git fetch origin && git checkout origin/main -- src/
# HEAD が壊れていないか確認してから学習
python3 -m py_compile src/models/UniConvNet.py src/train.py
```

- **空きGPU確認**: `nvidia-smi` (GPU 0-2 が空きがち)。埋まっていたら `--gpus '"device=N"'` で別番号。
- **spectral は単一GPU厳守**(train.py が device_count>1 かつ FFT機構なら DataParallel を自動回避)。
- 詳細な罠は運用メモ(tsubaki-docker-ops)参照。

---

## 6. これまでの主要 run (スナップショット, 正確な一覧は `list_runs.py`)

| tag | dw_mode / config | best Dice | 要点 |
|---|---|---|---|
| baseline | dense | ~0.941 | 対照 |
| spec_pure | spectral σ[8,6,4,2] α2 local=False | ~0.9406 | 精度中立だが遅い/AGD崩壊/σ凍結 |
| spec_learnsig | spectral local γ0.5 σlr1e-2 | ~0.9410 | **σが実学習(92-95%動く/per-ch分化)**, Dice最良 |
| separable | separable (rank-1 SVD) | ~0.9346 | 速い/AGD保持だが境界(assd)有意悪化 |

結論の現状: 「効率+AGD+精度」を同時達成する機構はまだ無い。separable=速いが境界劣化(rank-1)、
spectral=中立だが遅い/AGD崩壊(pure時)。**σ高lr で spectral の σ 学習は解決**、次は学習σ版の
AGD/境界/cost を評価して pure-spec の弱点が直ったかを見る段階。
