# ==============================================================
# UniConvNet-constraint-gauss 実験パイプライン
# (本命: ガウス微分基底 gauss_deriv で RF を σ のみに縛る制約機構)
#
# 【ローカル CPU 検証 — ホストから直接 docker run (GPU/compose 不要)】
#   make smoke             構造スモークテスト (全 dw_mode)
#   make agd               単層 AGD プローブ (SIGMA=6 DILATION=4)
#   make verify-gd         gauss_deriv の転移/eval往復/captured energy 検証
#   make check             上記3つを一括
#
# 【サーバー学習 — コンテナ内 (make shell 後) で実行】
#   make train-gd-inner        ★本命 gauss_deriv 学習
#                              (ORDER/INIT_SIGMA/MAX_SIGMA/SIGMA_LR/TAG で調整)
#   make train-baseline-inner  比較用 dense baseline (同 seed・予算)
#   make train-inner           従来 dense 学習
#   make train-erf-inner       ERF 正則化つき学習
#   make ls-runs               実験ランの一覧
#   make eval-inner            全指標評価 (worst×境界)
#   make stats-inner           パラメータ数 / FLOPs
#   make pipeline-inner        評価+FLOPs+ERF+推論可視化を一括
#
# 【Docker 操作 — ホストから】
#   make build / up / down / shell / gpu
#
# 【WEIGHTS の指定方法】
#   WEIGHTS 未指定時は最新の実験ランを自動使用。
#   明示的に指定する場合:
#     make pipeline-inner WEIGHTS=/workspace/experiments/run_XXXXXX/best_uniconvnet_unet.pth
# ==============================================================

COMPOSE     = docker compose
SERVICE     = app
SRC         = /workspace/src

DATA_DIR    = /workspace/scinti_segmentation
PRETRAINED  = /workspace/uniconvnet_t_1k_224_ema.pth
BATCH_SIZE  = 8
NUM_WORKERS = 4
MAX_EPOCHS  = 50
LR          = 1e-4

# 一括評価で使う GPU 番号 (CUDA_VISIBLE_DEVICES)
GPU         ?= 0

# ERF 正則化 (学習可能 dilation を目標 RF 広がりへ寄せる)
ERF_W       = 0.01
# stage0..3 別の目標 RF 広がり (低層=小/高層=大)。1値なら全stage共通
ERF_TARGET  = 2 3 4 6

# ── gauss_deriv (本命: ガウス微分基底) のパラメータ。make train-gd ORDER=1 等で上書き可 ──
ORDER       ?= 2
INIT_SIGMA  ?= 1.5
MAX_SIGMA   ?= 8 6 4 3
SIGMA_LR    ?= 1e-2
GD_BATCH    ?= 16
TAG         ?= gaussderiv_n$(ORDER)

# ── ローカル CPU 検証用 (compose を使わず直接 docker run。GPU 不要のワンショット) ──
IMAGE       ?= uniconvnet-t-app:latest
REPO        ?= $(CURDIR)
# MSYS_NO_PATHCONV=1: Windows Git Bash で "-v C:/...:/workspace" の C:/ が
# MSYS のパス変換で壊れるのを防ぐ (Linux サーバーでは未使用の env で無害)。
DOCKER_CPU   = MSYS_NO_PATHCONV=1 docker run --rm -e CUDA_VISIBLE_DEVICES="" \
	-v "$(REPO):/workspace" -w /workspace $(IMAGE)

# WEIGHTS 未指定時は /workspace/experiments の最新ランを自動検出
_LATEST    := $(shell ls /workspace/experiments/ 2>/dev/null | grep "^run_" | sort -r | head -1)
WEIGHTS    ?= $(if $(_LATEST),/workspace/experiments/$(_LATEST)/best_uniconvnet_unet.pth,)

# WEIGHTS のパスからランのフォルダ名を抽出 (例: run_20260626_112936)
RUN_NAME    = $(notdir $(patsubst %/,%,$(dir $(WEIGHTS))))

# WEIGHTS が未指定・存在しない場合に利用可能なランを表示してエラー終了するガード
define WEIGHTS_CHECK
	@[ -n "$(WEIGHTS)" ] || { \
		printf "\n[ERROR] WEIGHTS が未指定です\n\n"; \
		printf "使い方:\n  make $$@ WEIGHTS=/workspace/experiments/<run>/best_uniconvnet_unet.pth\n\n"; \
		printf "利用可能なラン:\n"; \
		ls /workspace/experiments/ 2>/dev/null | grep "^run_" | sort -r | \
			sed 's|^|  /workspace/experiments/|; s|$$|/best_uniconvnet_unet.pth|'; \
		printf "\n"; exit 1; }
	@[ -f "$(WEIGHTS)" ] || { \
		printf "\n[ERROR] ファイルが見つかりません: $(WEIGHTS)\n\n"; \
		printf "利用可能なラン:\n"; \
		ls /workspace/experiments/ 2>/dev/null | grep "^run_" | sort -r | \
			sed 's|^|  /workspace/experiments/|; s|$$|/best_uniconvnet_unet.pth|'; \
		printf "\n"; exit 1; }
endef

.PHONY: build up down gpu train train-freeze train-inner train-freeze-inner \
        train-gd-inner train-baseline-inner train-erf-inner smoke-inner \
        shell ls-runs eval-inner stats-inner eval-all-inner \
        erf-layers-inner erf-output-inner vis-pred-inner pipeline-inner \
        smoke agd verify-gd check git-push-init

# ──────────────────────────────────────────
# Docker 操作 (ホストから)
# ──────────────────────────────────────────

## Docker イメージをビルドする
build:
	$(COMPOSE) build

## コンテナをバックグラウンドで起動する
up:
	$(COMPOSE) up -d

## コンテナを停止・削除する
down:
	$(COMPOSE) down

## GPU 認識確認
gpu: up
	$(COMPOSE) exec $(SERVICE) python3 -c \
		"import torch; print('CUDA:', torch.cuda.is_available()); \
		 [print(' -', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"

## コンテナ内に入る (bash)
shell: up
	$(COMPOSE) exec $(SERVICE) bash

# ──────────────────────────────────────────
# 学習 (ホストから起動)
# ──────────────────────────────────────────

## 学習を実行する（ホストから）
train: up
	$(COMPOSE) exec  $(SERVICE) python3 $(SRC)/train.py \
		--data-dir   $(DATA_DIR) \
		--pretrained  $(PRETRAINED) \
		--batch-size  $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--max-epochs  $(MAX_EPOCHS) \
		--lr          $(LR)

## 学習（バックボーン固定・デコーダーのみ更新）ホストから
train-freeze: up
	$(COMPOSE) exec $(SERVICE) python3 $(SRC)/train.py \
		--data-dir      $(DATA_DIR) \
		--pretrained    $(PRETRAINED) \
		--batch-size    $(BATCH_SIZE) \
		--num-workers   $(NUM_WORKERS) \
		--max-epochs    $(MAX_EPOCHS) \
		--lr            $(LR) \
		--freeze-backbone

# ──────────────────────────────────────────
# 学習 (コンテナ内から)
# ──────────────────────────────────────────

## 学習を実行する（コンテナ内から）
train-inner:
	python3 $(SRC)/train.py \
		--data-dir    $(DATA_DIR) \
		--pretrained  $(PRETRAINED) \
		--batch-size  $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--max-epochs  $(MAX_EPOCHS) \
		--lr          $(LR)

## 学習（バックボーン固定）コンテナ内から
train-freeze-inner:
	python3 $(SRC)/train.py \
		--data-dir      $(DATA_DIR) \
		--pretrained    $(PRETRAINED) \
		--batch-size    $(BATCH_SIZE) \
		--num-workers   $(NUM_WORKERS) \
		--max-epochs    $(MAX_EPOCHS) \
		--lr            $(LR) \
		--freeze-backbone

## 学習（ERF 正則化 ON・学習可能 dilation を目標 RF へ寄せる）コンテナ内から
##   例: make train-erf-inner ERF_W=0.01 ERF_TARGET="2 3 4 6"
train-erf-inner:
	python3 $(SRC)/train.py \
		--data-dir          $(DATA_DIR) \
		--pretrained        $(PRETRAINED) \
		--batch-size        $(BATCH_SIZE) \
		--num-workers       $(NUM_WORKERS) \
		--max-epochs        $(MAX_EPOCHS) \
		--lr                $(LR) \
		--erf-reg-weight    $(ERF_W) \
		--erf-target-spread $(ERF_TARGET)

## 本命 gauss_deriv 学習（コンテナ内から / 単一GPU）
##   例: make train-gd-inner ORDER=2 INIT_SIGMA=1.5 MAX_SIGMA="8 6 4 3" TAG=gaussderiv_n2
train-gd-inner:
	CUDA_VISIBLE_DEVICES=$(GPU) python3 $(SRC)/train.py \
		--data-dir        $(DATA_DIR) \
		--pretrained      $(PRETRAINED) \
		--batch-size      $(GD_BATCH) \
		--num-workers     $(NUM_WORKERS) \
		--max-epochs      $(MAX_EPOCHS) \
		--lr              $(LR) \
		--dw-mode         gauss_deriv \
		--gauss-deriv-order $(ORDER) \
		--spectral-init-sigma $(INIT_SIGMA) \
		--spectral-max-sigma  $(MAX_SIGMA) \
		--spectral-sigma-lr   $(SIGMA_LR) \
		--tag             $(TAG)

## 比較用 baseline (dense) 学習（コンテナ内から / gauss_deriv と同 seed・予算で対照）
train-baseline-inner:
	CUDA_VISIBLE_DEVICES=$(GPU) python3 $(SRC)/train.py \
		--data-dir    $(DATA_DIR) \
		--pretrained  $(PRETRAINED) \
		--batch-size  $(GD_BATCH) \
		--num-workers $(NUM_WORKERS) \
		--max-epochs  $(MAX_EPOCHS) \
		--lr          $(LR) \
		--dw-mode     dense \
		--tag         baseline

## ローカル CPU スモークテスト（torch があれば GPU 不要で構造確認）
smoke-inner:
	python3 $(SRC)/smoke_test.py

## 一括評価（複数モデルを σ/worst-Dice/コストでまとめて評価）コンテナ内から
##   モデルは scripts/eval_all.sh の MODELS を編集。GPU=1 FULL=1 等で上書き可
eval-all-inner:
	GPU=$(GPU) bash scripts/eval_all.sh

# ──────────────────────────────────────────
# 評価・可視化 (コンテナ内から)
# ──────────────────────────────────────────

## 実験ランの一覧を表示する
ls-runs:
	@printf "\n利用可能な実験ラン:\n"
	@ls /workspace/experiments/ 2>/dev/null | grep "^run_" | sort -r | \
		sed 's|^|  /workspace/experiments/|; s|$$|/best_uniconvnet_unet.pth|'
	@printf "\n使い方: make pipeline-inner WEIGHTS=<上記のパス>\n\n"

## Dice スコア評価（コンテナ内から）
eval-inner:
	$(WEIGHTS_CHECK)
	@printf "\n[eval] WEIGHTS = $(WEIGHTS)\n"
	python3 $(SRC)/evaluate.py \
		--weights     $(WEIGHTS) \
		--data-dir    $(DATA_DIR) \
		--batch-size  16 \
		--num-workers 2

## パラメータ数 & FLOPs（コンテナ内から）
stats-inner:
	$(WEIGHTS_CHECK)
	@printf "\n[stats] WEIGHTS = $(WEIGHTS)\n"
	python3 $(SRC)/model_stats.py \
		--weights     $(WEIGHTS) \
		--input-size  256

## ERF 可視化・全層（エンコーダー4層 + デコーダー4層）コンテナ内から
erf-layers-inner:
	$(WEIGHTS_CHECK)
	@printf "\n[erf-layers] RUN = $(RUN_NAME)\n"
	python3 $(SRC)/visualize_erf.py \
		--weights    $(WEIGHTS) \
		--part       all \
		--input-size 512 \
		--n-samples  100 \
		--out-dir    /workspace/erf_results/$(RUN_NAME)

## ERF 可視化・U-Net 最終出力（コンテナ内から）
erf-output-inner:
	$(WEIGHTS_CHECK)
	@printf "\n[erf-output] RUN = $(RUN_NAME)\n"
	python3 $(SRC)/visualize_receptive_field.py \
		--weights    $(WEIGHTS) \
		--input-size 256 \
		--n-samples  100 \
		--out-dir    /workspace/erf_results/$(RUN_NAME)/output

## 推論結果の可視化（ベスト・平均・ワースト）コンテナ内から
vis-pred-inner:
	$(WEIGHTS_CHECK)
	@printf "\n[vis-pred] RUN = $(RUN_NAME)\n"
	python3 $(SRC)/visualize_predictions.py \
		--weights   $(WEIGHTS) \
		--data-dir  $(DATA_DIR) \
		--out-dir   /workspace/pred_results/$(RUN_NAME)

# ──────────────────────────────────────────
# パイプライン（評価・可視化を一括実行）
# ──────────────────────────────────────────

## 評価 + FLOPs + ERF + 推論可視化 を一括実行（コンテナ内から）
## 使い方: make pipeline-inner WEIGHTS=/workspace/experiments/<run>/best_uniconvnet_unet.pth
pipeline-inner: eval-inner stats-inner erf-layers-inner erf-output-inner vis-pred-inner
	@printf "\n========================================\n"
	@printf "  パイプライン完了: $(RUN_NAME)\n"
	@printf "  結果フォルダ:\n"
	@printf "    /workspace/erf_results/$(RUN_NAME)/\n"
	@printf "    /workspace/pred_results/$(RUN_NAME)/\n"
	@printf "========================================\n\n"

# ──────────────────────────────────────────
# ローカル CPU 検証 (ホストから直接 docker run。GPU/compose 不要のワンショット)
#   IMAGE=... で別イメージ、SIGMA=... 等でプローブ引数を上書き可
# ──────────────────────────────────────────

## 構造スモークテスト (全 dw_mode / gauss_deriv 含む)
smoke:
	$(DOCKER_CPU) python3 $(SRC)/smoke_test.py

## 単層 AGD プローブ (dense/dilated/separable/spectral/gauss-deriv を比較)
##   例: make agd SIGMA=6 DILATION=4
SIGMA    ?= 6
DILATION ?= 4
agd:
	$(DOCKER_CPU) python3 $(SRC)/agd_probe.py --sigma $(SIGMA) --dilation $(DILATION) --canvas 96

## gauss_deriv の 転移/eval往復/captured energy を一括検証
verify-gd:
	$(DOCKER_CPU) python3 $(SRC)/verify_gauss_deriv.py

## ローカル検証を全部 (smoke → agd → verify-gd)
check: smoke agd verify-gd
	@printf "\n✅ ローカル CPU 検証 完了\n"

# ──────────────────────────────────────────
# Git リモート接続 (新規リポジトリへの初回 push を一発で)
# ──────────────────────────────────────────

## 新規リモートに origin を繋いで push する。空リポジトリを作ってから URL を渡す。
##   例: make git-push-init URL=git@github.com:<you>/uniconvnet-constraint-gauss.git
git-push-init:
	@[ -n "$(URL)" ] || { printf "\n[ERROR] URL=<repo-url> を指定してください\n  例: make git-push-init URL=git@github.com:you/uniconvnet-constraint-gauss.git\n\n"; exit 1; }
	git remote add origin "$(URL)" 2>/dev/null || git remote set-url origin "$(URL)"
	git push -u origin main
