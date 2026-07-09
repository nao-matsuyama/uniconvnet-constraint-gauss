# ラボ運用ガイドライン準拠
#   ・PyTorch 公式イメージをベース (torch 同梱、ビルド高速・サイズ小)
#   ・最小限の依存のみ追加 / データ・重みはイメージに含めない (マウントする)
#   ・RTX 8000 (Turing sm_75) / V100 / P100 すべて cuda12.1 で動作
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 共有サーバーのスレッド上限で pip の進捗バー(rich)が落ちるのを防ぐ
ENV PIP_PROGRESS_BAR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_INPUT=1

# 共有 Docker の pids 制限下で OpenBLAS/OMP が大量スレッドを起動して
# "Operation not permitted" になるのを防ぐ
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# --user 実行 (UID/GID 指定) で HOME が無い場合に matplotlib 等が
# キャッシュを書けず落ちるのを防ぐ
ENV HOME=/tmp
ENV MPLCONFIGDIR=/tmp/mpl

# NFS 上の共有 Docker では docker-clean の apt フックが失敗するため無効化
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 依存ライブラリ (torch はベースイメージに同梱済み)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
