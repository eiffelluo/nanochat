#!/bin/bash

# Train a nanochat GPT-2 grade model on a single NVIDIA RTX 40-series GPU (48GB VRAM).
# Covers the full pipeline: tokenizer training, base pretraining, SFT finetuning, and evaluation.
# Designed for RTX 4090 (48GB) / RTX 6000 Ada (48GB) or similar Ada Lovelace GPUs.
# Expected total runtime: ~24-30 hours (vs ~3 hours on 8×H100).
#
# 1) Simple launch:
#    bash runs/run4090.sh
# 2) Launch in screen (recommended for long runs):
#    screen -L -Logfile runs/run4090.log -S run4090 bash runs/run4090.sh
# 3) With wandb logging:
#    WANDB_RUN=run4090 screen -L -Logfile runs/run4090.log -S run4090 bash runs/run4090.sh
#
# Tuning knobs (environment variables):
#   DEPTH           - model depth, default 26 (GPT-2 level). Reduce to 20 for faster runs.
#   DEVICE_BS       - per-device batch size, default 8. Reduce to 4 or 2 if OOM.
#   WANDB_RUN       - wandb run name, default "dummy" (disabled).
#   SKIP_SETUP      - set to 1 to skip venv/dataset/tokenizer setup.

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

DEPTH="${DEPTH:-26}"
DEVICE_BS="${DEVICE_BS:-8}"

# -----------------------------------------------------------------------------
# Python venv setup with uv

if [ -z "$SKIP_SETUP" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

python -m scripts.base_train -- \
    --depth=$DEPTH \
    --target-param-data-ratio=8.25 \
    --device-batch-size=$DEVICE_BS \
    --run=$WANDB_RUN

python -m scripts.base_eval -- \
    --device-batch-size=$DEVICE_BS

# -----------------------------------------------------------------------------
# SFT (supervised finetuning)

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft -- \
    --device-batch-size=$DEVICE_BS \
    --run=$WANDB_RUN

python -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate report
python -m nanochat.report generate

echo ""
echo "=============================================="
echo "Training complete! You can now chat with your model:"
echo "  CLI:   python -m scripts.chat_cli -p \"Why is the sky blue?\""
echo "  WebUI: python -m scripts.chat_web"
echo "=============================================="
