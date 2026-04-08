#!/bin/bash

# Train a small nanochat model on Mac Apple Silicon using MPS acceleration.
# Assumes you have already set up a conda environment with dependencies installed:
#   conda create -n nanochat python=3.10 -y
#   conda activate nanochat
#   pip install -e .
#
# Run as:
#   conda activate nanochat
#   bash runs/runmps.sh
#
# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.

set -e

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# ---- Tokenizer ----

python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# ---- Base model (pretraining) ----

python -m scripts.base_train \
    --device-type=mps \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN

python -m scripts.base_eval \
    --device-type=mps \
    --device-batch-size=1 \
    --split-tokens=16384 \
    --max-per-task=16

# ---- SFT ----

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

python -m scripts.chat_sft \
    --device-type=mps \
    --max-seq-len=512 \
    --device-batch-size=8 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# ---- Chat ----
# Uncomment one of the following to talk to your model:

# CLI:
# python -m scripts.chat_cli --device-type=mps -p "What is the capital of France?"

# WebUI:
# python -m scripts.chat_web --device-type=mps
