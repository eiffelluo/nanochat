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

python -m scripts.chat_sft \
    --device-type=mps \
    --max-seq-len=512 \
    --device-batch-size=16 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN