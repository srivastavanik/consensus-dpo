#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-training/configs/dpo.default.yaml}

MODEL_NAME=$(yq -r .model_name "$CONFIG")
BETA=$(yq -r .beta "$CONFIG")
LR=$(yq -r .learning_rate "$CONFIG")
WD=$(yq -r .weight_decay "$CONFIG")
MAXLEN=$(yq -r .max_seq_len "$CONFIG")
EPOCHS=$(yq -r .epochs "$CONFIG")
BSZ=$(yq -r .per_device_batch "$CONFIG")
ACC=$(yq -r .grad_accum_steps "$CONFIG")
OUTDIR=$(yq -r .output_dir "$CONFIG")
PAIRS=$(yq -r .pairs_path "$CONFIG")

export DPO_PAIRS_PATH="$PAIRS"
export STUDENT_MODEL="$MODEL_NAME"

python apps/trainer/train_dpo.py \
  --beta "$BETA" --lr "$LR" --wd "$WD" --max-len "$MAXLEN" \
  --epochs "$EPOCHS" --bsz "$BSZ" --acc "$ACC" --outdir "$OUTDIR"


