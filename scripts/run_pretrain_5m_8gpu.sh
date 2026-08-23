#!/usr/bin/env bash
set -euo pipefail

# Formal pretraining-only launcher for the prepared 3M ncRNA + ~2M coding-RNA
# corpus. Run this script inside an allocated 8-GPU node.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_DIR="${RNA_PRETRAIN_INDEXED_DIR:-$ROOT_DIR/data/processed/rna_pretraining_5m}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/runs/rna_pretraining_5m}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_DEVICES="${NUM_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2}"
WARMUP_STEPS="${WARMUP_STEPS:-20000}"
RESUME_CKPT="${RESUME_CKPT:-}"
MIN_CODING_RECORDS="${MIN_CODING_RECORDS:-1500000}"

MANIFEST="$DATA_DIR/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing prepared-corpus manifest: $MANIFEST" >&2
  echo "Run scripts/prepare_pretraining_5m.py first." >&2
  exit 2
fi
if [[ "$NUM_DEVICES" -le 0 || "$BATCH_SIZE" -le 0 || "$GRAD_ACCUM" -le 0 ]]; then
  echo "NUM_DEVICES, BATCH_SIZE and GRAD_ACCUM must be positive." >&2
  exit 2
fi
if [[ "$PRETRAIN_EPOCHS" -le 0 ]]; then
  echo "PRETRAIN_EPOCHS must be positive." >&2
  exit 2
fi

read -r TRAIN_RECORDS TOTAL_RECORDS NCRNA_RECORDS CODING_RECORDS < <(
  python - "$MANIFEST" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    manifest["splits"]["train"]["records"],
    manifest["totals"]["records"],
    manifest["totals"]["source_class_counts"]["ncRNA"],
    manifest["totals"]["source_class_counts"]["coding"],
)
PY
)

if [[ "$NCRNA_RECORDS" -ne 3000000 || "$CODING_RECORDS" -lt "$MIN_CODING_RECORDS" || "$CODING_RECORDS" -gt 2000000 ]]; then
  echo "Refusing formal run: expected 3M ncRNA + approximately 2M independent coding records." >&2
  echo "Allowed coding range: $MIN_CODING_RECORDS..2000000" >&2
  echo "Manifest reports ncRNA=$NCRNA_RECORDS coding=$CODING_RECORDS" >&2
  exit 2
fi

GLOBAL_BATCH=$((NUM_DEVICES * BATCH_SIZE * GRAD_ACCUM))
STEPS_PER_EPOCH=$(((TRAIN_RECORDS + GLOBAL_BATCH - 1) / GLOBAL_BATCH))
MAX_STEPS=$((STEPS_PER_EPOCH * PRETRAIN_EPOCHS))
if [[ "$WARMUP_STEPS" -ge "$MAX_STEPS" ]]; then
  WARMUP_STEPS=$((MAX_STEPS / 10))
fi

if [[ -n "$RESUME_CKPT" && ! -f "$RESUME_CKPT" ]]; then
  echo "Missing RESUME_CKPT: $RESUME_CKPT" >&2
  exit 2
fi

mkdir -p \
  "$RUN_DIR/checkpoints_best" \
  "$RUN_DIR/checkpoints_periodic" \
  "$RUN_DIR/checkpoints"

export CUDA_VISIBLE_DEVICES
export RNA_PRETRAIN_INDEXED_DIR="$DATA_DIR"
cd "$ROOT_DIR"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status_paths=$(git status --porcelain | wc -l)"
  echo "data_dir=$DATA_DIR"
  echo "total_records=$TOTAL_RECORDS"
  echo "train_records=$TRAIN_RECORDS"
  echo "ncrna_records=$NCRNA_RECORDS"
  echo "coding_records=$CODING_RECORDS"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "num_devices=$NUM_DEVICES"
  echo "batch_size_per_gpu=$BATCH_SIZE"
  echo "gradient_accumulation=$GRAD_ACCUM"
  echo "global_batch=$GLOBAL_BATCH"
  echo "target_epochs=$PRETRAIN_EPOCHS"
  echo "steps_per_epoch=$STEPS_PER_EPOCH"
  echo "max_steps=$MAX_STEPS"
  echo "warmup_steps=$WARMUP_STEPS"
  echo "resume_checkpoint=${RESUME_CKPT:-none}"
  python --version
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  fi
} > "$RUN_DIR/run_manifest.txt"

resume_args=(train.ckpt=null)
if [[ -n "$RESUME_CKPT" ]]; then
  resume_args=(train.ckpt="$RESUME_CKPT")
fi

echo "Starting formal RNA-Mamba pretraining:"
echo "  records: $TOTAL_RECORDS (train=$TRAIN_RECORDS)"
echo "  global batch: $GLOBAL_BATCH"
echo "  target: $PRETRAIN_EPOCHS epochs = $MAX_STEPS optimizer steps"
echo "  output: $RUN_DIR"

/usr/bin/time -v -o "$RUN_DIR/time.txt" \
  python -m train \
    experiment=rna_5m_pretrain \
    trainer.devices="$NUM_DEVICES" \
    trainer.accelerator=gpu \
    trainer.max_epochs=null \
    trainer.max_steps="$MAX_STEPS" \
    trainer.accumulate_grad_batches="$GRAD_ACCUM" \
    dataset.batch_size="$BATCH_SIZE" \
    dataset.batch_size_eval=1 \
    loader.num_workers="$NUM_WORKERS" \
    scheduler.t_initial="$MAX_STEPS" \
    scheduler.warmup_t="$WARMUP_STEPS" \
    train.pretrained_model_path=null \
    "${resume_args[@]}" \
    train.test=false \
    callbacks.model_checkpoint.dirpath="$RUN_DIR/checkpoints_best" \
    callbacks.model_checkpoint.filename=val_loss \
    callbacks.periodic_checkpoint.dirpath="$RUN_DIR/checkpoints_periodic" \
    callbacks.model_checkpoint_every_n_steps.dirpath="$RUN_DIR/checkpoints" \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps="$STEPS_PER_EPOCH" \
    callbacks.model_checkpoint_every_n_steps.save_last=true \
    wandb=null \
    hydra.run.dir="$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/console.log"
