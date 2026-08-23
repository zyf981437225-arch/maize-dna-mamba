#!/usr/bin/env bash
set -euo pipefail

# Portable launcher for the validated 8-GPU RNA-Mamba workflow. Run this
# inside an allocation; scheduler-specific resource requests belong in the
# surrounding Slurm/PBS script.

STAGE="${1:-all}"
case "$STAGE" in
  all|pretrain|finetune|evaluate) ;;
  *)
    echo "Usage: bash scripts/run_formal_8gpu.sh [all|pretrain|finetune|evaluate]" >&2
    exit 2
    ;;
esac

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/formal_8gpu}"
M6A_DATA="${M6A_DATA:-$ROOT_DIR/data/processed/human_m6a_full_transcript}"
RNA_FASTA_FILE="${RNA_FASTA_FILE:-$ROOT_DIR/data/rnacentral_small_ATCG_only.fasta}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_DEVICES="${NUM_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
GLOBAL_BATCH=$((NUM_DEVICES * BATCH_SIZE * GRAD_ACCUM))
NUM_WORKERS="${NUM_WORKERS:-8}"

PRETRAIN_RUN="${PRETRAIN_RUN:-$RUN_ROOT/pretrain}"
FINETUNE_RUN="${FINETUNE_RUN:-$RUN_ROOT/finetune}"
EVAL_DIR="${EVAL_DIR:-$FINETUNE_RUN/calibrated_evaluation}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-$PRETRAIN_RUN/checkpoints_best/val_loss.ckpt}"
FINETUNE_CKPT="${FINETUNE_CKPT:-$FINETUNE_RUN/checkpoints_best/val_m6a_ap.ckpt}"
PRETRAIN_INIT_CKPT="${PRETRAIN_INIT_CKPT:-}"
PRETRAIN_RESUME_CKPT="${PRETRAIN_RESUME_CKPT:-}"

if [[ "$GLOBAL_BATCH" -ne 16 ]]; then
  echo "Expected global batch 16, got $GLOBAL_BATCH:" >&2
  echo "  devices=$NUM_DEVICES batch_size=$BATCH_SIZE accumulation=$GRAD_ACCUM" >&2
  echo "Adjust BATCH_SIZE or GRAD_ACCUM deliberately before launching." >&2
  exit 2
fi

for required in \
  "$M6A_DATA/train.jsonl.gz" \
  "$M6A_DATA/val.jsonl.gz" \
  "$M6A_DATA/test.jsonl.gz" \
  "$M6A_DATA/stats.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing prepared data file: $required" >&2
    exit 2
  fi
done

if [[ "$STAGE" == "all" || "$STAGE" == "pretrain" ]] && [[ ! -f "$RNA_FASTA_FILE" ]]; then
  echo "Missing RNA FASTA: $RNA_FASTA_FILE" >&2
  exit 2
fi

if [[ -n "$PRETRAIN_INIT_CKPT" && -n "$PRETRAIN_RESUME_CKPT" ]]; then
  echo "Set only one of PRETRAIN_INIT_CKPT and PRETRAIN_RESUME_CKPT." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export M6A_FULL_DATA_DIR="$M6A_DATA"
export M6A_NT_DATA_DIR="$M6A_DATA"
export RNA_FASTA_FILE

cd "$ROOT_DIR"
mkdir -p "$RUN_ROOT"

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status=$(git status --porcelain | wc -l) modified/untracked paths"
  echo "stage=$STAGE"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "num_devices=$NUM_DEVICES"
  echo "batch_size_per_gpu=$BATCH_SIZE"
  echo "gradient_accumulation=$GRAD_ACCUM"
  echo "global_batch=$GLOBAL_BATCH"
  echo "m6a_data=$M6A_DATA"
  echo "rna_fasta=$RNA_FASTA_FILE"
  echo "pretrain_init_checkpoint=${PRETRAIN_INIT_CKPT:-none}"
  echo "pretrain_resume_checkpoint=${PRETRAIN_RESUME_CKPT:-none}"
  python --version
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  fi
} > "$RUN_ROOT/run_manifest.txt"

run_pretrain() {
  mkdir -p \
    "$PRETRAIN_RUN/checkpoints_best" \
    "$PRETRAIN_RUN/checkpoints_periodic" \
    "$PRETRAIN_RUN/checkpoints"

  local init_args=(train.pretrained_model_path=null)
  if [[ -n "$PRETRAIN_INIT_CKPT" ]]; then
    if [[ ! -f "$PRETRAIN_INIT_CKPT" ]]; then
      echo "Missing PRETRAIN_INIT_CKPT: $PRETRAIN_INIT_CKPT" >&2
      exit 2
    fi
    init_args=(
      train.pretrained_model_path="$PRETRAIN_INIT_CKPT"
      train.pretrained_model_strict_load=true
    )
  fi

  local resume_args=(train.ckpt=null)
  if [[ -n "$PRETRAIN_RESUME_CKPT" ]]; then
    if [[ ! -f "$PRETRAIN_RESUME_CKPT" ]]; then
      echo "Missing PRETRAIN_RESUME_CKPT: $PRETRAIN_RESUME_CKPT" >&2
      exit 2
    fi
    resume_args=(train.ckpt="$PRETRAIN_RESUME_CKPT")
  fi

  /usr/bin/time -v -o "$PRETRAIN_RUN/time.txt" \
    python -m train \
      experiment=rna_long_pretrain \
      trainer.devices="$NUM_DEVICES" \
      trainer.accelerator=gpu \
      trainer.max_epochs=null \
      trainer.max_steps=50000 \
      trainer.accumulate_grad_batches="$GRAD_ACCUM" \
      dataset.max_sequence_length=10240 \
      dataset.batch_size="$BATCH_SIZE" \
      dataset.batch_size_eval=1 \
      loader.num_workers="$NUM_WORKERS" \
      train.test=false \
      "${init_args[@]}" \
      "${resume_args[@]}" \
      callbacks.model_checkpoint.dirpath="$PRETRAIN_RUN/checkpoints_best" \
      callbacks.model_checkpoint.filename=val_loss \
      callbacks.periodic_checkpoint.dirpath="$PRETRAIN_RUN/checkpoints_periodic" \
      callbacks.model_checkpoint_every_n_steps.dirpath="$PRETRAIN_RUN/checkpoints" \
      callbacks.model_checkpoint_every_n_steps.every_n_train_steps=1000 \
      callbacks.model_checkpoint_every_n_steps.save_last=true \
      wandb=null \
      hydra.run.dir="$PRETRAIN_RUN" \
      2>&1 | tee "$PRETRAIN_RUN/console.log"
}

run_finetune() {
  if [[ ! -f "$PRETRAIN_CKPT" ]]; then
    echo "Missing PRETRAIN_CKPT: $PRETRAIN_CKPT" >&2
    exit 2
  fi

  mkdir -p \
    "$FINETUNE_RUN/checkpoints_best" \
    "$FINETUNE_RUN/checkpoints_periodic" \
    "$FINETUNE_RUN/checkpoints"

  /usr/bin/time -v -o "$FINETUNE_RUN/time.txt" \
    python -m train \
      experiment=human_m6a_full_transcript \
      trainer.devices="$NUM_DEVICES" \
      trainer.accelerator=gpu \
      trainer.max_epochs=1 \
      trainer.accumulate_grad_batches="$GRAD_ACCUM" \
      optimizer.lr=2e-5 \
      train.pretrained_model_path="$PRETRAIN_CKPT" \
      train.pretrained_model_strict_load=false \
      train.pretrained_model_state_hook._name_=load_matching_backbone \
      train.ckpt=null \
      train.test=false \
      dataset.data_dir="$M6A_DATA" \
      dataset.max_sequence_length=10240 \
      dataset.batch_size="$BATCH_SIZE" \
      dataset.batch_size_eval=1 \
      loader.num_workers="$NUM_WORKERS" \
      callbacks.model_checkpoint.dirpath="$FINETUNE_RUN/checkpoints_best" \
      callbacks.model_checkpoint.filename=val_m6a_ap \
      callbacks.periodic_checkpoint.dirpath="$FINETUNE_RUN/checkpoints_periodic" \
      callbacks.model_checkpoint_every_n_steps.dirpath="$FINETUNE_RUN/checkpoints" \
      wandb=null \
      hydra.run.dir="$FINETUNE_RUN" \
      2>&1 | tee "$FINETUNE_RUN/console.log"
}

run_evaluate() {
  if [[ ! -f "$FINETUNE_CKPT" ]]; then
    echo "Missing FINETUNE_CKPT: $FINETUNE_CKPT" >&2
    exit 2
  fi

  mkdir -p "$EVAL_DIR"
  local eval_gpu="${EVAL_GPU:-0}"

  CUDA_VISIBLE_DEVICES="$eval_gpu" /usr/bin/time -v -o "$EVAL_DIR/time.txt" \
    python scripts/evaluate_m6a_full_transcript.py \
      --checkpoint "$FINETUNE_CKPT" \
      --data-dir "$M6A_DATA" \
      --output-dir "$EVAL_DIR" \
      --batch-size 1 \
      --num-workers "$NUM_WORKERS" \
      --device cuda \
      --bins 4096 \
      --threshold-objective f1 \
      2>&1 | tee "$EVAL_DIR/console.log"
}

case "$STAGE" in
  pretrain) run_pretrain ;;
  finetune) run_finetune ;;
  evaluate) run_evaluate ;;
  all)
    run_pretrain
    run_finetune
    run_evaluate
    ;;
esac
