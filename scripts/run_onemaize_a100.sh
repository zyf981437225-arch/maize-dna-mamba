#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-smoke}"
case "$STAGE" in
  validate|smoke|8k|16k) ;;
  *)
    echo "Usage: bash scripts/run_onemaize_a100.sh [validate|smoke|8k|16k]" >&2
    exit 2
    ;;
esac

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ONEMAIZE_DATA_DIR="${ONEMAIZE_DATA_DIR:-$ROOT_DIR/data/processed/onemaize_b73}"
ONEMAIZE_FASTA_ROOT="${ONEMAIZE_FASTA_ROOT:-}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/onemaize}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
PHASE1_CKPT="${PHASE1_CKPT:-}"

if [[ ! -f "$ONEMAIZE_DATA_DIR/manifest.json" ]]; then
  echo "Missing OneMaize manifest: $ONEMAIZE_DATA_DIR/manifest.json" >&2
  exit 2
fi

export ONEMAIZE_DATA_DIR
cd "$ROOT_DIR"
mkdir -p "$RUN_ROOT/$STAGE"

fasta_args=()
hydra_fasta_args=()
if [[ -n "$ONEMAIZE_FASTA_ROOT" ]]; then
  fasta_args=(--fasta-root "$ONEMAIZE_FASTA_ROOT")
  hydra_fasta_args=(dataset.fasta_root="$ONEMAIZE_FASTA_ROOT")
fi

context=8192
if [[ "$STAGE" == "16k" ]]; then
  context=16384
fi

python scripts/validate_onemaize_data.py \
  --data-dir "$ONEMAIZE_DATA_DIR" \
  --context-length "$context" \
  --sampling-trials 10000 \
  --fetch-samples 1 \
  "${fasta_args[@]}" \
  > "$RUN_ROOT/$STAGE/data_validation.json"

if [[ "$STAGE" == "validate" ]]; then
  echo "Validation passed: $RUN_ROOT/$STAGE/data_validation.json"
  exit 0
fi

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "stage=$STAGE"
  echo "data_dir=$ONEMAIZE_DATA_DIR"
  echo "fasta_root=${ONEMAIZE_FASTA_ROOT:-paths-from-genomes.parquet}"
  echo "context=$context"
  python --version
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
} > "$RUN_ROOT/$STAGE/run_manifest.txt"

common_args=(
  trainer.devices=1
  trainer.accelerator=gpu
  loader.num_workers="$NUM_WORKERS"
  train.test=false
  wandb=null
  hydra.run.dir="$RUN_ROOT/$STAGE"
  "${hydra_fasta_args[@]}"
)

if [[ "$STAGE" == "smoke" ]]; then
  python -m train experiment=onemaize_b73_smoke "${common_args[@]}"
  exit 0
fi

if [[ -z "$MAX_STEPS" || -z "$WARMUP_STEPS" ]]; then
  echo "Set MAX_STEPS and WARMUP_STEPS after the smoke/benchmark is accepted." >&2
  exit 2
fi

if [[ "$STAGE" == "8k" ]]; then
  python -m train \
    experiment=onemaize_b73_8k \
    trainer.max_steps="$MAX_STEPS" \
    scheduler.warmup_t="$WARMUP_STEPS" \
    "${common_args[@]}"
  exit 0
fi

if [[ -z "$PHASE1_CKPT" || ! -f "$PHASE1_CKPT" ]]; then
  echo "Set PHASE1_CKPT to the completed 8K checkpoint." >&2
  exit 2
fi
python -m train \
  experiment=onemaize_b73_16k \
  trainer.max_steps="$MAX_STEPS" \
  scheduler.warmup_t="$WARMUP_STEPS" \
  train.pretrained_model_path="$PHASE1_CKPT" \
  "${common_args[@]}"
