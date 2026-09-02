#!/usr/bin/env bash
set -euo pipefail

# INTERNAL implementation used by pbs_scripts/02_b73_reference_phase2_train.pbs.
# Run once inside a single-node allocation; Lightning launches local GPU workers.
STAGE="${1:-validate}"
case "$STAGE" in
  validate|smoke|train|test) ;;
  *) echo "Usage: bash scripts/run_onemaize_b73_phase2_h200.sh [validate|smoke|train|test]" >&2; exit 2 ;;
esac

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ONEMAIZE_DATA_DIR="${ONEMAIZE_DATA_DIR:-}"
ONEMAIZE_FASTA_ROOT="${ONEMAIZE_FASTA_ROOT:-}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/onemaize_b73_phase2_h200}"
NUM_DEVICES="${NUM_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-100000}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-2048}"
TEST_SAMPLES_PER_EPOCH="${TEST_SAMPLES_PER_EPOCH:-2048}"
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-}"
PHASE1_CKPT="${PHASE1_CKPT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
EVAL_CKPT="${EVAL_CKPT:-}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"
EVAL_SAMPLES_PER_CLASS="${EVAL_SAMPLES_PER_CLASS:-256}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ -d "$ONEMAIZE_DATA_DIR" ]] || die "Set ONEMAIZE_DATA_DIR to B73 schema-v3 metadata"
[[ -f "$ONEMAIZE_DATA_DIR/manifest.json" ]] || die "Missing $ONEMAIZE_DATA_DIR/manifest.json"
[[ -z "$PHASE1_CKPT" || -z "$RESUME_CKPT" ]] || die "Use PHASE1_CKPT for a new run or RESUME_CKPT for exact resume, not both"
export ONEMAIZE_DATA_DIR
cd "$ROOT_DIR"
mkdir -p "$RUN_ROOT/$STAGE"

fasta_cli_args=()
fasta_hydra_args=()
if [[ -n "$ONEMAIZE_FASTA_ROOT" ]]; then
  fasta_cli_args=(--fasta-root "$ONEMAIZE_FASTA_ROOT")
  fasta_hydra_args=(dataset.fasta_root="$ONEMAIZE_FASTA_ROOT")
fi

run_validate() {
  "$PYTHON_BIN" scripts/validate_onemaize_data.py \
    --data-dir "$ONEMAIZE_DATA_DIR" --context-length 16384 \
    "${fasta_cli_args[@]}" | tee "$RUN_ROOT/validate/validation_16k.json"
  "$PYTHON_BIN" scripts/audit_onemaize_phase2_candidate_lengths.py \
    --data-dir "$ONEMAIZE_DATA_DIR" --context-length 16384 \
    --output-md "$RUN_ROOT/validate/candidate_lengths.md"
}

if [[ "$STAGE" == "validate" ]]; then
  mkdir -p "$RUN_ROOT/validate"
  run_validate
  exit 0
fi

if [[ "$STAGE" == "test" ]]; then
  [[ -n "$EVAL_CKPT" && -f "$EVAL_CKPT" ]] || die "Set EVAL_CKPT to an existing checkpoint"
  exec "$PYTHON_BIN" scripts/evaluate_onemaize_checkpoints.py \
    --checkpoint "B73_Phase-II=$(realpath "$EVAL_CKPT")" \
    --base-data-dir "$ONEMAIZE_DATA_DIR" --split test --context-length 16384 \
    --samples-per-class "$EVAL_SAMPLES_PER_CLASS" \
    --output-csv "$RUN_ROOT/test/checkpoint_evaluation.csv" \
    --output-markdown "$RUN_ROOT/test/checkpoint_evaluation.md" \
    "${fasta_cli_args[@]}"
fi

if [[ -n "$RESUME_CKPT" ]]; then
  [[ -f "$RESUME_CKPT" ]] || die "Missing RESUME_CKPT: $RESUME_CKPT"
else
  [[ -n "$PHASE1_CKPT" && -f "$PHASE1_CKPT" ]] || die "A new Phase-II run requires PHASE1_CKPT"
  "$PYTHON_BIN" scripts/check_onemaize_checkpoint_contract.py \
    --checkpoint "$PHASE1_CKPT" --kind phase1-init
fi

stage_steps="$MAX_STEPS"
stage_warmup="$WARMUP_STEPS"
if [[ "$STAGE" == "smoke" ]]; then
  stage_steps="$SMOKE_STEPS"
  stage_warmup=2
fi
[[ "$stage_steps" =~ ^[1-9][0-9]*$ ]] || die "Set MAX_STEPS to a positive integer"
[[ "$stage_warmup" =~ ^[0-9]+$ ]] || die "Set WARMUP_STEPS to a non-negative integer"
(( stage_warmup < stage_steps )) || die "WARMUP_STEPS must be smaller than MAX_STEPS"
if [[ -z "$CHECKPOINT_INTERVAL" ]]; then
  CHECKPOINT_INTERVAL=$((stage_steps / 20)); (( CHECKPOINT_INTERVAL > 0 )) || CHECKPOINT_INTERVAL=1
fi

RUN_DIR="$RUN_ROOT/$STAGE"
mkdir -p "$RUN_DIR/checkpoints_best" "$RUN_DIR/checkpoints_resume"
init_args=(train.pretrained_model_path=null)
resume_args=(train.ckpt=null)
if [[ -n "$RESUME_CKPT" ]]; then
  resume_args=(train.ckpt="$RESUME_CKPT")
else
  init_args=(train.pretrained_model_path="$PHASE1_CKPT" train.pretrained_model_strict_load=true)
fi
checkpoint_args=(
  callbacks.model_checkpoint.dirpath="$RUN_DIR/checkpoints_best"
  callbacks.model_checkpoint.filename=val_loss
  callbacks.model_checkpoint_every_n_steps.dirpath="$RUN_DIR/checkpoints_resume"
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps="$CHECKPOINT_INTERVAL"
  callbacks.model_checkpoint_every_n_steps.save_last=true
)
if [[ "$STAGE" == "smoke" ]]; then
  checkpoint_args=("~callbacks.periodic_checkpoint" "~callbacks.model_checkpoint_every_n_steps")
fi

command=(
  "$PYTHON_BIN" -m train experiment=onemaize_b73_phase2_16k_region_aware
  trainer.accelerator=gpu trainer.devices="$NUM_DEVICES"
  trainer.max_epochs=null trainer.max_steps="$stage_steps"
  trainer.accumulate_grad_batches="$GRAD_ACCUM"
  scheduler.warmup_t="$stage_warmup"
  dataset.data_dir="$ONEMAIZE_DATA_DIR"
  dataset.batch_size="$BATCH_SIZE" dataset.batch_size_eval="$BATCH_SIZE_EVAL"
  dataset.train_samples_per_epoch="$TRAIN_SAMPLES_PER_EPOCH"
  dataset.val_samples_per_epoch="$VAL_SAMPLES_PER_EPOCH"
  dataset.test_samples_per_epoch="$TEST_SAMPLES_PER_EPOCH"
  loader.num_workers="$NUM_WORKERS" wandb=null hydra.run.dir="$RUN_DIR"
  "${fasta_hydra_args[@]}" "${init_args[@]}" "${resume_args[@]}" "${checkpoint_args[@]}"
)

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "model_branch=B73_Phase-I_to_B73_Phase-II"
  echo "phase1_checkpoint=${PHASE1_CKPT:-none}"
  echo "resume_checkpoint=${RESUME_CKPT:-none}"
} > "$RUN_DIR/run_manifest.txt"

printf 'Running B73 Phase-II %s:' "$STAGE"; printf ' %q' "${command[@]}"; echo
"${command[@]}" 2>&1 | tee "$RUN_DIR/console.log"
