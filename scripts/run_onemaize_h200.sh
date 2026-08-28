#!/usr/bin/env bash
set -euo pipefail

# Portable single-node launcher for formal OneMaize training on 8 x H200.
# It is intentionally separate from run_formal_8gpu.sh, which belongs to the
# legacy RNA/m6A workflow.  Run this script once inside an allocation; when a
# Slurm allocation is detected, the script starts one Lightning process per GPU
# with srun.  Outside Slurm, Lightning launches the local worker processes.

STAGE="${1:-smoke}"
case "$STAGE" in
  validate|benchmark|smoke|8k|16k|test8k|test16k) ;;
  *)
    echo "Usage: bash scripts/run_onemaize_h200.sh [validate|benchmark|smoke|8k|16k|test8k|test16k]" >&2
    exit 2
    ;;
esac

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ONEMAIZE_DATA_DIR="${ONEMAIZE_DATA_DIR:-}"
ONEMAIZE_FASTA_ROOT="${ONEMAIZE_FASTA_ROOT:-}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/onemaize_h200}"
NUM_DEVICES="${NUM_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-100000}"
VAL_SAMPLES_PER_EPOCH="${VAL_SAMPLES_PER_EPOCH:-2048}"
TEST_SAMPLES_PER_EPOCH="${TEST_SAMPLES_PER_EPOCH:-2048}"
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-}"
INIT_CKPT="${INIT_CKPT:-}"
RESUME_CKPT="${RESUME_CKPT:-}"
EVAL_CKPT="${EVAL_CKPT:-}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_FORMAL="${REQUIRE_FORMAL:-1}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-2}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-10}"
BENCHMARK_IO_STEPS="${BENCHMARK_IO_STEPS:-64}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"
SMOKE_WARMUP_STEPS="${SMOKE_WARMUP_STEPS:-2}"
SMOKE_VAL_SAMPLES="${SMOKE_VAL_SAMPLES:-128}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer; got '$value'"
}

[[ -n "$ONEMAIZE_DATA_DIR" ]] || die "Set ONEMAIZE_DATA_DIR to the processed NAM26 metadata directory."
[[ -f "$ONEMAIZE_DATA_DIR/manifest.json" ]] || die "Missing $ONEMAIZE_DATA_DIR/manifest.json"
[[ "$REQUIRE_FORMAL" == "0" || "$REQUIRE_FORMAL" == "1" ]] || die "REQUIRE_FORMAL must be 0 or 1."

if [[ "$REQUIRE_FORMAL" == "1" ]]; then
  python - "$ONEMAIZE_DATA_DIR/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], "rt", encoding="utf-8") as handle:
    manifest = json.load(handle)

expected_splits = {"train": 23, "val": 1, "test": 2}
errors = []
if manifest.get("genotype_count") != 26:
    errors.append(f"genotype_count={manifest.get('genotype_count')!r}, expected 26")
if manifest.get("genotype_split_counts") != expected_splits:
    errors.append(
        f"genotype_split_counts={manifest.get('genotype_split_counts')!r}, "
        f"expected {expected_splits!r}"
    )
if not manifest.get("formal_split_validated", False):
    errors.append("formal_split_validated is not true")
if not manifest.get("expected_genotype_panel_validated", False):
    errors.append("expected_genotype_panel_validated is not true")
if "B73" not in manifest.get("required_train_genotypes", []):
    errors.append("B73 was not required in the training split")
if errors:
    raise SystemExit(
        "H200 production launcher requires formal NAM26 metadata:\n- "
        + "\n- ".join(errors)
        + "\nSet REQUIRE_FORMAL=0 only for an explicitly labelled ablation."
    )
PY
fi

require_positive_integer NUM_DEVICES "$NUM_DEVICES"
require_positive_integer BATCH_SIZE "$BATCH_SIZE"
require_positive_integer BATCH_SIZE_EVAL "$BATCH_SIZE_EVAL"
require_positive_integer GRAD_ACCUM "$GRAD_ACCUM"
require_positive_integer NUM_WORKERS "$NUM_WORKERS"
require_positive_integer TRAIN_SAMPLES_PER_EPOCH "$TRAIN_SAMPLES_PER_EPOCH"
require_positive_integer VAL_SAMPLES_PER_EPOCH "$VAL_SAMPLES_PER_EPOCH"
require_positive_integer TEST_SAMPLES_PER_EPOCH "$TEST_SAMPLES_PER_EPOCH"

if [[ -n "$INIT_CKPT" && -n "$RESUME_CKPT" ]]; then
  die "Set only one of INIT_CKPT (new stage) and RESUME_CKPT (exact training-state resume)."
fi
if [[ -n "$INIT_CKPT" && ! -f "$INIT_CKPT" ]]; then
  die "Missing INIT_CKPT: $INIT_CKPT"
fi
if [[ -n "$RESUME_CKPT" && ! -f "$RESUME_CKPT" ]]; then
  die "Missing RESUME_CKPT: $RESUME_CKPT"
fi

export ONEMAIZE_DATA_DIR
cd "$ROOT_DIR"
mkdir -p "$RUN_ROOT"

fasta_cli_args=()
fasta_hydra_args=()
if [[ -n "$ONEMAIZE_FASTA_ROOT" ]]; then
  fasta_cli_args=(--fasta-root "$ONEMAIZE_FASTA_ROOT")
  fasta_hydra_args=(dataset.fasta_root="$ONEMAIZE_FASTA_ROOT")
fi

if [[ "$STAGE" == "validate" ]]; then
  mkdir -p "$RUN_ROOT/validate"
  python scripts/validate_onemaize_data.py \
    --data-dir "$ONEMAIZE_DATA_DIR" \
    --context-length 8192 \
    --formal \
    "${fasta_cli_args[@]}" \
    > "$RUN_ROOT/validate/validation_8k.json"
  python scripts/validate_onemaize_data.py \
    --data-dir "$ONEMAIZE_DATA_DIR" \
    --context-length 16384 \
    --formal \
    "${fasta_cli_args[@]}" \
    > "$RUN_ROOT/validate/validation_16k.json"
  echo "Formal 8K and 16K validation passed: $RUN_ROOT/validate"
  exit 0
fi

if [[ "$STAGE" == "benchmark" ]]; then
  mkdir -p "$RUN_ROOT/benchmark"
  python scripts/check_onemaize_model_budget.py
  python scripts/benchmark_onemaize.py \
    --data-dir "$ONEMAIZE_DATA_DIR" \
    --context-length 8192 \
    --d-model 864 \
    --n-layer 24 \
    --batch-size "$BATCH_SIZE" \
    --precision bf16 \
    --warmup-steps "$BENCHMARK_WARMUP" \
    --steps "$BENCHMARK_STEPS" \
    --io-steps "$BENCHMARK_IO_STEPS" \
    --num-workers "$NUM_WORKERS" \
    --output-json "$RUN_ROOT/benchmark/phase0_h200.json" \
    "${fasta_cli_args[@]}"
  exit 0
fi

if [[ "$DRY_RUN" != "1" ]]; then
  python - "$NUM_DEVICES" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active Python environment")
visible = torch.cuda.device_count()
if visible < expected:
    raise SystemExit(f"Expected at least {expected} visible GPUs, found {visible}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The active CUDA/PyTorch stack does not report BF16 support")
print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "visible_gpus": visible,
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(visible)],
    "bf16_supported": True,
})
PY
fi

GLOBAL_BATCH=$((NUM_DEVICES * BATCH_SIZE * GRAD_ACCUM))

experiment=onemaize_b73_8k
context=8192
run_name="$STAGE"
stage_max_steps="$MAX_STEPS"
stage_warmup_steps="$WARMUP_STEPS"
validate_at_start=false

case "$STAGE" in
  smoke)
    stage_max_steps="$SMOKE_STEPS"
    stage_warmup_steps="$SMOKE_WARMUP_STEPS"
    TRAIN_SAMPLES_PER_EPOCH=$((GLOBAL_BATCH * SMOKE_STEPS))
    VAL_SAMPLES_PER_EPOCH="$SMOKE_VAL_SAMPLES"
    TEST_SAMPLES_PER_EPOCH="$SMOKE_VAL_SAMPLES"
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-$((SMOKE_STEPS > 1 ? SMOKE_STEPS / 2 : 1))}"
    validate_at_start=true
    run_name=smoke_8gpu
    ;;
  8k)
    ;;
  16k)
    experiment=onemaize_b73_16k
    context=16384
    if [[ -z "$RESUME_CKPT" && -z "$INIT_CKPT" ]]; then
      die "Stage 16k requires INIT_CKPT from the completed 8K stage, or RESUME_CKPT for a 16K resume."
    fi
    ;;
  test8k)
    experiment=onemaize_b73_8k
    context=8192
    ;;
  test16k)
    experiment=onemaize_b73_16k
    context=16384
    ;;
esac

STEPS_PER_EPOCH=$(((TRAIN_SAMPLES_PER_EPOCH + GLOBAL_BATCH - 1) / GLOBAL_BATCH))

RUN_DIR="$RUN_ROOT/$run_name"
mkdir -p "$RUN_DIR/checkpoints_best" "$RUN_DIR/checkpoints_resume"

launch_command() {
  local log_path="$1"
  shift
  local -a command=("$@")

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi

  if [[ -n "${SLURM_PROCID:-}" ]]; then
    die "Call this launcher once from the batch script, not as 'srun bash ...'; the launcher creates the srun step."
  fi

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local allocated_tasks="${SLURM_NTASKS:-0}"
    if (( allocated_tasks < NUM_DEVICES )); then
      die "Slurm allocation has SLURM_NTASKS=$allocated_tasks, but NUM_DEVICES=$NUM_DEVICES. Request one task per GPU."
    fi
    set +e
    srun --nodes=1 --ntasks="$NUM_DEVICES" --ntasks-per-node="$NUM_DEVICES" \
      --kill-on-bad-exit=1 "${command[@]}" 2>&1 | tee "$log_path"
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
  fi

  set +e
  "${command[@]}" 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

write_run_manifest() {
  {
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "stage=$STAGE"
    echo "context=$context"
    echo "data_dir=$ONEMAIZE_DATA_DIR"
    echo "fasta_root=${ONEMAIZE_FASTA_ROOT:-paths-from-genomes.parquet}"
    echo "num_devices=$NUM_DEVICES"
    echo "batch_size_per_gpu=$BATCH_SIZE"
    echo "gradient_accumulation=$GRAD_ACCUM"
    echo "global_batch=$GLOBAL_BATCH"
    echo "train_samples_per_epoch=$TRAIN_SAMPLES_PER_EPOCH"
    echo "approx_optimizer_steps_per_epoch=$STEPS_PER_EPOCH"
    echo "max_steps=${stage_max_steps:-not-applicable}"
    echo "warmup_steps=${stage_warmup_steps:-not-applicable}"
    echo "init_checkpoint=${INIT_CKPT:-none}"
    echo "resume_checkpoint=${RESUME_CKPT:-none}"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    python --version
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    fi
  } > "$RUN_DIR/run_manifest.txt"
}

if [[ "$STAGE" == "test8k" || "$STAGE" == "test16k" ]]; then
  [[ -n "$EVAL_CKPT" ]] || die "Set EVAL_CKPT to the best checkpoint to test."
  [[ -f "$EVAL_CKPT" ]] || die "Missing EVAL_CKPT: $EVAL_CKPT"
  write_run_manifest
  test_command=(
    python -m train
    experiment="$experiment"
    trainer.devices="$NUM_DEVICES"
    trainer.accelerator=gpu
    trainer.max_epochs=null
    trainer.max_steps=1
    scheduler.warmup_t=0
    dataset.batch_size="$BATCH_SIZE_EVAL"
    dataset.batch_size_eval="$BATCH_SIZE_EVAL"
    dataset.test_samples_per_epoch="$TEST_SAMPLES_PER_EPOCH"
    loader.num_workers="$NUM_WORKERS"
    train.eval_only=true
    train.ckpt="$EVAL_CKPT"
    train.pretrained_model_path=null
    train.test=false
    wandb=null
    "~callbacks.periodic_checkpoint"
    hydra.run.dir="$RUN_DIR"
    "${fasta_hydra_args[@]}"
  )
  launch_command "$RUN_DIR/test.log" "${test_command[@]}"
  exit 0
fi

[[ -n "$stage_max_steps" ]] || die "Set MAX_STEPS for stage $STAGE."
[[ -n "$stage_warmup_steps" ]] || die "Set WARMUP_STEPS for stage $STAGE."
require_positive_integer MAX_STEPS "$stage_max_steps"
[[ "$stage_warmup_steps" =~ ^[0-9]+$ ]] || die "WARMUP_STEPS must be a non-negative integer."
if (( stage_warmup_steps >= stage_max_steps )); then
  die "WARMUP_STEPS ($stage_warmup_steps) must be smaller than MAX_STEPS ($stage_max_steps)."
fi
if [[ -z "$CHECKPOINT_INTERVAL" ]]; then
  CHECKPOINT_INTERVAL=$((stage_max_steps / 20))
  (( CHECKPOINT_INTERVAL > 0 )) || CHECKPOINT_INTERVAL=1
fi
require_positive_integer CHECKPOINT_INTERVAL "$CHECKPOINT_INTERVAL"

init_args=(train.pretrained_model_path=null)
if [[ -n "$INIT_CKPT" ]]; then
  init_args=(
    train.pretrained_model_path="$INIT_CKPT"
    train.pretrained_model_strict_load=true
  )
fi
resume_args=(train.ckpt=null)
if [[ -n "$RESUME_CKPT" ]]; then
  resume_args=(train.ckpt="$RESUME_CKPT")
fi

write_run_manifest

train_command=(
  python -m train
  experiment="$experiment"
  trainer.devices="$NUM_DEVICES"
  trainer.accelerator=gpu
  trainer.max_epochs=null
  trainer.max_steps="$stage_max_steps"
  trainer.accumulate_grad_batches="$GRAD_ACCUM"
  dataset.batch_size="$BATCH_SIZE"
  dataset.batch_size_eval="$BATCH_SIZE_EVAL"
  dataset.train_samples_per_epoch="$TRAIN_SAMPLES_PER_EPOCH"
  dataset.val_samples_per_epoch="$VAL_SAMPLES_PER_EPOCH"
  dataset.test_samples_per_epoch="$TEST_SAMPLES_PER_EPOCH"
  loader.num_workers="$NUM_WORKERS"
  train.validate_at_start="$validate_at_start"
  train.test=false
  scheduler.warmup_t="$stage_warmup_steps"
  "${init_args[@]}"
  "${resume_args[@]}"
  "~callbacks.periodic_checkpoint"
  callbacks.model_checkpoint.dirpath="$RUN_DIR/checkpoints_best"
  callbacks.model_checkpoint.filename=val_loss
  callbacks.model_checkpoint_every_n_steps.dirpath="$RUN_DIR/checkpoints_resume"
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps="$CHECKPOINT_INTERVAL"
  callbacks.model_checkpoint_every_n_steps.save_last=true
  wandb=null
  hydra.run.dir="$RUN_DIR"
  "${fasta_hydra_args[@]}"
)

echo "OneMaize $STAGE: context=$context devices=$NUM_DEVICES global_batch=$GLOBAL_BATCH max_steps=$stage_max_steps"
echo "Approximate optimizer steps per epoch: $STEPS_PER_EPOCH"
launch_command "$RUN_DIR/console.log" "${train_command[@]}"
