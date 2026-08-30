#!/usr/bin/env bash
set -euo pipefail

# Teacher-facing Phase-I launcher.  It is safe to run validate/benchmark first;
# no command below uploads FASTA or checkpoints.  Lightning owns DDP for train.
STAGE="${1:-validate}"
case "$STAGE" in
  validate|benchmark|smoke|train|test) ;;
  *) echo "Usage: bash scripts/run_onemaize_phase1_h200.sh [validate|benchmark|smoke|train|test]" >&2; exit 2 ;;
esac

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PHASE1_MANIFEST="${ONEMAIZE_PHASE1_MANIFEST:-${PHASE1_MANIFEST:-}}"
B73_FASTA="${ONEMAIZE_B73_FASTA:-${B73_FASTA:-}}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/onemaize_phase1}"
NUM_DEVICES="${NUM_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_STEPS="${WARMUP_STEPS:-}"
SMOKE_STEPS="${SMOKE_STEPS:-20}"
SMOKE_WARMUP_STEPS="${SMOKE_WARMUP_STEPS:-2}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-}"
EVAL_CKPT="${EVAL_CKPT:-}"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ -n "$PHASE1_MANIFEST" ]] || die "Set ONEMAIZE_PHASE1_MANIFEST to the Phase-I parquet manifest."
[[ -n "$B73_FASTA" ]] || die "Set ONEMAIZE_B73_FASTA to the indexed B73 BGZF FASTA."
[[ -f "$PHASE1_MANIFEST" ]] || die "Missing manifest: $PHASE1_MANIFEST"
[[ -f "$B73_FASTA" ]] || die "Missing FASTA: $B73_FASTA"
[[ -f "$B73_FASTA.fai" && -f "$B73_FASTA.gzi" ]] || die "B73 FASTA requires matching .fai and .gzi."
[[ "$NUM_DEVICES" =~ ^[1-9][0-9]*$ ]] || die "NUM_DEVICES must be positive"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be positive"
mkdir -p "$RUN_ROOT"
cd "$ROOT_DIR"

if [[ "$STAGE" == "validate" ]]; then
  "$PYTHON_BIN" scripts/validate_b73_phase1_8k_manifest.py --manifest "$PHASE1_MANIFEST" --fasta "$B73_FASTA" | tee "$RUN_ROOT/validation.json"
  echo "Phase-I manifest validation passed."
  exit 0
fi

if [[ "$STAGE" == "benchmark" ]]; then
  command=("$PYTHON_BIN" scripts/benchmark_b73_phase1_8k.py --manifest "$PHASE1_MANIFEST" --fasta "$B73_FASTA" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --steps "${BENCHMARK_STEPS:-100}" --warmup-steps "${BENCHMARK_WARMUP_STEPS:-5}" --io-steps "${BENCHMARK_IO_STEPS:-32}" --output-json "$RUN_ROOT/benchmark.json")
  if (( NUM_DEVICES > 1 )); then
    command=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_DEVICES" "${command[@]:1}")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN benchmark:'; printf ' %q' "${command[@]}"; echo
    exit 0
  fi
  "$PYTHON_BIN" - "$NUM_DEVICES" <<'PY'
import sys, torch
n = int(sys.argv[1])
if torch.cuda.is_available() and torch.cuda.device_count() < n:
    raise SystemExit(f"requested {n} GPUs but only {torch.cuda.device_count()} are visible")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable for the benchmark")
PY
  "${command[@]}"
  exit 0
fi

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON_BIN" - "$NUM_DEVICES" <<'PY'
import sys, torch
n = int(sys.argv[1])
if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
if torch.cuda.device_count() < n: raise SystemExit(f"visible GPUs={torch.cuda.device_count()}, requested={n}")
if not torch.cuda.is_bf16_supported(): raise SystemExit("BF16 unsupported")
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpus": torch.cuda.device_count(), "names": [torch.cuda.get_device_name(i) for i in range(n)]})
PY
fi

experiment=onemaize_b73_phase1_8k_full_genome
run_name="$STAGE"
stage_max_steps="$MAX_STEPS"
stage_warmup_steps="$WARMUP_STEPS"
if [[ "$STAGE" == "smoke" ]]; then
  stage_max_steps="$SMOKE_STEPS"
  stage_warmup_steps="$SMOKE_WARMUP_STEPS"
  run_name=smoke
fi
if [[ "$STAGE" == "test" ]]; then
  stage_max_steps=1
  stage_warmup_steps=0
fi
if [[ "$STAGE" == "train" ]]; then
  [[ -n "$stage_max_steps" && -n "$stage_warmup_steps" ]] || die "train requires MAX_STEPS and WARMUP_STEPS"
fi
if [[ "$STAGE" == "smoke" ]]; then
  : "${stage_max_steps:?}"
fi
[[ "$stage_max_steps" =~ ^[1-9][0-9]*$ ]] || die "MAX_STEPS must be positive"
[[ "$stage_warmup_steps" =~ ^[0-9]+$ ]] || die "WARMUP_STEPS must be non-negative"
(( stage_warmup_steps < stage_max_steps )) || die "WARMUP_STEPS must be smaller than MAX_STEPS"
if [[ -z "$CHECKPOINT_INTERVAL" ]]; then CHECKPOINT_INTERVAL=$((stage_max_steps / 20)); (( CHECKPOINT_INTERVAL > 0 )) || CHECKPOINT_INTERVAL=1; fi
RUN_DIR="$RUN_ROOT/$run_name"
mkdir -p "$RUN_DIR/checkpoints_best" "$RUN_DIR/checkpoints_resume"
checkpoint_args=()
if [[ "$STAGE" == "smoke" ]]; then
  # A smoke test must not fill a shared filesystem with full optimizer states.
  checkpoint_args=("~callbacks.periodic_checkpoint" "~callbacks.model_checkpoint_every_n_steps")
fi

if [[ "$STAGE" == "test" ]]; then
  [[ -n "$EVAL_CKPT" && -f "$EVAL_CKPT" ]] || die "Set EVAL_CKPT to an existing checkpoint"
  command=("$PYTHON_BIN" -m train experiment="$experiment" trainer.devices="$NUM_DEVICES" trainer.accelerator=gpu trainer.max_epochs=null trainer.max_steps=1 scheduler.warmup_t=0 dataset.full_genome_manifest="$PHASE1_MANIFEST" dataset.fasta_path="$B73_FASTA" dataset.batch_size="$BATCH_SIZE_EVAL" dataset.batch_size_eval="$BATCH_SIZE_EVAL" loader.num_workers="$NUM_WORKERS" train.eval_only=true train.ckpt="$EVAL_CKPT" train.test=false wandb=null "~callbacks.periodic_checkpoint" hydra.run.dir="$RUN_DIR")
else
  command=("$PYTHON_BIN" -m train experiment="$experiment" trainer.devices="$NUM_DEVICES" trainer.accelerator=gpu trainer.max_epochs=null trainer.max_steps="$stage_max_steps" trainer.accumulate_grad_batches=1 dataset.full_genome_manifest="$PHASE1_MANIFEST" dataset.fasta_path="$B73_FASTA" dataset.train_samples_per_epoch=null dataset.batch_size="$BATCH_SIZE" dataset.batch_size_eval="$BATCH_SIZE_EVAL" loader.num_workers="$NUM_WORKERS" scheduler.warmup_t="$stage_warmup_steps" callbacks.model_checkpoint.dirpath="$RUN_DIR/checkpoints_best" callbacks.model_checkpoint.filename=val_loss callbacks.model_checkpoint_every_n_steps.dirpath="$RUN_DIR/checkpoints_resume" callbacks.model_checkpoint_every_n_steps.every_n_train_steps="$CHECKPOINT_INTERVAL" callbacks.model_checkpoint_every_n_steps.save_last=true wandb=null hydra.run.dir="$RUN_DIR" "${checkpoint_args[@]}")
fi
printf 'Running Phase-I %s:' "$STAGE"; printf ' %q' "${command[@]}"; echo
if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi
"${command[@]}" 2>&1 | tee "$RUN_DIR/console.log"
