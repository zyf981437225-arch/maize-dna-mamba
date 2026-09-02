#!/usr/bin/env bash
# INTERNAL / ALTERNATE SCHEDULER: formal teacher workflow uses numbered PBS files.
#SBATCH --job-name=onemaize-p1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE="${STAGE:-smoke}"
NUM_DEVICES="${NUM_DEVICES:-8}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

[[ -d "$PROJECT_DIR" ]] || { echo "Missing PROJECT_DIR: $PROJECT_DIR" >&2; exit 2; }
[[ -n "${ONEMAIZE_B73_FASTA:-}" ]] || { echo "Set ONEMAIZE_B73_FASTA" >&2; exit 2; }
[[ -n "${ONEMAIZE_PHASE1_MANIFEST:-}" ]] || { echo "Set ONEMAIZE_PHASE1_MANIFEST" >&2; exit 2; }

if [[ -n "$CONDA_ENV_NAME" ]]; then
  command -v conda >/dev/null 2>&1 || { echo "conda is not on PATH" >&2; exit 2; }
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi

cd "$PROJECT_DIR"
export NUM_DEVICES
echo "job_id=$SLURM_JOB_ID stage=$STAGE project_dir=$PROJECT_DIR num_devices=$NUM_DEVICES"
python --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
bash scripts/run_onemaize_phase1_h200.sh "$STAGE"
