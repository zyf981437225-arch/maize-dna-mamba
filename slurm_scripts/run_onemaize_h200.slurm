#!/usr/bin/env bash
#SBATCH --job-name=onemaize-h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Site-specific settings should be supplied with sbatch --export or edited here.
# Required: PROJECT_DIR, ONEMAIZE_DATA_DIR, STAGE.
# Optional: CONDA_ENV_NAME, RUN_ROOT, MAX_STEPS, WARMUP_STEPS, INIT_CKPT,
#           RESUME_CKPT, EVAL_CKPT, BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS.

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE="${STAGE:-smoke}"
NUM_DEVICES="${NUM_DEVICES:-8}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

[[ -d "$PROJECT_DIR" ]] || {
  echo "Missing PROJECT_DIR: $PROJECT_DIR" >&2
  exit 2
}
[[ -n "${ONEMAIZE_DATA_DIR:-}" ]] || {
  echo "Set ONEMAIZE_DATA_DIR with sbatch --export or in this file." >&2
  exit 2
}

# Load the CUDA/compiler modules required by the local H200 cluster before this
# point, either in the submission shell or by adding the site's module commands.
if [[ -n "$CONDA_ENV_NAME" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV_NAME was set, but conda is not on PATH." >&2
    exit 2
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
fi

cd "$PROJECT_DIR"
export NUM_DEVICES
export ONEMAIZE_DATA_DIR

echo "job_id=$SLURM_JOB_ID"
echo "node_list=$SLURM_JOB_NODELIST"
echo "stage=$STAGE"
echo "project_dir=$PROJECT_DIR"
echo "data_dir=$ONEMAIZE_DATA_DIR"
echo "num_devices=$NUM_DEVICES"
python --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Do not prefix this command with srun. The portable launcher creates one srun
# task per GPU so that Lightning receives the correct SLURM rank variables.
bash scripts/run_onemaize_h200.sh "$STAGE"
