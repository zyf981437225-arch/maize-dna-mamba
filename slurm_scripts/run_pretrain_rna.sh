#!/usr/bin/env bash
#SBATCH --job-name=rna_mamba
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

: "${RNA_TEXT_FILE:?Export RNA_TEXT_FILE as the absolute TXT dataset path}"
: "${RNA_FASTA_FILE:?Export RNA_FASTA_FILE as the absolute FASTA dataset path}"

NUM_DEVICES="${NUM_DEVICES:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-16}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
MAX_STEPS="${MAX_STEPS:-20000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_DIR="${RUN_DIR:-outputs/pretrain_rna/${SLURM_JOB_ID:-manual}}"

# Activate the environment before submitting, or export an activation command,
# for example: ENV_ACTIVATE='source ~/miniconda3/bin/activate caduceus_env'
if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  eval "${ENV_ACTIVATE}"
fi

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

srun python -m train \
  experiment=hg38/hg38 \
  model=caduceus \
  dataset.dataset_name=mixed_rna \
  +dataset.text_file="${RNA_TEXT_FILE}" \
  +dataset.rna_fasta_file="${RNA_FASTA_FILE}" \
  dataset.tokenizer_name=char \
  +dataset.kmer=1 \
  +dataset.frame=0 \
  dataset.max_length="${MAX_LENGTH}" \
  dataset.max_length_val="${MAX_LENGTH}" \
  dataset.max_length_test="${MAX_LENGTH}" \
  dataset.batch_size="${PER_DEVICE_BATCH}" \
  dataset.batch_size_eval="${PER_DEVICE_BATCH}" \
  loader.num_workers="${NUM_WORKERS}" \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  model.config.d_model=768 \
  model.config.n_layer=12 \
  model.config.vocab_size=12 \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=add \
  model.config.bidirectional_weight_tie=true \
  model.config.rcps=false \
  model.config.use_memory=true \
  model.config.memory_persist_across_batches=false \
  optimizer.lr=8e-5 \
  optimizer.weight_decay=0.01 \
  'optimizer.betas=[0.9,0.98]' \
  train.global_batch_size="${GLOBAL_BATCH}" \
  train.ckpt=checkpoints/last.ckpt \
  trainer.devices="${NUM_DEVICES}" \
  trainer.num_nodes=1 \
  trainer.precision=16 \
  trainer.gradient_clip_val=0.5 \
  trainer.max_epochs="${MAX_EPOCHS}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.num_sanity_val_steps=0 \
  +trainer.val_check_interval=100 \
  scheduler._name_=cosine_warmup_timm \
  scheduler.t_initial="${MAX_STEPS}" \
  scheduler.warmup_t=4000 \
  scheduler.lr_min=2e-5 \
  scheduler.warmup_lr_init=1e-6 \
  wandb=null \
  train.monitor=val/loss \
  train.mode=min \
  callbacks.model_checkpoint.monitor=val/loss \
  callbacks.model_checkpoint.mode=min \
  hydra.run.dir="${RUN_DIR}"
