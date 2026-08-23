#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

python -m train \
  experiment=hg38/hg38 \
  model=caduceus \
  dataset.dataset_name=mixed_rna \
  +dataset.text_file="${ROOT_DIR}/tests/fixtures/mixed_rna_small.txt" \
  +dataset.rna_fasta_file="${ROOT_DIR}/tests/fixtures/mixed_rna_small.fasta" \
  dataset.tokenizer_name=char \
  +dataset.kmer=1 \
  +dataset.frame=0 \
  dataset.max_length=64 \
  dataset.max_length_val=64 \
  dataset.max_length_test=64 \
  dataset.batch_size=2 \
  dataset.batch_size_eval=2 \
  loader.num_workers=0 \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  model.config.d_model=64 \
  model.config.n_layer=2 \
  model.config.vocab_size=12 \
  model.config.fused_add_norm=true \
  model.config.use_memory=true \
  model.config.memory_d_sum=32 \
  model.config.memory_d_mem=16 \
  model.config.memory_n_heads=4 \
  model.config.memory_write_stride=1 \
  model.config.memory_read_stride=1 \
  model.config.memory_persist_across_batches=false \
  train.global_batch_size=2 \
  train.ckpt=null \
  train.test=true \
  train.monitor=val/loss \
  train.mode=min \
  trainer.max_epochs=null \
  trainer.max_steps=3 \
  trainer.precision=16 \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 \
  +trainer.limit_test_batches=1 \
  +trainer.val_check_interval=1 \
  optimizer.lr=1e-3 \
  scheduler.t_initial=3 \
  scheduler.warmup_t=1 \
  scheduler.lr_min=1e-5 \
  callbacks.model_checkpoint.monitor=val/loss \
  callbacks.model_checkpoint.mode=min \
  wandb=null \
  hydra.run.dir=./outputs/local-smoke
