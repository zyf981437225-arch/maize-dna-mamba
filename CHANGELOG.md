# Changelog

## OneMaize acceptance hardening

- `src/onemaize/regions.py`: added exact NAM26/B73 formal guards, streaming
  Parquet writes, full FASTA A/C/G/T/N audit, assembly-gap coverage, and a 10%
  N candidate filter. RNA training: no. Checkpoint compatibility: no effect.
- `src/dataloaders/datasets/onemaize_dataset.py` and
  `src/dataloaders/onemaize_mlm.py`: added Arrow-backed scalable metadata and
  within-pool high-N crop retries that preserve hierarchical sampling. RNA
  training: no. Checkpoint compatibility: no effect.
- `configs/experiment/onemaize_b73_{8k,16k}.yaml`: changed the preserved
  backbone width from 768 to 864 so 24 blocks meet the approximately 120M plan
  target. RNA training: no. Checkpoint compatibility: these production configs
  define a new 864-width checkpoint class.
- `configs/experiment/onemaize_b73_pilot_8k.yaml`: added the requested small
  approximately 44M ablation. RNA training: no. Checkpoint compatibility: new
  isolated pilot class.
- `src/onemaize/model_budget.py` and
  `scripts/check_onemaize_model_budget.py`: added a hard analytic parameter
  budget gate. RNA training: no. Checkpoint compatibility: no effect.
- `scripts/benchmark_onemaize.py` and `scripts/run_onemaize_a100.sh`: added the
  Phase-0 data-I/O/model-cost comparison and guarded pilot stage. RNA training:
  no. Checkpoint compatibility: no effect.
- `scripts/validate_onemaize_data.py`: strengthened formal panel, FASTA audit,
  indexed access, class distribution, and masked-fraction validation. RNA
  training: no. Checkpoint compatibility: no effect.
- `tests/test_onemaize_pipeline.py`: added exact-panel, B73 split, parameter
  budget, N-filter, and runtime retry tests. RNA training: no. Checkpoint
  compatibility: no effect.
- `src/models/sequence/checkpoint_hooks.py` and `src/utils/registry.py`: moved
  pure state-dict mapping hooks behind a lightweight import boundary so
  checkpoint tests and CPU tooling do not require CUDA Mamba at import time.
  RNA training: behavior unchanged. Checkpoint compatibility: format and
  mapping semantics unchanged.

## OneMaize teacher-plan pipeline

- Added annotation-aware `regions.parquet` construction from per-genotype
  FASTA, protein-coding gene GFF3, and TE GFF3 inputs.
- Added uniform-genotype and 50/30/20 region-class sampling, dynamic 8K/16K
  crops, training-only 50% reverse complement, and 15% 80/10/10 DNA MLM.
- Added B73 chromosome pilot splits and strict formal validation for the
  26-genotype 23/1/2 split.
- Added A100 data validation, smoke, Phase-I 8K, and Phase-II 16K launch paths.
- Retained the bidirectional Caduceus/Mamba, BCW writer, memory pool, and
  cross-attention reader without changing their implementation files.

## Initial Maize DNA-Mamba implementation

### Project boundary

- Created Maize DNA-Mamba as an independent repository derived from the
  Caduceus/RNA-Mamba training framework.
- Made genomic DNA the only registered training data module and the DNA
  tokenizer the default and enforced sequence type.
- Preserved the reusable bidirectional backbone, BCW writer, cross-layer
  memory bank, lightweight reader, MLM task, and checkpoint loader.

### Data and tokenization

- Added A/C/G/T/N single-nucleotide tokenization and DNA complement mapping.
- Added streaming multi-FASTA parsing, whole-contig split assignment,
  configurable window/stride generation, ambiguity policies, and U/N gates.
- Added a fixed-window memory-mapped corpus with deterministic validation/test
  MLM and attention masks.
- Added generated `manifest.json` and `DATA_STATS.md` evidence artifacts.

### Training and verification

- Added two-step DNA smoke and guarded 768 × 12 single-A100 configurations.
- Added direct 10,240-token forward/loss/backward/optimizer validation.
- Added matched backbone-only versus BCW/memory performance benchmarking.
- Added tests for DNA token ids, reverse-complement involution, contig safety,
  split isolation, U gating, deterministic MLM, and memory-reader broadcasting.

### Compatibility

- Kept vocabulary size 12 and base ids A=7, C=8, G=9, T=10, N=11.
- Architecture-matched legacy checkpoints are shape-compatible, but row 10 is
  a documented U-to-T semantic remapping and scratch training remains default.
- Strict RCPS plus memory remains disabled pending the localized repair and
  numerical equivariance tests documented in `RC_MEMORY_COMPATIBILITY.md`.
