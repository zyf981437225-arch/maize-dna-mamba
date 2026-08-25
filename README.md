# Maize DNA-Mamba / OneMaize

This independent project pretrains a single-nucleotide maize DNA language
model. It preserves the existing bidirectional Caduceus/Mamba backbone, BCW
writer, memory pool, and memory reader, while implementing the teacher's
OneMaize data and training plan.

The canonical pipeline uses B73 plus 25 NAM founder genomes; per-genotype
FASTA, protein-coding gene GFF3, and TE GFF3; annotation-aware candidate
intervals; 50/30/20 region sampling; dynamic 8K-to-16K cropping; 50% reverse
complement augmentation; A/C/G/T/N tokens; and 15% same-position MLM.

## Current status

- The supplied B73 v5 files build successfully into 164,482 candidate regions.
- The real B73 sampler reproduces the intended 50/30/20 distribution.
- The dynamic DataLoader, MLM alignment, reverse complement, and 8K/16K paths
  have CPU tests.
- A two-step A100 smoke config and guarded 8K/16K production configs are ready.
- Formal mode enforces exactly 26 genotypes split 23/1/2.

B73 alone is enough to run the complete engineering pipeline, but it remains a
pilot. Population-scale training and genotype-held-out evaluation require all
26 genomes.

## Quick start

Create metadata without copying the large sequence files:

```bash
python scripts/build_onemaize_regions.py \
  --genotype B73 \
  --fasta /data/NAM/B73/Zm-B73-REFERENCE-NAM-5.0.fa.gz \
  --genes-gff3 /data/NAM/B73/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3.gz \
  --te-gff3 /data/NAM/B73/Zm-B73-REFERENCE-NAM-5.0.TE.gff3.gz \
  --output-dir /data/processed/onemaize_b73 \
  --val-seqid chr9 \
  --test-seqid chr10
```

The compressed FASTA must have matching `.fa.gz.fai` and `.fa.gz.gzi` files
for random access. Then, on the single A100 80 GB server:

```bash
export ONEMAIZE_DATA_DIR=/data/processed/onemaize_b73
bash scripts/run_onemaize_a100.sh validate
bash scripts/run_onemaize_a100.sh smoke
```

After accepting the measured peak memory and step time:

```bash
export MAX_STEPS=<approved_steps>
export WARMUP_STEPS=<approved_warmup>
bash scripts/run_onemaize_a100.sh 8k
```

See [docs/ONEMAIZE.md](docs/ONEMAIZE.md) for the full B73 and formal NAM26
workflow. The exact strict reverse-complement limitation of the preserved
memory architecture remains documented in
[RC_MEMORY_COMPATIBILITY.md](RC_MEMORY_COMPATIBILITY.md).

## Verification

Local data tests do not require the CUDA Mamba extension:

```bash
python -m pytest -q -p no:cacheprovider \
  tests/test_onemaize_pipeline.py \
  tests/test_maize_dna_pipeline.py
```

The full model forward/backward acceptance gate must run in the Linux A100
environment with the project's PyTorch, CUDA, `causal-conv1d`, and
`mamba-ssm` dependencies.

## Project boundary

This repository is separate from the original RNA-Mamba project. Legacy
RNA-oriented files inherited from the framework are not part of the supported
OneMaize training path and can be removed in a later cleanup after the A100
acceptance run.
