# Maize DNA-Mamba

Maize DNA-Mamba is a single-nucleotide masked-language model for maize genomic
DNA. It combines a bidirectional Caduceus/Mamba backbone with the existing BCW
writer, cross-layer memory bank, and lightweight reader, while replacing the
transcript-oriented data path with leakage-safe genomic windows.

This is an independent project derived from the RNA-Mamba/Caduceus training
framework. The supported pretraining path is DNA-only: A/C/G/T/N tokenization,
10,240-bp windows, chromosome/contig-level splits, and 15% same-position MLM.
It never converts T to U.

## What is implemented

- Streaming FASTA preprocessing without joining contigs.
- Whole-contig train/validation/test assignment to prevent window leakage.
- Configurable 10,240-bp window stride and ambiguity handling.
- A/C/G/T/N and raw U/IUPAC/invalid-character statistics plus readiness gates.
- Memory-mapped fixed-window corpus for multi-worker loading.
- DNA-only tokenizer with A↔T, C↔G, and N↔N complement mapping.
- Two-step structural smoke test and a matched memory-cost benchmark.
- Guarded 768 × 12 single-A100 training config that cannot start a long run
  until `max_steps` and warmup are supplied explicitly.

The current production path is bidirectional but is not claimed to be strictly
reverse-complement equivariant. The reason and the minimal repair boundary are
documented in [RC_MEMORY_COMPATIBILITY.md](RC_MEMORY_COMPATIBILITY.md).

## Environment

The intended runtime is Linux with one NVIDIA A100 80 GB. The inherited tested
stack uses Python 3.10, PyTorch 2.2, CUDA 12.x, `causal-conv1d==1.2.0.post2`,
and `mamba-ssm==1.2.2`.

```bash
conda env create -f caduceus_env.yml
conda activate caduceus_env
```

Use the established server environment if it already provides compatible
PyTorch, Mamba, Triton, Hydra, and Lightning packages.

## 1. Prepare maize genomic windows

Create a split file using exact FASTA contig identifiers. This example is only
the schema; choose the biological split deliberately.

```json
{
  "train": ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8"],
  "val": ["chr9"],
  "test": ["chr10"]
}
```

```bash
python scripts/prepare_maize_genome.py \
  --fasta /data/genomes/B73.fa \
  --genome-name B73 \
  --species "Zea mays" \
  --split-config /data/splits/b73_splits.json \
  --unassigned-policy error \
  --output-dir /data/processed/maize_b73_10240 \
  --window-size 10240 \
  --stride 5120 \
  --ambiguity-policy map_to_n \
  --max-n-fraction 1.0 \
  --max-u-fraction 0.0 \
  --require-all-splits
```

Review `/data/processed/maize_b73_10240/manifest.json` and the generated
`DATA_STATS.md`. Do not train unless `formal_pretraining_ready` is `true`.

## 2. Run the structural smoke test

```bash
python scripts/smoke_test_dna.py \
  --data-dir /data/processed/maize_b73_10240 \
  --window-size 10240 \
  --steps 2 \
  --device cuda \
  --precision fp16
```

This checks a full 10,240-token forward pass, MLM alignment and loss, backward,
finite gradients, optimizer update, and peak GPU memory. With non-empty
train/validation/test splits, also exercise the Hydra/Lightning path:

```bash
export MAIZE_DNA_INDEXED_DIR=/data/processed/maize_b73_10240
python train.py experiment=maize_dna_smoke
```

## 3. Benchmark the memory sidecar

```bash
python scripts/benchmark_dna.py \
  --data-dir /data/processed/maize_b73_10240 \
  --window-size 10240 \
  --batch-size 1 \
  --d-model 768 \
  --n-layer 12 \
  --mode both \
  --warmup-steps 2 \
  --steps 5 \
  --precision fp16 \
  --device cuda \
  --output-json /data/runs/maize_dna_benchmark.json
```

The benchmark compares the same bidirectional backbone with memory disabled
and enabled, reporting step time, forward/backward time, tokens per second,
parameter count, and peak allocated GPU memory.

## 4. Start a guarded full-size run

Only after the data report, smoke output, and benchmark are accepted:

```bash
export MAIZE_DNA_INDEXED_DIR=/data/processed/maize_b73_10240
python train.py experiment=maize_dna_pretrain \
  trainer.max_steps=<approved_steps> \
  scheduler.warmup_t=<approved_warmup_steps>
```

Training from scratch is the default. An architecture-matched RNA checkpoint
is shape-compatible, but token id 10 changes semantic meaning from U to T; any
such warm start must be tested separately and reported explicitly.

## Verification and project notes

- [docs/MAIZE_DNA.md](docs/MAIZE_DNA.md): detailed server workflow.
- [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md): inherited model/data audit
  and design decisions.
- [RC_MEMORY_COMPATIBILITY.md](RC_MEMORY_COMPATIBILITY.md): strict-RC analysis.
- [DATA_STATS.md](DATA_STATS.md): local data-readiness status.
- [CHANGELOG.md](CHANGELOG.md): initial independent-project implementation.

Local data-pipeline tests do not require `mamba_ssm`:

```bash
python -m pytest -q -p no:cacheprovider tests/test_maize_dna_pipeline.py
```

The complete forward/backward acceptance gate must run on the A100 environment.
