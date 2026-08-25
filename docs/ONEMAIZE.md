# OneMaize implementation and server workflow

## Scope

The code keeps the existing bidirectional Caduceus/Mamba backbone, BCW
(`BidirectionalMemoryWriter`), `MemoryPool`, and `MemoryCrossAttention`. The
transcript/fixed-window path is replaced by the teacher-plan genomic pipeline:

1. B73 plus 25 NAM founder genomes.
2. Per-genotype FASTA, protein-coding gene GFF3, and TE GFF3.
3. Candidate intervals in `regions.parquet`, not precomputed 8K strings.
4. Uniform genotype sampling followed by 50% gene-centered, 30% non-repeat,
   and 20% TE-rich sampling.
5. Dynamic 8,192-bp crops in Phase I and 16,384-bp crops in Phase II.
6. Training-only reverse complement with probability 0.5.
7. A/C/G/T/N single-base tokens and 15% MLM with the 80/10/10 corruption rule.
8. A formal genotype-level split of 23 train, 1 validation, and 2 test.

The preserved backbone is the repository's current Mamba implementation
(`mamba_ssm.modules.mamba_simple.Mamba`). It is not silently replaced with a
different Mamba2 kernel. This is the explicit exception created by the
requirement to keep the current Mamba/BCW/memory modules.

## What the current B73 files can do

The supplied B73 v5 FASTA, gene GFF3, and TE GFF3 are sufficient for an
end-to-end pilot. The pilot uses chromosomes 1-8 for train, chromosome 9 for
validation, and chromosome 10 for test. This is a technical leakage-safe split,
not the final population-genomics evaluation.

The completed local build contains 164,482 candidate intervals:

| Split | Gene-centered | Non-repeat | TE-rich |
|---|---:|---:|---:|
| train | 33,342 | 296 | 106,591 |
| validation | 2,988 | 62 | 9,512 |
| test | 2,705 | 6 | 8,980 |

The current data can therefore prove that preprocessing, random access, MLM,
BCW/memory forward/backward, and checkpointing work. It cannot replace the
formal 26-genotype pretraining corpus or support genotype-held-out conclusions.

## Build the B73 pilot

Compressed random access requires the matching `.fa.gz.fai` and `.fa.gz.gzi`
files next to the FASTA. MaizeGDB publishes both in the same B73 directory as
the assembly. Keep all three filenames unchanged.

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

The builder converts GFF3 coordinates once from 1-based inclusive to 0-based
half-open, merges overlapping TE annotations before calculating coverage, and
uses a 50% repeat threshold. Gene candidates are protein-coding gene bodies
plus 5 kb on both sides. Genome-wide candidates use 32,768-bp spans and
16,384-bp stride so either curriculum context can be dynamically cropped.

## Validate and smoke-test on the A100

```bash
export ONEMAIZE_DATA_DIR=/data/processed/onemaize_b73
bash scripts/run_onemaize_a100.sh validate
bash scripts/run_onemaize_a100.sh smoke
```

If the metadata was built on another machine, point the loader at the server
directory containing the FASTA and its indexes:

```bash
export ONEMAIZE_FASTA_ROOT=/data/NAM/B73
```

The launcher refuses to train when the metadata, FASTA, `.fai`, or `.gzi` is
missing. The smoke stage performs two full 8K optimization steps through the
dynamic DataLoader, bidirectional Mamba, BCW, and memory sidecar.

## Train the two context stages

Choose step and warmup counts only after observing smoke-test memory and step
time on the 80-GB A100.

```bash
export MAX_STEPS=<approved_8k_steps>
export WARMUP_STEPS=<approved_8k_warmup>
bash scripts/run_onemaize_a100.sh 8k
```

Then continue from the 8K weights at 16K:

```bash
export PHASE1_CKPT=/path/to/8k/checkpoint.ckpt
export MAX_STEPS=<approved_16k_steps>
export WARMUP_STEPS=<approved_16k_warmup>
bash scripts/run_onemaize_a100.sh 16k
```

Both production configs use BF16, batch size 1, gradient accumulation, 768
hidden dimensions, 24 blocks, tied bidirectional weights, and the existing
memory path. Actual parameter count and peak memory must be recorded on the
Linux/Mamba environment before a long run.

## Build the formal 26-genotype corpus

Prepare a TSV with one row per genotype:

```text
genotype\tfasta\tgenes_gff3\tte_gff3\tsplit
B73\t/data/NAM/B73/genome.fa.gz\t/data/NAM/B73/genes.gff3.gz\t/data/NAM/B73/TE.gff3.gz\ttrain
...
```

Exactly 23 rows must be `train`, one `val`, and two `test`. The held-out
genotypes should be agreed with the teacher before freezing the manifest.

```bash
python scripts/build_onemaize_regions.py \
  --input-manifest /data/NAM/onemaize_26.tsv \
  --output-dir /data/processed/onemaize_nam26 \
  --formal

python scripts/validate_onemaize_data.py \
  --data-dir /data/processed/onemaize_nam26 \
  --context-length 8192 \
  --formal
```

`--formal` rejects any corpus that is not exactly 26 genotypes with a 23/1/2
genotype split, and it rejects pilot chromosome overrides.
