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

The completed local build contains 164,150 candidate intervals after a full
A/C/G/T/N audit and removal of candidates with more than 10% `N`:

| Split | Gene-centered | Non-repeat | TE-rich |
|---|---:|---:|---:|
| train | 33,329 | 101 | 106,505 |
| validation | 2,988 | 40 | 9,507 |
| test | 2,704 | 2 | 8,974 |

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
  --test-seqid chr10 \
  --max-n-fraction 0.1
```

The builder converts GFF3 coordinates once from 1-based inclusive to 0-based
half-open, merges overlapping TE annotations before calculating coverage, and
uses a 50% repeat threshold. Gene candidates are protein-coding gene bodies
plus 5 kb on both sides. Genome-wide candidates use 32,768-bp spans and
16,384-bp stride so either curriculum context can be dynamically cropped. The
loader retries a high-`N` crop inside the already selected genotype/class pool,
so quality control cannot distort uniform-genotype or 50/30/20 sampling.

## Validate and smoke-test on the A100

```bash
export ONEMAIZE_DATA_DIR=/data/processed/onemaize_b73
bash scripts/run_onemaize_a100.sh validate
bash scripts/run_onemaize_a100.sh benchmark
bash scripts/run_onemaize_a100.sh smoke
```

If the metadata was built on another machine, point the loader at the server
directory containing the FASTA and its indexes:

```bash
export ONEMAIZE_FASTA_ROOT=/data/NAM/B73
```

The launcher refuses to train when the metadata, FASTA, `.fai`, or `.gzi` is
missing. The benchmark records host I/O throughput, forward/backward/step time,
tokens/s, peak GPU memory, instantiated parameter count, and the overhead of
BCW/memory versus the preserved backbone alone. The smoke stage performs two
full 8K optimization steps through the same path.

## Train the two context stages

First run the guarded ~44M B73 pilot after observing Phase-0 memory and step
time on the 80-GB A100:

```bash
export MAX_STEPS=<approved_pilot_steps>
export WARMUP_STEPS=<approved_pilot_warmup>
bash scripts/run_onemaize_a100.sh pilot
```

Then choose production step and warmup counts from measured throughput:

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

Both production configs use BF16, batch size 1, gradient accumulation, 864
hidden dimensions, 24 blocks, tied bidirectional weights, and the existing
memory path. The analytic estimate is 121,191,553 trainable parameters; the
instantiated count and peak memory from the Linux/Mamba environment remain the
final authority.

## Build the formal 26-genotype corpus

Prepare a TSV with one row per genotype:

```text
genotype\tfasta\tgenes_gff3\tte_gff3\tsplit
B73\t/data/NAM/B73/genome.fa.gz\t/data/NAM/B73/genes.gff3.gz\t/data/NAM/B73/TE.gff3.gz\ttrain
...
```

Exactly 23 rows must be `train`, one `val`, and two `test`; the panel must be
the exact B73 + 25 NAM list in the plan, and B73 must remain `train`. The
held-out genotypes should be chosen by diversity analysis and agreed with the
teacher before freezing the manifest.

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

`--formal` rejects an incorrect founder name, an incorrect 23/1/2 split, B73
held out from training, a skipped FASTA audit, or pilot chromosome overrides.
