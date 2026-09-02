# OneMaize data pipeline

## Required inputs

For each genotype provide an indexed BGZF FASTA (`.fa.gz`, matching `.fai` and `.gzi`), protein-coding gene GFF3, and TE GFF3. File names are flexible; the input TSV stores actual paths.

Recommended layout:

```text
raw/<genotype>/
├── genome.fa.gz
├── genome.fa.gz.fai
├── genome.fa.gz.gzi
├── genes.gff3.gz
└── TE.gff3.gz
```

## B73 Phase-I manifest

```bash
python scripts/build_b73_phase1_8k_manifest.py \
  --fasta raw/B73/genome.fa.gz \
  --output data/processed/onemaize_b73_phase1_8k/b73_phase1_8k_full_genome.parquet \
  --genotype B73 --chromosomes chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 \
  --window-size 8192 --stride 8192

python scripts/validate_b73_phase1_8k_manifest.py \
  --manifest data/processed/onemaize_b73_phase1_8k/b73_phase1_8k_full_genome.parquet \
  --fasta raw/B73/genome.fa.gz
```

Expected: 260,239 sequences, including ten padded tails, covering 2,131,846,805 bp across chr1–chr10.

## B73/NAM26 schema-v3 candidates

Create a tab-separated input manifest with this exact header:

```text
genotype	fasta	genes_gff3	te_gff3	split
```

NAM26 formal mode requires the standard 26 founders, exactly 23 train / 1 val / 2 test, and B73 in train. Build metadata:

```bash
python scripts/build_onemaize_regions.py \
  --input-manifest data/manifests/onemaize_26.tsv \
  --output-dir data/processed/onemaize_nam26 \
  --primary-context 8192 --extended-context 16384 \
  --candidate-span 32768 --candidate-stride 16384 \
  --gene-flank 5000 --repeat-threshold 0.5 --max-n-fraction 0.10 --formal
```

Outputs are `manifest.json`, `genomes.parquet`, `regions.parquet`, and `DATA_STATS.md`. The builder merges overlapping TE intervals before calculating unique repeat-covered bp. Candidates with repeat coverage at least 0.5 are TE-rich; the others are non-repeat. Gene-centered candidates use gene body plus 5 kb on each side.

## Mandatory gates

Before `qsub`, confirm:

```bash
test -s raw/B73/genome.fa.gz.fai
test -s raw/B73/genome.fa.gz.gzi
python scripts/validate_onemaize_data.py \
  --data-dir data/processed/onemaize_nam26 --context-length 16384 --formal
python scripts/audit_onemaize_allcultivar.py \
  --data-dir data/processed/onemaize_nam26 \
  --output-dir runs/nam26_preflight/audit --context-length 16384 --formal
```

Do not train if any genotype/file/class is missing, chromosome names disagree, FASTA contains unsupported symbols, a candidate is shorter than 16K, the formal split gate fails, or the filesystem lacks checkpoint space.
