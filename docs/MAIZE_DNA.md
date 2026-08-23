# Maize DNA-Mamba server workflow

Run these commands in the established Linux/CUDA environment on the 80 GB A100.
The workflow does not download data and does not start a long run implicitly.

## 1. Define leakage-safe contig splits

Create a JSON file using the contig identifiers exactly as they appear after
`>` in the B73 FASTA. The choice below is only a schema example, not a required
scientific split:

```json
{
  "train": ["chr1", "chr2"],
  "val": ["chr9"],
  "test": ["chr10"]
}
```

For multiple genomes, selectors may be qualified as `B73::chr1`. A contig
cannot be assigned to more than one split.

## 2. Prepare 10,240-bp windows and statistics

```bash
python scripts/prepare_maize_genome.py \
  --fasta /path/to/B73.fa \
  --genome-name B73 \
  --species "Zea mays" \
  --split-config /path/to/b73_splits.json \
  --unassigned-policy error \
  --output-dir /path/to/processed/maize_b73_10240 \
  --window-size 10240 \
  --stride 5120 \
  --ambiguity-policy map_to_n \
  --max-n-fraction 1.0 \
  --max-u-fraction 0.0 \
  --require-all-splits
```

Inspect both `manifest.json` and `DATA_STATS.md` in the output directory. Do
not continue if `formal_pretraining_ready` is false. To test non-overlapping
windows, change only `--stride 10240`.

For a single-contig Stage A subset, omit `--split-config` and
`--require-all-splits`; all windows go to train. Use the direct smoke script
below, not the full Hydra training loop, because validation/test will be empty.

## 3. Direct end-to-end smoke test

```bash
python scripts/smoke_test_dna.py \
  --data-dir /path/to/processed/maize_b73_10240 \
  --window-size 10240 \
  --steps 2 \
  --device cuda \
  --precision fp16
```

This uses a small 128 x 4 model but retains bidirectional Mamba, BCW, cross-layer
memory, the lightweight reader, MLM loss, backward, and an optimizer step. It
must print `"status": "PASS"`, finite losses, `[1, 10240]` input, aligned
logits, and peak GPU memory.

With non-empty train/val/test contig splits, the same path can be exercised
through the real trainer:

```bash
export MAIZE_DNA_INDEXED_DIR=/path/to/processed/maize_b73_10240
python train.py experiment=maize_dna_smoke
```

## 4. Full-size A100 memory-cost benchmark

```bash
python scripts/benchmark_dna.py \
  --data-dir /path/to/processed/maize_b73_10240 \
  --window-size 10240 \
  --batch-size 1 \
  --d-model 768 \
  --n-layer 12 \
  --mode both \
  --warmup-steps 2 \
  --steps 5 \
  --precision fp16 \
  --device cuda \
  --output-json /path/to/runs/maize_dna_benchmark.json
```

The two cases keep the same tokenizer, batch, bidirectional backbone, MLM head,
and optimizer. The only experimental switch is the BCW/memory sidecar. Review
step time, forward/backward time, tokens/s, peak memory, and overhead percentage.

## 5. Guarded full-size pretraining

`experiment=maize_dna_pretrain` uses the current production dimensions and
single-A100 layout, but training length and warmup are mandatory values. Supply
them only after the smoke and benchmark reports are accepted:

```bash
export MAIZE_DNA_INDEXED_DIR=/path/to/processed/maize_b73_10240
python train.py experiment=maize_dna_pretrain \
  trainer.max_steps=<approved_steps> \
  scheduler.warmup_t=<approved_warmup_steps>
```

The default is training from scratch. To evaluate an RNA checkpoint warm start,
first use a separate short run and record that token id 10 changes semantic
meaning from RNA U to DNA T.
