# Five-million-sequence RNA-Mamba pretraining

This document is the authoritative recipe for the next pretraining run. The
scope is pretraining only: three million non-coding RNA records and
approximately two million coding-RNA records. The m6A tables in the delivery
are deliberately reserved for later fine-tuning.

## 1. Delivered data and the final corpus contract

The sources are the original delivery archive plus the newly supplied mouse
master, mask and exon-map files. The archive's local display name is not part
of the training contract. `scripts/organize_rna_data.py` converts all inputs
once into the canonical English directory below.

| Source | Delivered content | Role in the 5M corpus |
| --- | --- | --- |
| RNAcentral selection | exactly 3,000,000 records selected from 40,712,942 active RNAcentral records; seed 2357; lengths 18--10,240 nt | 3M non-coding RNA records |
| Eukaryote archives | 45 species and 1,532,392 protein-coding full-mRNA transcript records, with matching CDS/UTR FASTA files | primary coding records |
| Prokaryote archive | 51 species and 149,969 coding records, primarily CDS | primary coding records |
| Human transcript master | 211,446 full-transcript rows with `5'UTR + CDS + 3'UTR` sequences and coordinates | primary coding records |
| Mouse transcript master | 59,294 full-transcript rows, 21,807 genes and 144,128,429 nt; all IDs agree with its exon table | primary coding records |
| Human/mouse m6A tables and m6A archive | nucleotide modification labels; the mouse mask covers 48,352 master transcripts | **not used in pretraining**; retained for fine-tuning |

```text
rna_mamba_data/
  pretraining/
    non_coding_rna/
      rnacentral_active_mRNA100_priority_3M.fasta.gz
      rnacentral_active_mRNA100_priority_3M.ids.txt
      rnacentral_active_mRNA100_priority_3M.summary.json
    coding_rna/
      eukaryote/eukaryote_mRNA_dataset_part1.zip ... part4.zip
      prokaryote/prokaryote_mRNA_dataset.zip
      human/human_transcript_master.csv.gz
      mouse/mouse_transcript_master.csv.gz
  finetuning/m6a/
    human/human_m6a_nt_mask_full_mrna.csv.gz
    human/human_exon_coordinate_map.csv.gz
    mouse/mouse_m6a_nt_mask_full_mrna.csv.gz
    mouse/mouse_exon_coordinate_map.csv.gz
    multispecies/m6A_modification_dataset.zip
  manifests/source_inventory.json
  documentation/delivery_readme.md
```

The exact teacher-provided basenames are preserved. Only the directory layout
is standardized. The obsolete mouse `.textClipping` placeholder is never
copied.

The independent coding sources contain 1,953,101 raw records before filtering:
1,532,392 eukaryotic full mRNAs, 211,446 human full mRNAs, 59,294 mouse full
mRNAs and 149,969 prokaryotic coding records. The preparation code uses this
explicit policy:

1. keep valid unique eukaryotic, human and mouse full mRNA plus prokaryotic CDS
   as primary coding records;
2. exclude invalid, duplicate and out-of-range records without replacing them
   with repeated views of the same transcript;
3. use at most 2M coding records and combine them with the delivered 3M
   RNAcentral records;
4. write the exact source type of every record to compressed metadata.

Consequently, "2M coding" is a rounded planning label. The exact usable count
after quality control is recorded in `manifest.json`; it must not be reported
as two million independent full-length mRNAs unless the manifest actually
contains that count.

All sequences are upper-cased, `T` is converted to `U`, the accepted alphabet
is `A/U/C/G/N`, and lengths outside 18--10,240 nt are excluded. Sequences are
never truncated. Primary coding records are content-deduplicated across all
sources. The already curated RNAcentral 3M record selection is preserved at
accession level. An optional CDS-view filler exists only for a controlled
ablation and is disabled in the formal recipe.

## 2. Why the indexed format is required

The older data loader stored every RNA sequence as a Python string. With five
million records and eight DDP processes, that would duplicate the corpus in
host memory eight times. The new preparation script writes:

```text
rna_pretraining_5m/
  manifest.json
  train.sequences.bin
  train.offsets.u64
  train.sources.u8
  train.records.tsv.gz
  val.*
  test.*
```

Each worker memory-maps sequence bytes and offsets on demand. The loader keeps
random access for distributed sampling but does not load all sequence strings
into RAM. The metadata files preserve source class, source type, species,
record identifier, and length for every example.

## 3. Organize and prepare the data once

Use a fast local or parallel filesystem with at least 40 GB of temporary free
space and at least 30 GB for the final prepared data. Actual usage depends on
the sequence-length distribution. Do not prepare the archive independently in
multiple distributed ranks.

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

DELIVERY_ARCHIVE=/path/to/source_delivery.zip
MOUSE_MASTER=/path/to/mouse_transcript_master.csv.gz
MOUSE_MASK=/path/to/mouse_m6a_nt_mask_full_mrna.csv.gz
MOUSE_EXON=/path/to/mouse_exon_coordinate_map.csv.gz
SOURCE_DIR=/path/to/rna_mamba_data
DATA_DIR=/path/to/data/processed/rna_pretraining_5m
TEMP_DIR=/path/to/fast-temporary-storage

python scripts/organize_rna_data.py \
  --delivery-archive "$DELIVERY_ARCHIVE" \
  --mouse-transcript-master "$MOUSE_MASTER" \
  --mouse-m6a-mask "$MOUSE_MASK" \
  --mouse-exon-map "$MOUSE_EXON" \
  --output-dir "$SOURCE_DIR" \
  2>&1 | tee /path/to/organize_rna_data.log

python scripts/prepare_pretraining_5m.py \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$DATA_DIR" \
  --temp-dir "$TEMP_DIR" \
  2>&1 | tee /path/to/prepare_pretraining_5m.log
```

Preparation is transactional. Files are first created under
`rna_pretraining_5m.building` and moved into place only after every count and
file-size check succeeds. Pass `--overwrite` only when intentionally replacing
an existing prepared corpus.

Audit the result before requesting GPUs:

```bash
python - "$DATA_DIR/manifest.json" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["schema_version"] == 2
assert d["totals"]["source_class_counts"]["ncRNA"] == 3_000_000
assert 1_500_000 <= d["totals"]["source_class_counts"]["coding"] <= 2_000_000
assert d["corpus_contract"]["coding_cds_view_fillers"] == 0
assert d["corpus_contract"]["m6a_used"] is False
print(json.dumps(d["corpus_contract"], indent=2))
print(json.dumps(d["splits"], indent=2))
print("5M planning-corpus data audit: PASS")
PY
```

The default split is stable 98%/1%/1% assignment by biological record. It
produces approximately 4.85M training records if all 1.95M coding records pass
quality control. The exact
counts are recorded in `manifest.json`; training never relies on an estimated
count.

## 4. Preflight test

Install the environment described in the main README, then run:

```bash
python -m pytest -q \
  caduceus/tests \
  tests/test_full_transcript_mlm.py \
  tests/test_indexed_rna_pretraining.py \
  tests/test_checkpoint_and_collate_contracts.py
```

Run a short one-GPU smoke job before allocating an eight-GPU node:

```bash
export RNA_PRETRAIN_INDEXED_DIR="$DATA_DIR"

CUDA_VISIBLE_DEVICES=0 python -m train \
  experiment=rna_5m_pretrain \
  trainer.devices=1 \
  trainer.max_steps=20 \
  trainer.accumulate_grad_batches=1 \
  dataset.max_train_sequences=100 \
  dataset.max_val_sequences=16 \
  dataset.max_test_sequences=16 \
  callbacks.model_checkpoint.save_top_k=0 \
  callbacks.periodic_checkpoint.save_top_k=0 \
  callbacks.model_checkpoint_every_n_steps.save_top_k=0 \
  train.test=false \
  wandb=null \
  hydra.run.dir=runs/rna_5m_smoke
```

## 5. Formal eight-GPU launch

The formal default is a from-scratch 12-layer, 768-dimensional RNA-Mamba with
about 49M trainable parameters, character-level MLM, 15% masking, a 10,240-nt
maximum sequence length, FP16, and the current lightweight BCW/cross-layer
memory configuration. m6A labels are not inputs or targets in this stage.

| Item | Formal setting |
| --- | --- |
| Backbone | 12-layer, 768-dimensional weight-tied BiMamba |
| Parameters | approximately 49M trainable parameters |
| Lightweight memory | enabled; write stride 6, read stride 2 |
| Memory dimensions | summary 64, memory 64, 4 heads, maximum 32 slots |
| Cross-batch state | disabled (`memory_persist_across_batches=false`) |
| Objective | same-position character MLM; 15% eligible bases selected |
| Maximum length | 10,240 nt; dynamic right padding; no truncation |
| Precision | FP16 |
| Optimizer | AdamW, LR `8e-5`, weight decay `0.01`, betas `[0.9, 0.98]` |
| Scheduler | cosine decay over the exact run, 20k warmup steps, minimum LR `2e-5` |
| Default global batch | 16 sequences (`8 GPUs x 1 x accumulation 2`) |
| Initial stopping plan | 2 complete training-corpus passes; best validation-loss checkpoint |

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

export RNA_PRETRAIN_INDEXED_DIR=/path/to/data/processed/rna_pretraining_5m
export RUN_DIR=/path/to/runs/rna_pretraining_5m
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_DEVICES=8
export BATCH_SIZE=1
export GRAD_ACCUM=2
export NUM_WORKERS=4
export PRETRAIN_EPOCHS=2

bash scripts/run_pretrain_5m_8gpu.sh
```

The launcher calculates the optimizer-step target from the actual manifest:

```text
global batch = GPU count x per-GPU batch x gradient accumulation
steps per epoch = ceil(training records / global batch)
max steps = steps per epoch x PRETRAIN_EPOCHS
```

For an approximately 4.85M-record training split and global batch 16, this is
about 303,000 optimizer steps per epoch and 606,000 steps for two epochs. If
quality control leaves exactly five million total records, the corresponding
values are about 306,250 and 612,500 steps. The exact values printed by the
launcher take precedence. The old 50,000-step setting is only about 0.16 epoch
on this corpus and is not a complete training pass.

Two epochs are the initial compute plan, not a claim that convergence is
guaranteed. Select the best `val/loss` checkpoint. Add a third epoch only if
the second-epoch validation loss is still improving materially. Do not choose
the last checkpoint merely because it is later.

## 6. Time and resource planning

The teacher's earlier eight-H200 run on the much smaller mixed corpus took
about three hours for 50,000 optimizer steps. A purely linear extrapolation is
approximately 36--37 hours for roughly 606,000--612,500 steps. That is a planning estimate, not a
measurement of the new corpus: the new length distribution, storage bandwidth,
padding, and validation set can change throughput.

Use the first 2,000--5,000 steps of the formal job to record measured seconds
per optimizer step, then estimate:

```text
remaining wall time = remaining optimizer steps x measured seconds per step
```

The launcher writes the Git commit, GPU inventory, exact record counts, global
batch, epoch-derived step target, scheduler length, and resume choice to
`run_manifest.txt`. `/usr/bin/time` writes measured total runtime to
`time.txt`.

Recommended minimum allocation for preparation and training:

- one node with 8 H200 GPUs;
- 128 GB host RAM (the indexed loader itself needs far less, but preparation
  uses hash sets and the operating system benefits from filesystem cache);
- at least 100 GB free run/data space for prepared data, logs, and recovery
  checkpoints;
- local NVMe or a high-throughput parallel filesystem.

## 7. Resume and required outputs

To resume an interrupted run exactly, restore model, optimizer, scheduler and
step state from the last checkpoint:

```bash
export RESUME_CKPT=/path/to/runs/rna_pretraining_5m/checkpoints/last.ckpt
bash scripts/run_pretrain_5m_8gpu.sh
```

Keep these outputs:

```text
data/processed/rna_pretraining_5m/manifest.json
runs/rna_pretraining_5m/run_manifest.txt
runs/rna_pretraining_5m/console.log
runs/rna_pretraining_5m/time.txt
runs/rna_pretraining_5m/checkpoints_best/val_loss.ckpt
runs/rna_pretraining_5m/checkpoints/last.ckpt
```

Fine-tuning is intentionally a separate later decision. At that point the m6A
tables can be processed and the best pretraining checkpoint loaded into the
nucleotide-level classification model.
