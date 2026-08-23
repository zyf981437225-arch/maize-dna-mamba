# Formal 8-GPU RNA-Mamba Training

This is the authoritative cluster recipe for the current RNA-Mamba model and
the complete-transcript m6A task. It supersedes the older 1-GPU and 2-GPU
development examples in the README when an 8-GPU node is available.

## What is trained

The production workflow has three stages:

1. masked-language-model (MLM) pretraining of RNA-Mamba on complete RNA
   sequences up to 10,240 nt;
2. full-model fine-tuning for nucleotide-level m6A classification on complete
   `5'UTR + CDS + 3'UTR` transcripts;
3. validation-calibrated, gene-disjoint test evaluation.

The m6A target is defined only at adenosines: `1` is methylated, `0` is
unmethylated, and non-A positions are excluded from the loss. Transcript
regions are combined during training and are separated only for stratified
evaluation.

## Validated configuration

| Item | Setting |
| --- | --- |
| Backbone | 12-layer, 768-dimensional weight-tied BiMamba |
| Lightweight memory | BCW writer plus forward-local cross-layer memory |
| Memory write/read strides | 6 / 2 |
| Memory dimensions | summary 64, slot 64 |
| Cross-batch memory | disabled |
| Vocabulary | character-level RNA, 12 tokens |
| Maximum input length | 10,240 nt |
| MLM masking probability | 0.15 |
| Precision | FP16 |
| Optimizer | AdamW, LR `8e-5`, weight decay `0.01`, betas `[0.9, 0.98]` |
| Pretraining global batch | 16 complete sequences |
| Pretraining stop | 50,000 optimizer steps, about 15.7 epochs |
| Fine-tuning optimizer | AdamW, LR `2e-5`, weight decay `0.01` |
| Fine-tuning global batch | 16 complete transcripts |
| Fine-tuning stop | 1 epoch, about 2,460 optimizer steps |
| Model selection | minimum MLM `val/loss`; maximum m6A `val/m6a_average_precision` |

An optimizer step is counted after gradient accumulation. On eight GPUs, use
one sequence per GPU and two accumulation steps:

```text
global batch = 8 GPUs x 1 sequence/GPU x 2 accumulation steps = 16
```

Do not leave accumulation at 16 on eight GPUs. That would increase the global
batch to 128 and make 50,000 steps an eight-times-larger experiment.

## Data preparation

Place the three source tables anywhere accessible to the compute node and run:

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

SOURCE_DIR=/path/to/source_tables
M6A_DATA=/path/to/processed/human_m6a_full_transcript
mkdir -p "$M6A_DATA"

python scripts/prepare_m6a_nucleotide.py \
  --transcript-master "$SOURCE_DIR/transcript_master.csv.gz" \
  --mask-table "$SOURCE_DIR/m6a_nt_mask_full_mrna.csv.gz" \
  --exon-map "$SOURCE_DIR/exon_coordinate_map.csv.gz" \
  --output-dir "$M6A_DATA" \
  2>&1 | tee "$M6A_DATA/prepare.log"

cat "$M6A_DATA/stats.json"
```

The preparation audit must report all of the following before training:

- `schema_version: 2`;
- `transcript_sequence = 5'UTR + CDS + 3'UTR`;
- zero genes shared across train, validation, and test;
- reliable transcript coordinates and CDS boundaries for retained examples;
- identical sequence and m6A-mask lengths.

The validated processed dataset contained:

| Split | Eligible transcripts | Candidate A | Positive m6A |
| --- | ---: | ---: | ---: |
| Train | 39,349 | 29,849,887 | 1,290,814 |
| Validation | 4,805 | 3,676,317 | 160,032 |
| Test | 4,723 | 3,629,334 | 158,602 |

These are the records retained after reliability checks and the 10,240-nt
length cap, not the raw source-table counts.

## Launch inside an 8-GPU allocation

The portable launcher runs pretraining, fine-tuning, and calibrated evaluation
sequentially. It is intended to be called from an interactive allocation or a
site-specific Slurm/PBS wrapper.

```bash
cd /path/to/mamba-for-RNA
conda activate rna-mamba

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_DEVICES=8
export M6A_DATA=/path/to/processed/human_m6a_full_transcript
export RNA_FASTA_FILE=/path/to/rnacentral_small_ATCG_only.fasta
export RUN_ROOT=/path/to/runs/formal_8gpu

bash scripts/run_formal_8gpu.sh all
```

The default is a clean from-scratch long-context pretraining run. To continue
from the established 1,024-nt MLM model instead, set:

```bash
export PRETRAIN_INIT_CKPT=/path/to/rna100_lightweight/checkpoints_best/val_loss.ckpt
bash scripts/run_formal_8gpu.sh all
```

Continued pretraining currently gives the best downstream AP, while the clean
from-scratch run is the appropriate independent ablation. Individual stages
can also be launched or resumed separately:

```bash
bash scripts/run_formal_8gpu.sh pretrain

export PRETRAIN_CKPT=/path/to/pretrain/checkpoints_best/val_loss.ckpt
bash scripts/run_formal_8gpu.sh finetune

export FINETUNE_CKPT=/path/to/finetune/checkpoints_best/val_m6a_ap.ckpt
bash scripts/run_formal_8gpu.sh evaluate
```

Use `train.ckpt=/path/to/checkpoints/last.ckpt` only for an exact interrupted
pretraining resume, because it restores the optimizer, scheduler, epoch, and
step. Use `train.pretrained_model_path` for a new fine-tuning stage.

The launcher exposes the exact-resume path as:

```bash
export PRETRAIN_RESUME_CKPT=/path/to/pretrain/checkpoints/last.ckpt
bash scripts/run_formal_8gpu.sh pretrain
```

Do not set `PRETRAIN_INIT_CKPT` at the same time. The launcher writes the Git
commit, GPU inventory, data paths, global batch, and initialization/resume
choice to `RUN_ROOT/run_manifest.txt` for the final resource report.

## Resource estimate

The current single-A100 measurement reached 20,000 long-context optimizer steps
in 8 h 56 min. At the same global batch, 50,000 steps are approximately 22--23
single-A100 hours. Expected wall time on one 8 x A100 node is 3.5--5 hours for
pretraining plus 15--25 minutes for one full m6A epoch. Reserve 4--6 hours for
the complete workflow, including validation, checkpoint I/O, and evaluation.

These 8-GPU times are planning estimates rather than measured cluster
benchmarks. Run a short throughput test on the target node if its GPU model,
interconnect, or storage differs. Checkpoints are about 579 MB each; the
launcher keeps the best model, a last checkpoint, and periodic recovery
checkpoints under the chosen run directory.

## Current results

All results below use the same gene-disjoint test set. The decision threshold
is selected on validation only and then frozen before test evaluation. Average
precision (AP) is the primary metric because only about 4.37% of candidate
adenosines are positive.

| Initialization before m6A fine-tuning | Test AP | AUROC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original full-mRNA baseline | 0.6924 | 0.9839 | 0.6242 | 0.7623 | 0.6864 |
| Scratch long-context MLM, 20k steps | 0.7143 | 0.9854 | 0.6346 | 0.7801 | 0.6999 |
| Scratch long-context MLM, 50k steps | 0.7179 | 0.9854 | 0.6364 | 0.7679 | 0.6960 |
| Continued long-context MLM | **0.7259** | **0.9856** | 0.6351 | 0.7724 | 0.6970 |

For scratch pretraining, validation MLM loss improved from 1.0429 near 20k
steps to a best value of 1.0330 at step 47,851. Extending the same run to 70k
steps did not improve the best checkpoint and the displayed validation loss
rose to about 1.09. The formal ceiling is therefore 50k optimizer steps with
best-checkpoint selection, not 70k.

Increasing scratch pretraining from 20k to 50k improved overall test AP from
0.7143 to 0.7179. Its largest gain was on transcripts longer than 8,192 nt,
where AP increased from 0.6535 to 0.6775. Continued pretraining remains the
recommended initialization for the best current downstream result.

## Required outputs

Retain these artifacts from the formal run:

```text
pretrain/checkpoints_best/val_loss.ckpt
pretrain/checkpoints/last.ckpt
pretrain/console.log
pretrain/time.txt
run_manifest.txt
finetune/checkpoints_best/val_m6a_ap.ckpt
finetune/console.log
finetune/time.txt
finetune/calibrated_evaluation/m6a_calibrated_evaluation.json
finetune/calibrated_evaluation/rna_mamba_m6a_calibrated_evaluation.{png,pdf,svg}
```

The final report must identify the Git commit, GPU count and model, input cap,
global batch, optimizer steps, best checkpoint step, wall time, test AP/AUROC,
and the validation-selected threshold.
