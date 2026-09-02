# OneMaize training details

## Fixed model contract

All formal stages use the existing `mamba_ssm.modules.mamba_simple.Mamba` Caduceus implementation, not Mamba2. The configuration is 24 layers, `d_model=864`, tied bidirectional weights, approximately 121.2M parameters, BF16, single-base `A/C/G/T/N` tokens, 15% MLM with 80/10/10 corruption, and training-only reverse-complement probability 0.5.

## Formal budgets on one 8×H200 node

| Stage | Context | Batch/GPU | Grad accumulation | Optimizer steps | Warmup | LR → minimum |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B73 Phase-I | 8,192 | 1 | 1 | 325,300 | 16,265 | `8e-5` → `2e-5` |
| B73 Phase-II | 16,384 | 1 | 8 | 15,630 | 782 | `8e-5` → `2e-5` |
| NAM26 Phase-II | 16,384 | 1 | 8 | 15,630 | 782 | `8e-5` → `2e-5` |

Phase-I has 260,239 unique fixed windows. With world size 8, `DistributedSampler` draws 32,530 samples per rank, so one full pass is 32,530 optimizer steps and ten passes are 325,300 steps. The single padding duplicate per pass is expected.

Phase-II defines one virtual epoch as 100,000 sampled sequences. Effective global batch is `8 GPUs × 1 × 8 accumulation = 64`; `ceil(100000/64)=1563` optimizer steps per virtual epoch, hence 15,630 steps for ten virtual epochs.

## Checkpoint semantics

- New Phase-II runs use `train.pretrained_model_path`: model weights load strictly from the B73 Phase-I best checkpoint; optimizer and scheduler start fresh.
- `RESUME_CKPT` is only for exact continuation of the same stage and restores model, optimizer, scheduler, and global step.
- Do not set initialization and resume checkpoints together.
- Best checkpoints are selected by finite minimum `val/loss`; periodic `last.ckpt` supports recovery.

## Sampling contract

Phase-I is deterministic, exhaustive, `window=stride=8192`, and pads ten chromosome tails. Phase-II first selects genotype (uniformly for NAM26), then class: 50% gene-centered, 30% non-repeat, 20% TE-rich. A dynamic 16,384-bp crop is drawn from each compatible candidate.

## Validation and stopping

The PBS jobs run validation and smoke gates before formal optimization. During training, monitor finite train/validation loss, checkpoint timestamps, GPU memory/utilization, and output filesystem space. Stop and investigate on NaN/Inf, traceback, stalled checkpoints, or low disk space. Final comparison uses held-out validation/test loss with the same context and sampling contract.

## Runtime implementation

The numbered PBS scripts are the public operator interface. Their internal launchers are:

- `scripts/run_onemaize_phase1_h200.sh`
- `scripts/run_onemaize_b73_phase2_h200.sh`
- `scripts/run_onemaize_allcultivar_phase2_h200.sh`

Developers may invoke launcher stages directly for diagnosis, but formal runs should use the PBS entrypoints so paths and budgets remain frozen.
