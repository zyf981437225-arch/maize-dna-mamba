# Repository cleanup audit

## Decision

The repository is now scoped to OneMaize genomic MLM training. Cleanup was performed only after reference searches across tracked source, configs, scripts, tests, and documentation.

| Area | Decision | Dependency check |
| --- | --- | --- |
| Caduceus model, bidirectional Mamba, memory, tokenizer | Keep | Required by all three formal routes |
| `train.py`, tasks, callbacks, optimizers, schedulers, utilities | Keep | Shared runtime; only m6A-specific task/metrics were removed |
| OneMaize schema-v3 builders, datasets, validators, benchmarks, checkpoint contracts | Keep | Required by B73/NAM26 workflows |
| schema-v4 variant/TE implementation | Keep as experimental | Isolated from the formal schema-v3 routes |
| RNA/m6A configs, loaders, datasets, scripts, tests, fixtures, docs | Delete | References formed an RNA-only dependency island; shared registries were cleaned |
| old random-window maize pilot configs/launchers/tests/docs | Delete | Superseded by exhaustive Phase-I and schema-v3 Phase-II |
| `.save` and backup model/data files | Delete | No runtime imports or test references |
| old Slurm/PBS wrappers | Delete where superseded | Formal teacher entrypoints are the three numbered PBS files |

## Deleted groups

- RNA pretraining and m6A fine-tuning: dataset/experiment/task configs; preprocessing, evaluation, plotting and launch scripts; RNA/m6A datasets, datamodules, metrics, tests and fixtures.
- Superseded maize pilot: `maize_dna_*`, old B73 8K/16K/smoke/pilot configs, generic old DNA dataset, A100/old H200 wrappers and pilot documentation.
- Repository debris: model/data `.save` files, backup model implementation, old architecture/RC audits, RNA t-SNE and local RNA smoke entrypoints.

## Shared-code edits

- Removed RNA dataset branches from `src/dataloaders/genomics.py` while retaining HG38/bacteria support.
- Removed the m6A task and metrics from shared registries while retaining language-model and generic downstream tasks.
- Removed the obsolete genomic-DNA batch loader from `scripts/dna_runtime.py`; its tokenizer/model/precision helpers remain used by OneMaize benchmarks.

## Learning-rate audit

Git history shows Model B was introduced by changing Phase-II LR from `8e-5`/`2e-5` to `4e-5`/`1e-5`, without a benchmark, result, or written rationale. Model B is therefore aligned with the B73 Phase-II continuation schedule: initial LR `8e-5`, minimum LR `2e-5`. Any later change must be recorded with an H200 benchmark or ablation.

## Boundary after cleanup

Formal routes are B73 exhaustive 8K Phase-I, B73 16K reference Phase-II, and NAM26 schema-v3 16K Phase-II. Variant-aware schema-v4 remains explicitly experimental and is not required by any formal PBS job.
