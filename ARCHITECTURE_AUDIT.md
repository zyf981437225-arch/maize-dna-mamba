# Maize DNA-Mamba architecture audit

## Scope and provenance

Maize DNA-Mamba is an independent genomic-DNA project derived from the
Caduceus/RNA-Mamba framework at baseline commit `172f304`. The reusable model,
trainer, task, optimizer, and cross-layer memory components are retained; the
supported data path and public workflow are DNA-only.

Target execution hardware is one NVIDIA A100 with 80 GB memory. No formal
training or external genome download was performed while building the project.

## Reused model path

The training flow is:

1. Hydra composes a maize DNA experiment, indexed DNA dataset, MLM task,
   Caduceus model, optimizer, scheduler, and callbacks.
2. `train.py` creates `SequenceLightningModule`, initializes the selected data
   module, transfers the tokenizer complement map into model configuration,
   and constructs `CaduceusForMaskedLM`.
3. `src/dataloaders/genomic_dna_mlm.py` creates the DNA tokenizer and the
   train/validation/test memory-mapped datasets.
4. `src/dataloaders/datasets/genomic_dna_dataset.py` maps A/C/G/T/N bytes to
   single-nucleotide ids and performs 15% same-position MLM. A/C/G/T are
   prediction targets; A/C/G/T/N are eligible random replacements.
5. `src/tasks/tasks.py` keeps logits and targets aligned and computes
   pad-id-ignored cross entropy.
6. `caduceus/modeling_caduceus.py` runs embeddings, bidirectional Mamba layers,
   selective BCW memory writes, earlier-memory reads, gated residual injection,
   final normalization, and the MLM head.

Production dimensions are 12 layers, hidden size 768, vocabulary size 12,
window length 10,240, batch size 1 with gradient accumulation, and MLM
probability 0.15. Memory uses `d_sum=64`, `d_mem=64`, write stride 6, read
stride 2, and no cross-batch persistence.

## Genomic DNA boundary

The supported data flow is:

`FASTA -> streaming contig parser -> whole-contig split assignment -> ACGTN
normalization -> fixed genomic windows -> indexed binary corpus -> manifest
and DATA_STATS.md -> deterministic/evolving MLM -> model`.

The implementation enforces these properties:

- contigs are streamed independently and are never concatenated;
- window size and stride are independent configuration values;
- split assignment is performed on complete contigs, with optional
  `genome::contig` selectors for multi-genome inputs;
- A/C/G/T/N, U, IUPAC ambiguity, and invalid characters are counted before
  normalization;
- ambiguity handling is `map_to_n`, `filter_window`, or `error`;
- every retained sample has exactly the declared length and only A/C/G/T/N;
- raw U above the declared tolerance prevents training;
- the loader validates schema, sizes, final offsets, alphabet, and readiness;
- training corruption changes across accesses while validation/test corruption
  is deterministic for reproducible evaluation.

The older generic hg38 loader is not used: it assumes 2^20 interval
partitioning, treats N differently, and does not meet the 10,240-bp maize data
contract.

## Length and memory audit

The backbone does not use learned absolute positional embeddings or a fixed
1,024-position reshape. BCW mean-pools over the runtime sequence length, and
the reader produces `[B, 1, D]`, which broadcasts across `[B, 10240, D]`.
Therefore 10,240 bp is shape-compatible with the model and memory sidecar.

Shape compatibility is not a performance claim. The A100 smoke and benchmark
scripts remain required because activation memory and kernel behavior must be
measured in the real CUDA/Mamba environment.

## Bidirectionality and reverse complement

`BiMambaWrapper` processes forward and reversed hidden-state streams. It does
not complement nucleotide ids and therefore does not prove strict
reverse-complement equivariance. Strict RCPS components exist in
`caduceus/modeling_rcps.py`, but the model explicitly rejects `rcps=true` with
`use_memory=true` because the current BCW/reader sidecar is not RC-equivariant.

The production configuration consequently uses:

- `bidirectional=true`;
- `bidirectional_strategy=add`;
- `bidirectional_weight_tie=true`;
- `rcps=false`;
- `use_memory=true`.

This project makes no strict-RC claim. See `RC_MEMORY_COMPATIBILITY.md` for the
module-level classification and repair proposal.

## Checkpoint compatibility

The DNA vocabulary contains seven special tokens followed by A/C/G/T/N, so
the logical size remains 12 and Caduceus pads it to 16 internally. A/C/G retain
ids 7/8/9, T occupies id 10, and N is id 11. An architecture-matched non-RCPS
checkpoint is therefore shape-compatible with the backbone, memory modules,
embedding, and MLM head.

For an RNA-derived checkpoint, row 10 changes meaning from U to T. This is a
semantic remapping, not transparent equivalence. Training from scratch is the
default; any warm start must use a short isolated run and report the remapping.

## Verification gates

Completed locally:

- tokenizer DNA-only contract and token ids;
- reverse-complement involution;
- contig-safe window generation and split isolation;
- deterministic evaluation MLM and same-position alignment;
- U-frequency failure gate and generated statistics;
- reader broadcasting over 10,240 positions;
- Hydra composition and Python syntax checks.

Required on the server before formal pretraining:

1. generate real B73 `manifest.json` and `DATA_STATS.md`;
2. run `scripts/smoke_test_dna.py` on the A100 and require `status=PASS`;
3. run `scripts/benchmark_dna.py` and review memory/time overhead;
4. decide approved training steps and warmup;
5. only then compose `experiment=maize_dna_pretrain` with explicit values.
