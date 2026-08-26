# OneMaize teacher-plan acceptance matrix

This matrix distinguishes implementation completion from experiments that
cannot be honestly completed with one B73 genome or without the target GPU.

| Teacher-plan item | Status | Evidence / remaining input |
|---|---|---|
| Independent DNA project; preserve bidirectional Mamba, BCW, memory bank, reader | Implemented | This repository is independent; the preserved model and memory source files were not rewritten for OneMaize. |
| Exact B73 + 25 NAM founder panel | Guard implemented; data pending | `--formal` validates all 26 names case-insensitively. Only B73 files are currently available. |
| Genotype split 23 train / 1 validation / 2 test; B73 in train | Guard implemented; held-outs pending | Builder and validator enforce the counts and B73 constraint. Diversity analysis still needs all founder data. |
| `genomes.parquet` and candidate-interval `regions.parquet` | Implemented and real-data tested | Schema v3 stores file paths, annotations, repeat fraction, N fraction, split, coordinates, and gene metadata without fixed sequence strings. |
| Gene-centered / non-repeat / TE-rich pools | Implemented and real-data tested | Protein-coding gene ±5 kb; TE union coverage threshold 50%; B73 contains all pools in all pilot splits. |
| Uniform genotype then 50% / 30% / 20% class sampling | Implemented and tested | Sampling is hierarchical. Runtime N-quality retries remain inside the selected genotype/class pool. |
| Dynamic 8,192 then 16,384 bp curriculum | Implemented and real-data tested | Indexed BGZF FASTA fetches both lengths from the same candidate metadata. |
| 50% training reverse-complement augmentation | Implemented | Validation/test use deterministic forward orientation; training uses configurable probability 0.5. |
| Single-base A/C/G/T/N tokenizer, no genotype token | Implemented and tested | DNA mode excludes U and genotype identity is never inserted into the token stream. |
| 15% MLM with 80/10/10 corruption and selected-position-only loss | Implemented and real-data tested | Real B73 8K/16K samples produced masking fractions near 15%; labels elsewhere use the loss ignore index. |
| FASTA alphabet and assembly-gap quality audit | Implemented and real-data tested | Full FASTA scan rejects non-ACGTN input; candidates above 10% N are removed; crops are rechecked dynamically. |
| Bidirectional model, approximately 24 blocks / 120M parameters | Implemented as preserved-backbone equivalent | Production width 864 × 24 analytically estimates 121,191,553 parameters. Actual A100 instantiation is the final gate. |
| Mamba2 | Intentional approved exception | The user required the current Mamba/BCW/memory modules to remain. The project therefore keeps its existing tied bidirectional Mamba implementation rather than silently changing kernels. |
| BF16 and Phase-0 throughput/memory/I/O measurements | Script and hard gates implemented; GPU run pending | `benchmark_onemaize.py` compares backbone-only and BCW/memory, recording forward/backward/step time, tokens/s, I/O, peak memory, loss, and actual parameters. |
| Small 30–50M versus base ~120M ablation | Configured; training pending | Pilot config estimates ~43.7M; production config estimates ~121.2M. |
| 8K Phase I followed by 16K Phase II checkpoint continuation | Configured and guarded; training pending | Launcher requires measured/approved steps and warmup; Phase II additionally requires a real Phase-I checkpoint. |
| B73-only / 5 / 13 / full population ablations | Formal loader supports them; data/training pending | Requires the missing founder FASTA/gene/TE files and a frozen split manifest. |
| Downstream genome-element, regulation, variation and trait evaluation | Data/model-results pending | The supplied three B73 files contain no ATAC, UMR, expression, SNP/SV/PAV, or phenotype labels; no biological claim can yet be evaluated. |

## Current executable boundary

With the supplied B73 FASTA, gene GFF3, and TE GFF3, the project can validate
the complete data engineering and MLM path and can run a guarded B73 pilot on
the target A100. It must not be described as a completed OneMaize population
foundation model until all 26 founders, the genotype-held-out split, the A100
Phase-0 report, training checkpoints, and downstream evaluation data exist.
