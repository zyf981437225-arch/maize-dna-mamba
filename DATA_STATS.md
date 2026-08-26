# OneMaize B73 data audit

This report summarizes the supplied B73 NAM v5 files used by the current
annotation-aware pilot. Formal OneMaize readiness remains false because the
other 25 founder genomes are not present.

## Genome and annotation totals

| Metric | B73 chr1–chr10 |
|---|---:|
| Total bp | 2,131,846,805 |
| Sequences | 10 |
| Protein-coding genes | 39,035 |
| TE union bp | 1,820,540,623 (85.3973%) |
| A | 565,533,532 (26.5279%) |
| C | 498,060,568 (23.3629%) |
| G | 498,293,478 (23.3738%) |
| T | 566,151,341 (26.5568%) |
| N | 3,807,886 (0.1786%) |
| Invalid symbols | 0 (full FASTA alphabet audit passed) |

## Candidate-region totals

- Coordinate system: zero-based, half-open.
- Contexts: dynamic 8,192 bp and 16,384 bp.
- Gene candidates: protein-coding body plus 5 kb upstream/downstream.
- Genome-wide candidate span/stride: 32,768 / 16,384 bp.
- TE-rich threshold: repeat union coverage at least 50%.
- Maximum retained candidate N fraction: 10%.
- Total retained candidates: 164,150.

| Pilot split | Gene-centered | Non-repeat | TE-rich | Total |
|---|---:|---:|---:|---:|
| train (chr1–chr8) | 33,329 | 101 | 106,505 | 139,935 |
| validation (chr9) | 2,988 | 40 | 9,507 | 12,535 |
| test (chr10) | 2,704 | 2 | 8,974 | 11,680 |

Real indexed FASTA validation passed for both context lengths. Across 16
fetched samples per split and context, the selected-position fraction remained
near the configured 15% MLM target. The generated machine-readable authority
is `data/processed/onemaize_b73/manifest.json`; generated data files are kept
outside version control.
