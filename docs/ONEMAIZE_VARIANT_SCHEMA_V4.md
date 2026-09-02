# OneMaize variant metadata schema-v4

> Experimental explicit variant-aware extension. This schema is implemented but
> is not required for the current formal schema-v3 all-cultivar Model B.

Schema-v4 is an independent metadata layer. It does not replace or modify schema-v3.

## Files

```text
schema-v3 directory/
├── manifest.json
├── genomes.parquet
└── regions.parquet

schema-v4 directory/
├── variant_manifest.json
└── variant_regions.parquet
```

## Coordinate contract

- Internal coordinates are 0-based half-open.
- VCF `POS`/`END` conversion occurs only in `src/onemaize/variants.py`.
- `coordinate_genotype` must equal `genotype` before an event can enter training.
- A zero-length `[start, start)` interval represents an insertion boundary.
- B73-reference coordinates must not index another cultivar's assembly FASTA without a validated mapping.

## `variant_regions.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `variant_id` | string | Globally unique event/ALT identifier |
| `genotype` | string | Cultivar sequence used for the model input |
| `reference_genotype` | string | Comparison reference, normally B73 |
| `coordinate_genotype` | string | FASTA whose coordinates are used |
| `seqid` | string | FASTA sequence identifier |
| `start`, `end` | int64 | 0-based half-open event interval/boundary |
| `variant_type` | string | `snp`, `indel`, `deletion`, `insertion`, `inversion`, `duplication`, `sv`, `pav`, `te_insertion`, or `te_deletion` |
| `reference_allele`, `alternate_allele` | nullable string | Alleles when meaningful |
| `variant_length` | int64 | Signed indel length; interval length for other events |
| `source` | string | Caller/dataset provenance |
| `split` | string | Genotype-derived `train`, `val`, or `test` |
| `left_breakpoint`, `right_breakpoint` | nullable int64 | Explicit SV breakpoints when available |
| `te_family`, `te_superfamily`, `te_id`, `te_class` | nullable string | TE metadata when supplied by a real source |
| `reference_presence`, `alternate_presence` | nullable bool | PAV/TE presence state when supplied |

Nullable TE/PAV fields are not inferred. Missing values mean unavailable annotation, not biological absence.

## Sampling mapping

| Sampling class | Events |
| --- | --- |
| `small_variant` | SNP/indel/insertion/deletion up to the configured length threshold |
| `structural_variant` | longer insertion/deletion, inversion, duplication, generic SV and PAV |
| `te_variant` | explicit `te_insertion` and `te_deletion` only |

An event that fits within 16K is fully retained. A longer structural event is represented by a deterministically selected left- or right-breakpoint context. Jitter is clamped to the feasible crop interval, so the target never disappears from the model input.
