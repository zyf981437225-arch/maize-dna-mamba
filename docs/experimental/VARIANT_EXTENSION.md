# Experimental variant/TE extension

Schema-v4 explicit variant-aware sampling is retained for future ablation only. It is not part of the formal B73 or NAM26 schema-v3 workflow and no formal PBS job depends on it.

Implementation entrypoints are `scripts/build_onemaize_variant_metadata.py`, `scripts/validate_onemaize_variant_te.py`, `scripts/audit_onemaize_variant_te.py`, and `scripts/run_onemaize_variant_te_phase2_h200.sh`. Supporting configuration and audits remain under `configs/*onemaize*variant*` and `docs/audits/onemaize_variant_te/`.

Do not run this route until the required normalized per-genotype variant inputs are available and its audit gates pass. Results from this extension must be labelled experimental and compared against the formal schema-v3 Model B baseline.
