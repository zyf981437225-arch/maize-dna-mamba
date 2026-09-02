#!/usr/bin/env python
"""Audit schema-v3 candidates and schema-v4 explicit variant metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.onemaize.variant_audit import audit_variant_metadata, write_audit_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--variant-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--fasta-root", type=Path)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    report, rows = audit_variant_metadata(
        args.base_data_dir,
        args.variant_data_dir,
        context_length=args.context_length,
        fasta_root=args.fasta_root,
        formal=args.formal,
    )
    write_audit_outputs(args.output_dir, report, rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
