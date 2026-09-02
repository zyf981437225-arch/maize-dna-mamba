#!/usr/bin/env python
"""Audit formal all-cultivar schema-v3 metadata for Model B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.onemaize.population_audit import audit_population_metadata, write_population_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fasta-root", type=Path)
    parser.add_argument("--context-length", type=int, default=16_384)
    parser.add_argument("--low-pool-warning", type=int, default=100)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    report, rows = audit_population_metadata(
        args.data_dir,
        context_length=args.context_length,
        fasta_root=args.fasta_root,
        formal=args.formal,
        low_pool_warning=args.low_pool_warning,
    )
    write_population_audit(args.output_dir, report, rows)
    print(f"status={report['status']} output={args.output_dir}")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
