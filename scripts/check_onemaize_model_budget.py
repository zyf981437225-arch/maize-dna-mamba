#!/usr/bin/env python
"""Check preserved OneMaize model dimensions against the teacher-plan budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.onemaize.model_budget import estimate_caduceus_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=864)
    parser.add_argument("--n-layer", type=int, default=24)
    parser.add_argument("--minimum", type=int, default=110_000_000)
    parser.add_argument("--maximum", type=int, default=130_000_000)
    args = parser.parse_args()
    result = estimate_caduceus_parameters(
        d_model=args.d_model,
        n_layer=args.n_layer,
    )
    result["minimum"] = args.minimum
    result["maximum"] = args.maximum
    result["target_met"] = args.minimum <= result["total"] <= args.maximum
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["target_met"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
