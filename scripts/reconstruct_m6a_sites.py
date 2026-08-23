#!/usr/bin/env python
"""Recover sparse per-adenosine m6A scores from overlapping window counts.

Use ``--mode oracle`` first to test whether the chosen window geometry can
recover observed sites from exact counts.  ``--mode model`` replaces exact
counts with predictions from a fine-tuned RNA-Mamba checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.m6a_sparse import (  # noqa: E402
    WindowCandidateSystem,
    average_precision,
    observed_site_metrics,
    solve_nonnegative_lasso,
)


def read_records(path: Path, max_transcripts: Optional[int]) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_transcripts is not None and index >= max_transcripts:
                break
            if line.strip():
                yield json.loads(line)


def build_model_predictor(
    checkpoint_path: Path,
    data_dir: Path,
    window_length: int,
    batch_size: int,
    device_name: str,
) -> Callable[[str, WindowCandidateSystem], np.ndarray]:
    """Load the complete fine-tuned module and return a window predictor."""

    import torch
    from omegaconf import DictConfig, OmegaConf

    from train import SequenceLightningModule

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in checkpoint or "hyper_parameters" not in checkpoint:
        raise ValueError("Expected a complete PyTorch Lightning checkpoint")

    stored_config = checkpoint["hyper_parameters"]
    if isinstance(stored_config, DictConfig):
        stored_config = OmegaConf.to_container(stored_config, resolve=False)
    config = OmegaConf.create(stored_config)
    OmegaConf.update(config, "dataset.data_dir", str(data_dir.resolve()), force_add=True)
    OmegaConf.update(config, "dataset.max_train_windows", None, force_add=True)
    OmegaConf.update(config, "dataset.max_val_windows", None, force_add=True)
    OmegaConf.update(config, "dataset.max_test_windows", None, force_add=True)
    OmegaConf.update(config, "train.pretrained_model_path", None, force_add=True)
    OmegaConf.update(config, "train.pretrained_model_state_hook._name_", None, force_add=True)
    OmegaConf.update(config, "train.ckpt", None, force_add=True)

    configured_length = int(config.dataset.window_length)
    if configured_length != int(window_length):
        raise ValueError(
            f"Checkpoint was fine-tuned with window_length={configured_length}; "
            f"received --window-length={window_length}"
        )
    if str(config.dataset.get("target_transform", "none")) != "none":
        raise ValueError("Sparse reconstruction currently requires untransformed count targets")

    module = SequenceLightningModule(config)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    module.to(device)
    module.eval()
    module.requires_grad_(False)
    tokenizer = module.dataset.tokenizer

    def predict(sequence: str, system: WindowCandidateSystem) -> np.ndarray:
        predictions = []
        for offset in range(0, system.n_windows, batch_size):
            starts = system.starts[offset : offset + batch_size]
            ends = system.ends[offset : offset + batch_size]
            encoded_windows = []
            lengths = []
            for start, end in zip(starts.tolist(), ends.tolist()):
                window = sequence[start:end]
                encoded = tokenizer(
                    window,
                    add_special_tokens=False,
                    padding="max_length",
                    max_length=window_length,
                    truncation=True,
                )
                encoded_windows.append(encoded["input_ids"])
                lengths.append(len(window))

            input_ids = torch.tensor(encoded_windows, dtype=torch.long, device=device)
            valid_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            dummy_targets = torch.zeros((input_ids.shape[0], 1), dtype=torch.float32, device=device)
            batch = (input_ids, dummy_targets, {"lengths": valid_lengths})
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with torch.inference_mode(), autocast:
                output, _, _ = module.forward(batch)
            predictions.append(output.detach().float().reshape(-1).cpu().numpy())
        if not predictions:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(predictions).astype(np.float64, copy=False)

    return predict


def _top_sites(
    positions: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> list[Dict[str, float]]:
    if positions.size == 0 or top_k <= 0:
        return []
    order = np.lexsort((positions, -scores))[: min(top_k, positions.size)]
    return [
        {"position": int(positions[index]), "score": float(scores[index])}
        for index in order
    ]


def run(args: argparse.Namespace) -> Mapping[str, object]:
    if args.split == "test" and not args.allow_test:
        raise ValueError(
            "Test-set reconstruction is locked. Tune stride/lambda on validation, "
            "then pass --allow-test once after freezing the method."
        )

    split_path = args.data_dir / f"{args.split}.jsonl"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if args.mode == "model" and args.checkpoint is None:
        raise ValueError("--checkpoint is required in --mode model")

    predictor = None
    if args.mode == "model":
        predictor = build_model_predictor(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            window_length=args.window_length,
            batch_size=args.batch_size,
            device_name=args.device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.split}_{args.mode}_site_scores.jsonl"
    summary_path = args.output_dir / f"{args.split}_{args.mode}_summary.json"

    total_windows = 0
    count_absolute_error = 0.0
    count_squared_error = 0.0
    total_candidates = 0
    total_observed = 0
    exact_recovered = 0.0
    tolerant_recovered = 0.0
    macro_ap = []
    micro_scores = []
    micro_labels = []
    processed = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output_handle:
        for record in read_records(split_path, args.max_transcripts):
            sequence = str(record["sequence"])
            observed_positions = [int(value) for value in record["observed_m6a_positions"]]
            system = WindowCandidateSystem.from_sequence(
                sequence,
                window_length=args.window_length,
                stride=args.stride,
            )
            true_counts = system.counts_for_positions(observed_positions)
            if predictor is None:
                input_counts = true_counts.copy()
            else:
                input_counts = predictor(sequence, system)

            scores, diagnostics = solve_nonnegative_lasso(
                system,
                input_counts,
                l1_penalty=args.l1_penalty,
                max_iterations=args.max_iterations,
                tolerance=args.solver_tolerance,
                max_score=args.max_score,
            )
            metrics = observed_site_metrics(
                system.candidate_positions,
                scores,
                observed_positions,
                tolerance_nt=args.position_tolerance,
            )

            errors = input_counts - true_counts
            total_windows += system.n_windows
            count_absolute_error += float(np.abs(errors).sum())
            count_squared_error += float(np.square(errors).sum())
            total_candidates += system.n_candidates
            total_observed += len(observed_positions)
            if metrics["average_precision"] is not None:
                macro_ap.append(float(metrics["average_precision"]))
            if metrics["recall_at_observed_k"] is not None:
                exact_recovered += float(metrics["recall_at_observed_k"]) * len(observed_positions)
                tolerant_recovered += (
                    float(metrics["recall_at_observed_k_with_tolerance"])
                    * len(observed_positions)
                )

            observed_set = set(observed_positions)
            micro_scores.extend(scores.tolist())
            micro_labels.extend(
                position in observed_set for position in system.candidate_positions.tolist()
            )

            result = {
                "transcript_id": record["transcript_id"],
                "gene_id": record["gene_id"],
                "sequence_length": len(sequence),
                "n_windows": system.n_windows,
                "n_candidate_a": system.n_candidates,
                "observed_m6a_positions": observed_positions,
                "count_prediction_mae": (
                    float(np.mean(np.abs(errors))) if errors.size else 0.0
                ),
                "sparse_fit": diagnostics,
                "observed_site_metrics": metrics,
                "top_candidate_sites": _top_sites(
                    system.candidate_positions,
                    scores,
                    args.top_k,
                ),
            }
            output_handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            processed += 1

    micro_ap_value = average_precision(micro_scores, micro_labels) if micro_scores else None
    summary = {
        "mode": args.mode,
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "window_length": args.window_length,
        "inference_stride": args.stride,
        "l1_penalty": args.l1_penalty,
        "position_tolerance_nt": args.position_tolerance,
        "transcripts": processed,
        "windows": total_windows,
        "candidate_adenosines": total_candidates,
        "observed_sites": total_observed,
        "window_count_mae": count_absolute_error / max(1, total_windows),
        "window_count_mse": count_squared_error / max(1, total_windows),
        "observed_site_macro_average_precision": (
            float(np.mean(macro_ap)) if macro_ap else None
        ),
        "observed_site_micro_average_precision": micro_ap_value,
        "observed_site_recall_at_k": exact_recovered / max(1, total_observed),
        "observed_site_recall_at_k_with_tolerance": (
            tolerant_recovered / max(1, total_observed)
        ),
        "label_caveat": (
            "Unlabelled adenosines are unknown, so site-ranking metrics measure "
            "recovery of observed calls rather than biological specificity."
        ),
        "output_jsonl": str(output_path.resolve()),
    }
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--mode", choices=["oracle", "model"], default="oracle")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--l1-penalty", type=float, default=0.1)
    parser.add_argument("--max-score", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--solver-tolerance", type=float, default=1e-6)
    parser.add_argument("--position-tolerance", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-transcripts", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Acknowledge that hyperparameters are frozen before test evaluation",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
