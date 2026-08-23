#!/usr/bin/env python
"""Calibrate and evaluate a full-transcript nucleotide-level m6A model.

The decision threshold is selected on the validation split and then frozen for
the test split.  Ranking and thresholded metrics are accumulated with bounded
memory histograms.  Test metrics are also reported separately for 5'UTR, CDS,
3'UTR, and transcript-length strata.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for import_root in (REPO_ROOT, SCRIPTS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from plot_m6a_nucleotide_results import _move_batch, load_module  # noqa: E402


IGNORE_INDEX = -100.0
REGIONS = ("utr5", "cds", "utr3")


def empty_histograms(bins: int) -> Dict[str, np.ndarray]:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    return {
        "positive": np.zeros(int(bins), dtype=np.int64),
        "negative": np.zeros(int(bins), dtype=np.int64),
    }


def update_histograms(
    histograms: MutableMapping[str, np.ndarray],
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> None:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    if probabilities.shape != labels.shape:
        raise ValueError("probabilities and labels must have matching shapes")
    if probabilities.size == 0:
        return
    bins = int(histograms["positive"].size)
    if histograms["negative"].shape != (bins,):
        raise ValueError("positive and negative histograms must have matching shapes")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    indices = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    if np.any(labels):
        histograms["positive"] += np.bincount(indices[labels], minlength=bins)
    if np.any(~labels):
        histograms["negative"] += np.bincount(indices[~labels], minlength=bins)


def ranking_curves(
    positive_hist: np.ndarray,
    negative_hist: np.ndarray,
) -> Dict[str, object]:
    positive_hist = np.asarray(positive_hist, dtype=np.float64)
    negative_hist = np.asarray(negative_hist, dtype=np.float64)
    if positive_hist.ndim != 1 or positive_hist.shape != negative_hist.shape:
        raise ValueError("histograms must be same-length vectors")

    positives = positive_hist[::-1]
    negatives = negative_hist[::-1]
    total_positive = float(positives.sum())
    total_negative = float(negatives.sum())
    true_positive = np.cumsum(positives)
    false_positive = np.cumsum(negatives)

    if total_positive > 0:
        recall = true_positive / total_positive
        precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
        average_precision = float(np.sum(precision * positives) / total_positive)
    else:
        recall = np.zeros_like(true_positive)
        precision = np.zeros_like(true_positive)
        average_precision = None

    if total_positive > 0 and total_negative > 0:
        false_positive_rate = false_positive / total_negative
        true_positive_rate = true_positive / total_positive
        roc_x = np.concatenate(([0.0], false_positive_rate))
        roc_y = np.concatenate(([0.0], true_positive_rate))
        auroc = float(np.sum((roc_y[1:] + roc_y[:-1]) * 0.5 * np.diff(roc_x)))
    else:
        false_positive_rate = np.zeros_like(false_positive)
        true_positive_rate = np.zeros_like(true_positive)
        auroc = None

    return {
        "recall": np.concatenate(([0.0], recall)),
        "precision": np.concatenate(([1.0], precision)),
        "false_positive_rate": np.concatenate(([0.0], false_positive_rate)),
        "true_positive_rate": np.concatenate(([0.0], true_positive_rate)),
        "average_precision": average_precision,
        "auroc": auroc,
    }


def threshold_metrics(
    positive_hist: np.ndarray,
    negative_hist: np.ndarray,
    cutoff_bin: int,
) -> Dict[str, object]:
    positive_hist = np.asarray(positive_hist, dtype=np.int64)
    negative_hist = np.asarray(negative_hist, dtype=np.int64)
    if positive_hist.ndim != 1 or positive_hist.shape != negative_hist.shape:
        raise ValueError("histograms must be same-length vectors")
    bins = int(positive_hist.size)
    if not 0 <= int(cutoff_bin) <= bins:
        raise ValueError("cutoff_bin must be between 0 and bins")

    cutoff_bin = int(cutoff_bin)
    tp = int(positive_hist[cutoff_bin:].sum())
    fp = int(negative_hist[cutoff_bin:].sum())
    fn = int(positive_hist[:cutoff_bin].sum())
    tn = int(negative_hist[:cutoff_bin].sum())
    total = tp + fp + fn + tn
    positive_count = tp + fn
    negative_count = tn + fp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, positive_count)
    specificity = tn / max(1, negative_count)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    accuracy = (tp + tn) / max(1, total)
    balanced_accuracy = 0.5 * (recall + specificity)
    denominator = math.sqrt(
        max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    matthews_correlation = (tp * tn - fp * fn) / denominator
    curves = ranking_curves(positive_hist, negative_hist)
    return {
        "candidate_adenosines": total,
        "positive_m6a": positive_count,
        "negative_adenosines": negative_count,
        "positive_prevalence": positive_count / max(1, total),
        "average_precision": curves["average_precision"],
        "auroc": curves["auroc"],
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "matthews_correlation": matthews_correlation,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def select_validation_threshold(
    positive_hist: np.ndarray,
    negative_hist: np.ndarray,
    objective: str = "f1",
) -> Dict[str, object]:
    """Select a threshold from validation histograms only.

    A higher threshold wins exact objective ties, producing the more
    conservative classifier without consulting test labels.
    """

    positive_hist = np.asarray(positive_hist, dtype=np.int64)
    negative_hist = np.asarray(negative_hist, dtype=np.int64)
    if positive_hist.ndim != 1 or positive_hist.shape != negative_hist.shape:
        raise ValueError("histograms must be same-length vectors")
    if positive_hist.sum() == 0 or negative_hist.sum() == 0:
        raise ValueError("validation threshold selection requires both classes")
    if objective not in {"f1", "balanced_accuracy"}:
        raise ValueError("objective must be f1 or balanced_accuracy")

    bins = int(positive_hist.size)
    f1_values, balanced_accuracy_values = threshold_metric_arrays(
        positive_hist, negative_hist
    )
    values = f1_values if objective == "f1" else balanced_accuracy_values

    best_value = float(np.max(values))
    tied = np.flatnonzero(np.isclose(values, best_value, rtol=0.0, atol=1e-15))
    cutoff_bin = int(tied[-1])
    return {
        "objective": objective,
        "objective_value": best_value,
        "cutoff_bin": cutoff_bin,
        "threshold": cutoff_bin / bins,
        "bins": bins,
    }


def threshold_metric_arrays(
    positive_hist: np.ndarray,
    negative_hist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return F1 and balanced accuracy for every histogram-bin threshold."""

    positive_hist = np.asarray(positive_hist, dtype=np.int64)
    negative_hist = np.asarray(negative_hist, dtype=np.int64)
    if positive_hist.ndim != 1 or positive_hist.shape != negative_hist.shape:
        raise ValueError("histograms must be same-length vectors")
    positive_tail = np.cumsum(positive_hist[::-1])[::-1]
    negative_tail = np.cumsum(negative_hist[::-1])[::-1]
    total_positive = int(positive_hist.sum())
    total_negative = int(negative_hist.sum())
    tp = positive_tail.astype(np.float64)
    fp = negative_tail.astype(np.float64)
    fn = total_positive - tp
    tn = total_negative - fp
    f1_values = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1.0)
    recall = tp / max(1, total_positive)
    specificity = tn / max(1, total_negative)
    balanced_accuracy_values = 0.5 * (recall + specificity)
    return f1_values, balanced_accuracy_values


def parse_length_edges(value: str) -> Tuple[int, ...]:
    edges = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not edges or any(edge <= 0 for edge in edges):
        raise ValueError("length-bin edges must be positive integers")
    if tuple(sorted(set(edges))) != edges:
        raise ValueError("length-bin edges must be strictly increasing")
    return edges


def length_bin_label(length: int, edges: Sequence[int]) -> str:
    lower = 1
    for edge in edges:
        if length <= int(edge):
            return f"{lower}-{int(edge)}"
        lower = int(edge) + 1
    return f">{int(edges[-1])}"


def all_length_bin_labels(edges: Sequence[int]) -> Tuple[str, ...]:
    labels = []
    lower = 1
    for edge in edges:
        labels.append(f"{lower}-{int(edge)}")
        lower = int(edge) + 1
    labels.append(f">{int(edges[-1])}")
    return tuple(labels)


def _resolve_loader(loader):
    if isinstance(loader, (list, tuple)):
        if len(loader) != 1:
            raise ValueError("Expected exactly one data loader")
        return loader[0]
    return loader


def collect_split_histograms(
    module,
    split: str,
    device_name: str,
    bins: int,
    length_edges: Sequence[int],
    stratify: bool,
) -> Dict[str, object]:
    import torch

    if split == "val":
        dataset = module.dataset.dataset_val
        loader = _resolve_loader(module.val_dataloader())
    elif split == "test":
        dataset = module.dataset.dataset_test
        loader = _resolve_loader(module.test_dataloader())
    else:
        raise ValueError("split must be val or test")
    if not hasattr(dataset, "transcript_metadata") or not hasattr(dataset, "records"):
        raise TypeError("This evaluator requires HumanM6AFullTranscriptDataset")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    module.to(device)

    overall = empty_histograms(bins)
    region_histograms = {region: empty_histograms(bins) for region in REGIONS}
    length_labels = all_length_bin_labels(length_edges)
    length_histograms = {label: empty_histograms(bins) for label in length_labels}
    length_transcripts = {label: 0 for label in length_labels}
    transcript_cursor = 0

    for batch_index, batch_cpu in enumerate(loader):
        labels_cpu = batch_cpu[1]
        batch = _move_batch(batch_cpu, device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            logits, targets, _ = module.forward(batch)
            probabilities = torch.sigmoid(logits.reshape(-1).float()).cpu().numpy()
        target_values = targets.reshape(-1).to(torch.bool).cpu().numpy()
        update_histograms(overall, probabilities, target_values)

        selected_cursor = 0
        batch_size = int(labels_cpu.shape[0])
        for row in range(batch_size):
            record_index = transcript_cursor + row
            record = dataset.records[record_index]
            transcript_length = len(str(record["sequence"]))
            row_mask = labels_cpu[row] != IGNORE_INDEX
            positions = torch.nonzero(row_mask, as_tuple=False).reshape(-1).numpy()
            count = int(positions.size)
            row_scores = probabilities[selected_cursor : selected_cursor + count]
            row_labels = target_values[selected_cursor : selected_cursor + count]
            selected_cursor += count

            if stratify:
                label = length_bin_label(transcript_length, length_edges)
                length_transcripts[label] += 1
                update_histograms(length_histograms[label], row_scores, row_labels)
                for region in REGIONS:
                    start = int(record[f"{region}_start"])
                    end = int(record[f"{region}_end"])
                    belongs = (positions >= start) & (positions < end)
                    update_histograms(
                        region_histograms[region], row_scores[belongs], row_labels[belongs]
                    )

        if selected_cursor != probabilities.size:
            raise RuntimeError("Candidate-position accounting does not match model output")
        transcript_cursor += batch_size
        if batch_index % 100 == 0 or batch_index + 1 == len(loader):
            count = int(overall["positive"].sum() + overall["negative"].sum())
            print(f"[{split}] batch {batch_index + 1}/{len(loader)} candidates={count}")

    if transcript_cursor != len(dataset):
        raise RuntimeError(
            f"Evaluated {transcript_cursor} transcripts but dataset contains {len(dataset)}"
        )
    return {
        "transcripts": transcript_cursor,
        "overall": overall,
        "regions": region_histograms,
        "length_bins": length_histograms,
        "length_bin_transcripts": length_transcripts,
    }


def _stratum_summary(
    histograms: Mapping[str, Mapping[str, np.ndarray]],
    cutoff_bin: int,
    transcript_counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    result = {}
    for name, hist in histograms.items():
        metrics = threshold_metrics(hist["positive"], hist["negative"], cutoff_bin)
        if transcript_counts is not None:
            metrics["transcripts"] = int(transcript_counts[name])
        result[name] = metrics
    return result


def build_summary(
    checkpoint: Path,
    checkpoint_metadata: Mapping[str, object],
    threshold_selection: Mapping[str, object],
    validation: Mapping[str, object],
    test: Mapping[str, object],
    length_edges: Sequence[int],
) -> Dict[str, object]:
    cutoff_bin = int(threshold_selection["cutoff_bin"])
    val_overall = threshold_metrics(
        validation["overall"]["positive"],
        validation["overall"]["negative"],
        cutoff_bin,
    )
    test_overall = threshold_metrics(
        test["overall"]["positive"], test["overall"]["negative"], cutoff_bin
    )
    val_overall["transcripts"] = int(validation["transcripts"])
    test_overall["transcripts"] = int(test["transcripts"])
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint_metadata.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint_metadata.get("global_step", -1)),
        "evaluation_protocol": {
            "gene_disjoint_splits": True,
            "threshold_selection_split": "val",
            "threshold_objective": threshold_selection["objective"],
            "threshold_objective_value_on_validation": threshold_selection["objective_value"],
            "frozen_test_threshold": threshold_selection["threshold"],
            "histogram_bins": threshold_selection["bins"],
            "test_labels_used_for_threshold_selection": False,
        },
        "validation": val_overall,
        "test": test_overall,
        "test_by_region": _stratum_summary(test["regions"], cutoff_bin),
        "test_by_transcript_length": _stratum_summary(
            test["length_bins"], cutoff_bin, test["length_bin_transcripts"]
        ),
        "transcript_length_bin_edges": [int(edge) for edge in length_edges],
    }


def _metric_text(value: object) -> str:
    return "" if value is None else f"{float(value):.10g}"


def write_outputs(
    output_dir: Path,
    summary: Mapping[str, object],
    validation: Mapping[str, object],
    test: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m6a_calibrated_evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    metric_names = (
        "candidate_adenosines",
        "positive_m6a",
        "positive_prevalence",
        "average_precision",
        "auroc",
        "precision",
        "recall",
        "specificity",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "matthews_correlation",
    )
    with (output_dir / "m6a_calibrated_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "stratification", "stratum", "transcripts", *metric_names])
        for split, stratification, stratum, metrics in (
            ("val", "overall", "all", summary["validation"]),
            ("test", "overall", "all", summary["test"]),
            *(
                ("test", "region", name, metrics)
                for name, metrics in summary["test_by_region"].items()
            ),
            *(
                ("test", "transcript_length", name, metrics)
                for name, metrics in summary["test_by_transcript_length"].items()
            ),
        ):
            writer.writerow(
                [
                    split,
                    stratification,
                    stratum,
                    metrics.get("transcripts", ""),
                    *(_metric_text(metrics.get(name)) for name in metric_names),
                ]
            )

    bins = int(summary["evaluation_protocol"]["histogram_bins"])
    with (output_dir / "m6a_threshold_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "validation_f1", "validation_balanced_accuracy"])
        f1_values, balanced_accuracy_values = threshold_metric_arrays(
            validation["overall"]["positive"], validation["overall"]["negative"]
        )
        for cutoff, (f1, balanced_accuracy) in enumerate(
            zip(f1_values, balanced_accuracy_values)
        ):
            writer.writerow(
                [
                    f"{cutoff / bins:.10g}",
                    f"{float(f1):.10g}",
                    f"{float(balanced_accuracy):.10g}",
                ]
            )

    with (output_dir / "m6a_pr_roc_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["split", "recall", "precision", "false_positive_rate", "true_positive_rate"]
        )
        for split, collected in (("val", validation), ("test", test)):
            curves = ranking_curves(
                collected["overall"]["positive"], collected["overall"]["negative"]
            )
            for values in zip(
                curves["recall"],
                curves["precision"],
                curves["false_positive_rate"],
                curves["true_positive_rate"],
            ):
                writer.writerow([split, *(f"{float(value):.10g}" for value in values)])


def plot_outputs(
    output_dir: Path,
    summary: Mapping[str, object],
    validation: Mapping[str, object],
    test: Mapping[str, object],
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    blue, orange, green, purple = "#2F6BFF", "#EE8731", "#278B65", "#7655C5"
    gray, grid = "#657184", "#D8DEE8"
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)

    ax = axes[0, 0]
    for split, collected, color in (("Validation", validation, blue), ("Test", test, orange)):
        curves = ranking_curves(
            collected["overall"]["positive"], collected["overall"]["negative"]
        )
        ax.plot(
            curves["recall"],
            curves["precision"],
            color=color,
            linewidth=1.8,
            label=f"{split} (AP={float(curves['average_precision']):.3f})",
        )
    ax.axhline(float(summary["test"]["positive_prevalence"]), color=gray, linestyle="--")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="a  Precision-recall")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    names = ["Overall", "5'UTR", "CDS", "3'UTR"]
    entries = [summary["test"], *(summary["test_by_region"][name] for name in REGIONS)]
    values = [float(entry["average_precision"]) for entry in entries]
    bars = ax.bar(names, values, color=[blue, green, orange, purple], width=0.65)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center")
    ax.set(ylabel="Average precision", ylim=(0, min(1.0, max(values) + 0.12)), title="b  Test ranking by transcript region")

    ax = axes[1, 0]
    x = np.arange(len(names))
    width = 0.24
    for offset, metric, color in ((-width, "precision", blue), (0, "recall", orange), (width, "f1", green)):
        ax.bar(x + offset, [float(entry[metric]) for entry in entries], width, label=metric.capitalize(), color=color)
    ax.set_xticks(x, names)
    ax.set(ylabel="Score", ylim=(0, 1), title="c  Validation-calibrated threshold on test")
    ax.legend(frameon=False, ncol=3)

    ax = axes[1, 1]
    length_names = list(summary["test_by_transcript_length"])
    length_entries = [summary["test_by_transcript_length"][name] for name in length_names]
    length_ap = [
        np.nan if entry["average_precision"] is None else float(entry["average_precision"])
        for entry in length_entries
    ]
    bars = ax.bar(np.arange(len(length_names)), length_ap, color=blue, alpha=0.85)
    ax.set_xticks(np.arange(len(length_names)), length_names, rotation=25, ha="right")
    ax.set(ylabel="Average precision", ylim=(0, 1), title="d  Test ranking by transcript length (nt)")
    for bar, entry in zip(bars, length_entries):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.015, bar.get_height() + 0.012),
            f"n={int(entry['transcripts'])}",
            ha="center",
            fontsize=6.5,
        )

    for ax in axes.reshape(-1):
        ax.grid(axis="y", color=grid, linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    threshold = float(summary["evaluation_protocol"]["frozen_test_threshold"])
    fig.suptitle(
        "RNA-Mamba full-transcript m6A evaluation\n"
        f"Threshold {threshold:.4f} selected on validation and frozen for gene-disjoint test",
        fontsize=12,
        fontweight="bold",
    )
    output_base = output_dir / "rna_mamba_m6a_calibrated_evaluation"
    fig.savefig(output_base.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bins", type=int, default=4096)
    parser.add_argument(
        "--threshold-objective", choices=("f1", "balanced_accuracy"), default="f1"
    )
    parser.add_argument("--length-bin-edges", default="1024,2048,4096,8192")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.bins < 2:
        parser.error("Require batch_size > 0, num_workers >= 0, and bins >= 2")
    try:
        args.length_bin_edges = parse_length_edges(args.length_bin_edges)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    validation_module, checkpoint_metadata = load_module(
        args.checkpoint,
        args.data_dir,
        "val",
        args.batch_size,
        args.num_workers,
    )
    validation = collect_split_histograms(
        validation_module,
        "val",
        args.device,
        args.bins,
        args.length_bin_edges,
        stratify=False,
    )
    threshold_selection = select_validation_threshold(
        validation["overall"]["positive"],
        validation["overall"]["negative"],
        args.threshold_objective,
    )
    print(
        "[validation] selected threshold "
        f"{float(threshold_selection['threshold']):.6f} by "
        f"{threshold_selection['objective']}={float(threshold_selection['objective_value']):.6f}"
    )
    del validation_module
    test_module, _ = load_module(
        args.checkpoint,
        args.data_dir,
        "test",
        args.batch_size,
        args.num_workers,
    )
    test = collect_split_histograms(
        test_module,
        "test",
        args.device,
        args.bins,
        args.length_bin_edges,
        stratify=True,
    )
    summary = build_summary(
        args.checkpoint,
        checkpoint_metadata,
        threshold_selection,
        validation,
        test,
        args.length_bin_edges,
    )
    write_outputs(args.output_dir, summary, validation, test)
    if not args.skip_plots:
        try:
            plot_outputs(args.output_dir, summary, validation, test)
        except ModuleNotFoundError as error:
            if error.name != "matplotlib":
                raise
            print("[warning] matplotlib is unavailable; metric tables were still written")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
