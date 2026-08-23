#!/usr/bin/env python
"""Evaluate a nucleotide-level m6A checkpoint and create a paper-ready figure.

The script performs one inference pass over a gene-disjoint split. Ranking
curves are accumulated with the same bounded-memory histogram approximation as
the training metrics, so millions of candidate adenosines do not have to be
kept in RAM. A deterministic representative transcript is retained for the
site-level probability track.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


IGNORE_INDEX = -100.0


def curves_from_histograms(
    positive_hist: np.ndarray,
    negative_hist: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """Return descending-threshold PR/ROC curves and their areas."""

    positive_hist = np.asarray(positive_hist, dtype=np.float64)
    negative_hist = np.asarray(negative_hist, dtype=np.float64)
    if positive_hist.shape != negative_hist.shape or positive_hist.ndim != 1:
        raise ValueError("positive_hist and negative_hist must be same-length vectors")

    positives = positive_hist[::-1]
    negatives = negative_hist[::-1]
    total_positive = positives.sum()
    total_negative = negatives.sum()
    if total_positive <= 0 or total_negative <= 0:
        raise ValueError("Both positive and negative examples are required")

    true_positive = np.cumsum(positives)
    false_positive = np.cumsum(negatives)
    recall = true_positive / total_positive
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    false_positive_rate = false_positive / total_negative

    average_precision = float(np.sum(precision * positives) / total_positive)
    roc_y = np.concatenate(([0.0], true_positive / total_positive))
    roc_x = np.concatenate(([0.0], false_positive_rate))
    auroc = float(np.sum((roc_y[1:] + roc_y[:-1]) * 0.5 * np.diff(roc_x)))
    return {
        "recall": np.concatenate(([0.0], recall)),
        "precision": np.concatenate(([1.0], precision)),
        "false_positive_rate": np.concatenate(([0.0], false_positive_rate)),
        "true_positive_rate": np.concatenate(([0.0], true_positive / total_positive)),
        "average_precision": average_precision,
        "auroc": auroc,
        "positive_count": float(total_positive),
        "negative_count": float(total_negative),
    }


def choose_representative_record(dataset, transcript_id: Optional[str] = None) -> int:
    """Choose a deterministic label-defined example, never using model scores."""

    indexed_records = {int(item[0]) for item in dataset.window_index}
    if transcript_id is not None:
        for index, record in enumerate(dataset.records):
            if str(record["transcript_id"]) == transcript_id:
                if index not in indexed_records:
                    raise ValueError(f"Transcript {transcript_id!r} has no evaluated A-containing windows")
                return index
        raise ValueError(f"Transcript {transcript_id!r} is not present in the selected split")

    candidates = []
    for index in sorted(indexed_records):
        record = dataset.records[index]
        sequence_length = len(str(record["sequence"]))
        positive_count = len(record["m6a_positions"])
        if 3 <= positive_count <= 20 and 512 <= sequence_length <= 6000:
            candidates.append((index, positive_count, sequence_length, str(record["transcript_id"])))
    if not candidates:
        for index in sorted(indexed_records):
            record = dataset.records[index]
            positive_count = len(record["m6a_positions"])
            if positive_count:
                candidates.append(
                    (index, positive_count, len(str(record["sequence"])), str(record["transcript_id"]))
                )
    if not candidates:
        raise ValueError("No positive transcript is available for a representative site track")

    positive_median = float(np.median([item[1] for item in candidates]))
    length_median = float(np.median([item[2] for item in candidates]))
    candidates.sort(
        key=lambda item: (
            abs(item[1] - positive_median),
            abs(item[2] - length_median),
            item[3],
        )
    )
    return int(candidates[0][0])


def _move_batch(batch, device):
    import torch

    moved = []
    for value in batch:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device, non_blocking=True))
        elif isinstance(value, dict):
            moved.append(
                {
                    key: item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                    for key, item in value.items()
                }
            )
        else:
            moved.append(value)
    return tuple(moved)


def load_module(
    checkpoint_path: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
):
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
    OmegaConf.update(config, "dataset.shuffle", False, force_add=True)
    OmegaConf.update(config, "dataset.batch_size_eval", int(batch_size), force_add=True)
    OmegaConf.update(config, "dataset.max_train_transcripts", 1, force_add=True)
    OmegaConf.update(config, "dataset.max_train_windows", 1, force_add=True)
    if split == "val":
        OmegaConf.update(config, "dataset.max_val_transcripts", None, force_add=True)
        OmegaConf.update(config, "dataset.max_val_windows", None, force_add=True)
        OmegaConf.update(config, "dataset.max_test_transcripts", 1, force_add=True)
        OmegaConf.update(config, "dataset.max_test_windows", 1, force_add=True)
    else:
        OmegaConf.update(config, "dataset.max_val_transcripts", 1, force_add=True)
        OmegaConf.update(config, "dataset.max_val_windows", 1, force_add=True)
        OmegaConf.update(config, "dataset.max_test_transcripts", None, force_add=True)
        OmegaConf.update(config, "dataset.max_test_windows", None, force_add=True)
    OmegaConf.update(config, "loader.num_workers", int(num_workers), force_add=True)
    OmegaConf.update(config, "loader.pin_memory", True, force_add=True)
    OmegaConf.update(config, "loader.drop_last", False, force_add=True)
    OmegaConf.update(config, "task.pos_weight", 1.0, force_add=True)
    OmegaConf.update(config, "train.pretrained_model_path", None, force_add=True)
    OmegaConf.update(config, "train.pretrained_model_state_hook._name_", None, force_add=True)
    OmegaConf.update(config, "train.ckpt", None, force_add=True)

    module = SequenceLightningModule(config)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    module.eval()
    module.requires_grad_(False)
    return module, checkpoint


def evaluate(
    checkpoint_path: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device_name: str,
    bins: int,
    threshold: float,
    transcript_id: Optional[str],
) -> Tuple[Mapping[str, object], Mapping[str, np.ndarray]]:
    import torch

    if split not in {"val", "test"}:
        raise ValueError("--split must be val or test")
    module, checkpoint = load_module(
        checkpoint_path,
        data_dir,
        split,
        batch_size,
        num_workers,
    )

    dataset = module.dataset.dataset_val if split == "val" else module.dataset.dataset_test
    representative_index = choose_representative_record(dataset, transcript_id)
    representative_record = dataset.records[representative_index]

    if split == "val":
        loader = module.val_dataloader()[0]
    else:
        loader = module.test_dataloader()[0]

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    module.to(device)

    positive_hist = np.zeros(bins, dtype=np.int64)
    negative_hist = np.zeros(bins, dtype=np.int64)
    tp = fp = tn = fn = 0
    window_cursor = 0
    track_positions = []
    track_scores = []
    track_labels = []

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
            probabilities = torch.sigmoid(logits.reshape(-1).float())

        probabilities_np = probabilities.cpu().numpy()
        targets_np = targets.reshape(-1).to(torch.bool).cpu().numpy()
        bin_indices = np.minimum((probabilities_np * bins).astype(np.int64), bins - 1)
        if targets_np.any():
            positive_hist += np.bincount(bin_indices[targets_np], minlength=bins)
        negative_mask = ~targets_np
        if negative_mask.any():
            negative_hist += np.bincount(bin_indices[negative_mask], minlength=bins)

        predicted_np = probabilities_np >= threshold
        tp += int(np.sum(predicted_np & targets_np))
        fp += int(np.sum(predicted_np & ~targets_np))
        tn += int(np.sum(~predicted_np & ~targets_np))
        fn += int(np.sum(~predicted_np & targets_np))

        selected_cursor = 0
        current_batch_size = int(labels_cpu.shape[0])
        for row in range(current_batch_size):
            dataset_index = window_cursor + row
            metadata = dataset.window_metadata(dataset_index)
            row_mask = labels_cpu[row] != IGNORE_INDEX
            count = int(row_mask.sum().item())
            row_scores = probabilities_np[selected_cursor : selected_cursor + count]
            row_labels = targets_np[selected_cursor : selected_cursor + count]
            selected_cursor += count

            record_index = int(dataset.window_index[dataset_index][0])
            if record_index == representative_index:
                local_positions = torch.nonzero(row_mask, as_tuple=False).reshape(-1).numpy()
                global_positions = local_positions + int(metadata["start"])
                track_positions.extend(global_positions.tolist())
                track_scores.extend(row_scores.tolist())
                track_labels.extend(row_labels.astype(np.int8).tolist())

        if selected_cursor != probabilities_np.size:
            raise RuntimeError("Candidate-position accounting does not match model output")
        window_cursor += current_batch_size
        if batch_index % 100 == 0 or batch_index + 1 == len(loader):
            print(
                f"[{split}] batch {batch_index + 1}/{len(loader)} "
                f"candidates={positive_hist.sum() + negative_hist.sum()}"
            )

    if window_cursor != len(dataset):
        raise RuntimeError(f"Evaluated {window_cursor} windows but dataset contains {len(dataset)}")

    curves = curves_from_histograms(positive_hist, negative_hist)
    total = tp + fp + tn + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    accuracy = (tp + tn) / max(1, total)
    prevalence = (tp + fn) / max(1, total)

    order = np.argsort(np.asarray(track_positions, dtype=np.int64))
    track_positions_array = np.asarray(track_positions, dtype=np.int64)[order]
    track_scores_array = np.asarray(track_scores, dtype=np.float64)[order]
    track_labels_array = np.asarray(track_labels, dtype=np.int8)[order]
    if np.unique(track_positions_array).size != track_positions_array.size:
        raise RuntimeError("Representative transcript contains duplicate owned positions")

    summary = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "split": split,
        "ranking_bins": bins,
        "decision_threshold": threshold,
        "windows": len(dataset),
        "candidate_adenosines": total,
        "positive_m6a": tp + fn,
        "negative_adenosines": tn + fp,
        "positive_prevalence": prevalence,
        "average_precision": float(curves["average_precision"]),
        "auroc": float(curves["auroc"]),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "representative_transcript": {
            "selection": (
                "user-specified transcript" if transcript_id is not None else
                "deterministic label-only median positive-count/length selection"
            ),
            "transcript_id": str(representative_record["transcript_id"]),
            "gene_id": str(representative_record["gene_id"]),
            "sequence_length": len(str(representative_record["sequence"])),
            "candidate_adenosines": int(track_positions_array.size),
            "positive_m6a": int(track_labels_array.sum()),
        },
    }
    arrays = {
        "recall": np.asarray(curves["recall"]),
        "precision_curve": np.asarray(curves["precision"]),
        "false_positive_rate": np.asarray(curves["false_positive_rate"]),
        "true_positive_rate": np.asarray(curves["true_positive_rate"]),
        "positive_hist": positive_hist,
        "negative_hist": negative_hist,
        "track_positions": track_positions_array,
        "track_scores": track_scores_array,
        "track_labels": track_labels_array,
    }
    return summary, arrays


def write_source_data(output_dir: Path, summary: Mapping[str, object], arrays: Mapping[str, np.ndarray]):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "m6a_nucleotide_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "m6a_nucleotide_curves_and_track.npz", **arrays)

    with (output_dir / "m6a_nucleotide_curve_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["recall", "precision", "false_positive_rate", "true_positive_rate"])
        for row in zip(
            arrays["recall"],
            arrays["precision_curve"],
            arrays["false_positive_rate"],
            arrays["true_positive_rate"],
        ):
            writer.writerow([f"{float(value):.10g}" for value in row])

    with (output_dir / "m6a_nucleotide_transcript_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["transcript_position_0_based", "predicted_probability", "observed_m6a"])
        for position, score, label in zip(
            arrays["track_positions"], arrays["track_scores"], arrays["track_labels"]
        ):
            writer.writerow([int(position), f"{float(score):.10g}", int(label)])


def _rounded_box(ax, xy, width, height, text, face, edge, fontsize=7, weight="normal"):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
    )


def _arrow(ax, x1, y1, x2, y2, color):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def plot_figure(output_dir: Path, summary: Mapping[str, object], arrays: Mapping[str, np.ndarray]):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    blue = "#2F6BFF"
    blue_dark = "#143B78"
    blue_pale = "#EAF1FF"
    orange = "#EE8731"
    orange_pale = "#FFF0E3"
    green = "#278B65"
    green_pale = "#E7F5EF"
    red = "#CE3D4F"
    gray = "#657184"
    grid = "#D8DEE8"
    black = "#172033"

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "text.color": black,
            "axes.labelcolor": black,
            "xtick.color": black,
            "ytick.color": black,
        }
    )

    fig = plt.figure(figsize=(7.2, 5.0))
    grid_spec = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[0.94, 1.06],
        left=0.07,
        right=0.985,
        top=0.87,
        bottom=0.12,
        wspace=0.42,
        hspace=0.52,
    )
    ax_task = fig.add_subplot(grid_spec[0, :2])
    ax_pr = fig.add_subplot(grid_spec[0, 2])
    ax_roc = fig.add_subplot(grid_spec[1, 0])
    ax_track = fig.add_subplot(grid_spec[1, 1:])

    # a: task schematic
    ax_task.set_axis_off()
    ax_task.set_xlim(0, 1)
    ax_task.set_ylim(0, 1)
    ax_task.text(-0.055, 1.04, "a", fontsize=9, weight="bold")
    ax_task.text(0.0, 1.04, "Nucleotide-level m6A prediction", fontsize=8, weight="bold")
    bases = list("AUGCAAGUCA")
    colors = {"A": green, "U": orange, "G": "#B87817", "C": blue}
    for index, base in enumerate(bases):
        x = 0.01 + index * 0.055
        is_a = base == "A"
        ax_task.add_patch(
            FancyBboxPatch(
                (x, 0.72),
                0.044,
                0.14,
                boxstyle="round,pad=0.007,rounding_size=0.01",
                facecolor=green_pale if is_a else "white",
                edgecolor=green if is_a else grid,
                linewidth=1.0 if is_a else 0.7,
            )
        )
        ax_task.text(x + 0.022, 0.79, base, ha="center", va="center", color=colors[base], weight="bold")
    ax_task.text(0.57, 0.79, "... 1024 nt", color=gray, va="center")
    _rounded_box(ax_task, (0.08, 0.34), 0.24, 0.19, "RNA-Mamba\nfull fine-tuning", blue_pale, blue, 7, "bold")
    _rounded_box(ax_task, (0.42, 0.34), 0.22, 0.19, "Per-position\nclassification head", orange_pale, orange, 7, "bold")
    _rounded_box(ax_task, (0.74, 0.34), 0.19, 0.19, "P(m6A)\nfor every A", green_pale, green, 7, "bold")
    _arrow(ax_task, 0.20, 0.70, 0.20, 0.54, gray)
    _arrow(ax_task, 0.32, 0.435, 0.42, 0.435, gray)
    _arrow(ax_task, 0.64, 0.435, 0.74, 0.435, gray)
    ax_task.text(0.08, 0.17, "1 = methylated A", color=green, weight="bold")
    ax_task.text(0.34, 0.17, "0 = unmethylated A", color=gray, weight="bold")
    ax_task.text(0.08, 0.05, "Loss is evaluated only at A positions; C/G/U and padding are excluded.", color=gray, fontsize=6.3)

    # b: PR curve (hero evidence)
    prevalence = float(summary["positive_prevalence"])
    ap = float(summary["average_precision"])
    ax_pr.text(-0.18, 1.04, "b", transform=ax_pr.transAxes, fontsize=9, weight="bold")
    ax_pr.set_title("Precision–recall", loc="left", weight="bold", pad=7)
    ax_pr.plot(arrays["recall"], arrays["precision_curve"], color=blue, linewidth=2.0)
    ax_pr.axhline(prevalence, color=gray, linestyle="--", linewidth=1.0)
    ax_pr.text(0.98, prevalence + 0.025, f"prevalence = {prevalence:.3f}", ha="right", color=gray, fontsize=6)
    ax_pr.text(0.04, 0.92, f"AUPRC = {ap:.3f}", transform=ax_pr.transAxes, color=blue_dark, weight="bold", fontsize=8)
    ax_pr.text(0.04, 0.82, f"{ap / prevalence:.1f}× random baseline", transform=ax_pr.transAxes, color=gray, fontsize=6.2)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(color=grid, linewidth=0.55)
    ax_pr.set_axisbelow(True)

    # c: ROC curve
    auroc = float(summary["auroc"])
    ax_roc.text(-0.22, 1.04, "c", transform=ax_roc.transAxes, fontsize=9, weight="bold")
    ax_roc.set_title("Receiver operating characteristic", loc="left", weight="bold", pad=7)
    ax_roc.plot(arrays["false_positive_rate"], arrays["true_positive_rate"], color=orange, linewidth=2.0)
    ax_roc.plot([0, 1], [0, 1], color=gray, linestyle="--", linewidth=0.9)
    ax_roc.text(0.96, 0.08, f"AUROC = {auroc:.3f}", ha="right", color=orange, weight="bold", fontsize=8)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.grid(color=grid, linewidth=0.55)
    ax_roc.set_axisbelow(True)

    # d: site-level transcript track
    transcript = summary["representative_transcript"]
    positions = arrays["track_positions"]
    scores = arrays["track_scores"]
    labels = arrays["track_labels"].astype(bool)
    ax_track.text(-0.11, 1.04, "d", transform=ax_track.transAxes, fontsize=9, weight="bold")
    ax_track.set_title("Example transcript: predicted m6A probability at every A", loc="left", weight="bold", pad=7)
    ax_track.vlines(positions, 0, scores, color=blue, alpha=0.28, linewidth=0.45)
    ax_track.scatter(positions, scores, s=5, color=blue, alpha=0.7, linewidths=0, label="Predicted P(m6A)")
    if labels.any():
        ax_track.scatter(
            positions[labels],
            np.full(int(labels.sum()), 1.03),
            marker="v",
            s=18,
            color=red,
            edgecolor="white",
            linewidth=0.35,
            clip_on=False,
            label="Observed m6A",
            zorder=5,
        )
    ax_track.axhline(float(summary["decision_threshold"]), color=gray, linestyle="--", linewidth=0.9)
    ax_track.text(
        0.99,
        float(summary["decision_threshold"]) + 0.025,
        f"threshold {float(summary['decision_threshold']):.2f}",
        transform=ax_track.get_yaxis_transform(),
        ha="right",
        color=gray,
        fontsize=6,
    )
    ax_track.set_xlim(0, max(1, int(transcript["sequence_length"])))
    ax_track.set_ylim(0, 1.08)
    ax_track.set_xlabel("Transcript position (nt)")
    ax_track.set_ylabel("Predicted probability")
    ax_track.grid(axis="y", color=grid, linewidth=0.55)
    ax_track.set_axisbelow(True)
    ax_track.legend(loc="upper right", fontsize=6.2, ncol=2)
    ax_track.text(
        0.01,
        0.94,
        f"{transcript['transcript_id']}  |  gene {transcript['gene_id']}  |  "
        f"{transcript['candidate_adenosines']} A sites; {transcript['positive_m6a']} observed m6A",
        transform=ax_track.transAxes,
        va="top",
        fontsize=6.1,
        color=gray,
    )

    fig.suptitle(
        "RNA-Mamba predicts m6A at single-nucleotide resolution",
        x=0.07,
        y=0.965,
        ha="left",
        fontsize=13,
        weight="bold",
        color=blue_dark,
    )
    fig.text(
        0.07,
        0.915,
        f"Best validation checkpoint • held-out gene-disjoint {summary['split']} split • "
        f"{int(summary['candidate_adenosines']):,} candidate A sites",
        ha="left",
        fontsize=7.3,
        color=gray,
    )
    fig.text(
        0.07,
        0.035,
        f"At threshold {float(summary['decision_threshold']):.2f}: precision {float(summary['precision']):.3f}, "
        f"recall {float(summary['recall']):.3f}, F1 {float(summary['f1']):.3f}. "
        "Curves use the same bounded-memory histogram estimator as model evaluation.",
        ha="left",
        fontsize=6.1,
        color=gray,
    )

    output_base = output_dir / "rna_mamba_m6a_nucleotide_results"
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bins", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--transcript-id", default=None)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0 or args.bins < 2:
        parser.error("Require batch_size > 0, num_workers >= 0, and bins >= 2")
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary, arrays = evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        bins=args.bins,
        threshold=args.threshold,
        transcript_id=args.transcript_id,
    )
    write_source_data(args.output_dir, summary, arrays)
    plot_figure(args.output_dir, summary, arrays)
    print(json.dumps(summary, indent=2))
    print(f"Wrote figure and source data to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
