import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_m6a_nucleotide_results.py"
SPEC = importlib.util.spec_from_file_location("plot_m6a_nucleotide_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_curves_from_histograms_perfect_ranking():
    positive = np.array([0, 0, 2], dtype=np.int64)
    negative = np.array([2, 0, 0], dtype=np.int64)
    result = MODULE.curves_from_histograms(positive, negative)
    assert result["average_precision"] == 1.0
    assert result["auroc"] == 1.0
    assert result["recall"][0] == 0.0
    assert result["precision"][0] == 1.0


def test_curves_from_histograms_reject_single_class():
    try:
        MODULE.curves_from_histograms(np.array([0, 1]), np.array([0, 0]))
    except ValueError as error:
        assert "positive and negative" in str(error)
    else:
        raise AssertionError("Expected a single-class histogram to be rejected")


def test_plot_figure_writes_publication_bundle(tmp_path):
    summary = {
        "split": "test",
        "candidate_adenosines": 100,
        "positive_prevalence": 0.1,
        "average_precision": 0.75,
        "auroc": 0.9,
        "decision_threshold": 0.5,
        "precision": 0.7,
        "recall": 0.8,
        "f1": 0.746,
        "representative_transcript": {
            "transcript_id": "TX1",
            "gene_id": "GENE1",
            "sequence_length": 40,
            "candidate_adenosines": 4,
            "positive_m6a": 2,
        },
    }
    arrays = {
        "recall": np.array([0.0, 0.4, 1.0]),
        "precision_curve": np.array([1.0, 0.8, 0.1]),
        "false_positive_rate": np.array([0.0, 0.1, 1.0]),
        "true_positive_rate": np.array([0.0, 0.8, 1.0]),
        "track_positions": np.array([2, 10, 20, 31]),
        "track_scores": np.array([0.1, 0.9, 0.2, 0.8]),
        "track_labels": np.array([0, 1, 0, 1]),
    }
    MODULE.plot_figure(tmp_path, summary, arrays)
    for suffix in ("png", "svg", "pdf", "tiff"):
        path = tmp_path / f"rna_mamba_m6a_nucleotide_results.{suffix}"
        assert path.is_file()
        assert path.stat().st_size > 0
