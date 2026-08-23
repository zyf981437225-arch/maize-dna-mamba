import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_m6a_full_transcript.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_m6a_full_transcript", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validation_threshold_is_selected_from_histograms_with_conservative_tie_break():
    positive = np.array([0, 0, 0, 0, 0, 0, 2, 0], dtype=np.int64)
    negative = np.array([2, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)

    selected = MODULE.select_validation_threshold(positive, negative, objective="f1")

    assert selected["cutoff_bin"] == 6
    assert selected["threshold"] == 0.75
    assert selected["objective_value"] == 1.0


def test_threshold_metrics_reports_expected_confusion_and_ranking_metrics():
    positive = np.array([0, 1, 0, 2], dtype=np.int64)
    negative = np.array([4, 1, 0, 0], dtype=np.int64)

    metrics = MODULE.threshold_metrics(positive, negative, cutoff_bin=2)

    assert metrics["confusion"] == {"tp": 2, "fp": 0, "tn": 5, "fn": 1}
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 2 / 3
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] == 0.8
    assert metrics["average_precision"] > 0.8
    assert metrics["auroc"] > 0.9


def test_length_bin_helpers_reject_ambiguous_edges_and_cover_tail():
    edges = MODULE.parse_length_edges("1024, 2048,4096")
    assert edges == (1024, 2048, 4096)
    assert MODULE.all_length_bin_labels(edges) == (
        "1-1024",
        "1025-2048",
        "2049-4096",
        ">4096",
    )
    assert MODULE.length_bin_label(1, edges) == "1-1024"
    assert MODULE.length_bin_label(2048, edges) == "1025-2048"
    assert MODULE.length_bin_label(9000, edges) == ">4096"

    for invalid in ("", "1024,1024", "2048,1024", "0,1024"):
        try:
            MODULE.parse_length_edges(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid length edges to fail: {invalid!r}")


def test_output_bundle_records_validation_only_threshold_protocol(tmp_path):
    bins = 4
    validation = {
        "overall": {
            "positive": np.array([0, 0, 1, 2]),
            "negative": np.array([5, 1, 0, 0]),
        }
    }
    test = {
        "overall": {
            "positive": np.array([0, 1, 1, 2]),
            "negative": np.array([4, 1, 0, 0]),
        }
    }
    summary = {
        "evaluation_protocol": {
            "histogram_bins": bins,
            "threshold_selection_split": "val",
            "test_labels_used_for_threshold_selection": False,
        },
        "validation": {"transcripts": 2},
        "test": {"transcripts": 2},
        "test_by_region": {},
        "test_by_transcript_length": {},
    }

    MODULE.write_outputs(tmp_path, summary, validation, test)

    saved = json.loads((tmp_path / "m6a_calibrated_evaluation.json").read_text())
    assert saved["evaluation_protocol"]["threshold_selection_split"] == "val"
    assert saved["evaluation_protocol"]["test_labels_used_for_threshold_selection"] is False
    threshold_rows = (tmp_path / "m6a_threshold_search.csv").read_text().splitlines()
    assert len(threshold_rows) == bins + 1
    assert (tmp_path / "m6a_calibrated_metrics.csv").is_file()
    assert (tmp_path / "m6a_pr_roc_source_data.csv").is_file()


def test_full_transcript_collection_assigns_candidates_to_regions_and_length_bins():
    import pytest

    torch = pytest.importorskip("torch")
    labels = torch.tensor(
        [
            [1.0, -100.0, 0.0, -100.0, -100.0, 1.0, -100.0, -100.0],
            [0.0, 1.0, -100.0, 0.0, -100.0, -100.0, -100.0, -100.0],
        ]
    )
    probabilities = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
    batch = (
        torch.zeros((2, 8), dtype=torch.long),
        labels,
        {
            "attention_mask": labels != -100.0,
            "lengths": torch.tensor([6, 4]),
        },
    )

    class FakeLoader:
        def __iter__(self):
            return iter([batch])

        def __len__(self):
            return 1

    class FakeDataset:
        records = [
            {
                "sequence": "AAAAAA",
                "utr5_start": 0,
                "utr5_end": 2,
                "cds_start": 2,
                "cds_end": 5,
                "utr3_start": 5,
                "utr3_end": 6,
            },
            {
                "sequence": "AAAA",
                "utr5_start": 0,
                "utr5_end": 1,
                "cds_start": 1,
                "cds_end": 3,
                "utr3_start": 3,
                "utr3_end": 4,
            },
        ]

        def __len__(self):
            return len(self.records)

        def transcript_metadata(self, index):
            return {"index": index}

    class FakeModule:
        dataset = type("Data", (), {"dataset_test": FakeDataset()})()

        def test_dataloader(self):
            return [FakeLoader()]

        def to(self, device):
            return self

        def forward(self, incoming_batch):
            selected = incoming_batch[1] != -100.0
            logits = torch.logit(probabilities.to(incoming_batch[1].device)).reshape(-1, 1)
            return logits, incoming_batch[1][selected], {}

    result = MODULE.collect_split_histograms(
        FakeModule(),
        split="test",
        device_name="cpu",
        bins=10,
        length_edges=(4,),
        stratify=True,
    )

    for region in MODULE.REGIONS:
        assert result["regions"][region]["positive"].sum() == 1
        assert result["regions"][region]["negative"].sum() == 1
    assert result["length_bins"]["1-4"]["positive"].sum() == 1
    assert result["length_bins"]["1-4"]["negative"].sum() == 2
    assert result["length_bins"][">4"]["positive"].sum() == 2
    assert result["length_bins"][">4"]["negative"].sum() == 1


def test_stratified_figure_bundle_is_written(tmp_path):
    hist = {
        "positive": np.array([0, 0, 1, 4], dtype=np.int64),
        "negative": np.array([12, 2, 1, 0], dtype=np.int64),
    }
    validation = {"transcripts": 3, "overall": hist}
    test = {
        "transcripts": 4,
        "overall": hist,
        "regions": {name: hist for name in MODULE.REGIONS},
        "length_bins": {"1-1024": hist, ">1024": hist},
        "length_bin_transcripts": {"1-1024": 2, ">1024": 2},
    }
    selection = MODULE.select_validation_threshold(
        hist["positive"], hist["negative"], objective="f1"
    )
    summary = MODULE.build_summary(
        tmp_path / "model.ckpt",
        {"epoch": 0, "global_step": 10},
        selection,
        validation,
        test,
        (1024,),
    )

    MODULE.plot_outputs(tmp_path, summary, validation, test)

    for suffix in ("png", "svg", "pdf"):
        output = tmp_path / f"rna_mamba_m6a_calibrated_evaluation.{suffix}"
        assert output.is_file()
        assert output.stat().st_size > 0
