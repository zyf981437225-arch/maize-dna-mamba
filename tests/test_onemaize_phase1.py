from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from caduceus.tokenization_caduceus import CaduceusTokenizer
from scripts.build_b73_phase1_8k_manifest import build_rows, read_fai, write_parquet
from src.dataloaders.datasets.onemaize_phase1_dataset import (
    OneMaizePhase1FullGenomeMLMDataset,
    collate_onemaize_phase1_mlm,
)
from src.onemaize.phase1_coverage import Phase1CoverageTracker, distributed_sampler_stats


def _write_fasta_with_fai(path: Path, records: list[tuple[str, str]]) -> Path:
    offset = 0
    fai_rows = []
    chunks = []
    for name, sequence in records:
        header = f">{name}\n".encode("ascii")
        body = f"{sequence}\n".encode("ascii")
        chunks.extend([header, body])
        offset += len(header)
        fai_rows.append(f"{name}\t{len(sequence)}\t{offset}\t{len(sequence)}\t{len(sequence) + 1}")
        offset += len(body)
    path.write_bytes(b"".join(chunks))
    fai = Path(f"{path}.fai")
    fai.write_text("\n".join(fai_rows) + "\n", encoding="ascii")
    return fai


def _tokenizer():
    return CaduceusTokenizer(model_max_length=8, sequence_type="dna")


def test_builder_counts_and_required_columns(tmp_path):
    fasta = tmp_path / "B73.fa"
    fai = _write_fasta_with_fai(fasta, [("chr1", "ACGT" * 5), ("chr2", "T" * 8)])
    lengths = read_fai(fai)
    rows = build_rows(lengths, ("chr1", "chr2"), fasta, fai, window_size=8, stride=8)
    assert len(rows) == 4
    assert sum(not row["is_tail"] for row in rows) == 3
    assert sum(row["valid_bp"] for row in rows) == 28
    assert rows[2]["chromosome"] == "chr1"
    assert rows[2]["start"] == 16 and rows[2]["end"] == 20
    assert rows[2]["padded_bp"] == 4 and rows[2]["is_tail"] is True
    output = tmp_path / "phase1.parquet"
    write_parquet(rows, output)
    assert set((pq.read_table(output)).column_names) >= {
        "region_id", "chromosome", "start", "end", "valid_bp", "padded_bp", "is_tail", "window_size", "stride"
    }


def test_phase1_dataset_never_crosses_chromosome_and_pads_tail(tmp_path):
    fasta = tmp_path / "B73.fa"
    fai = _write_fasta_with_fai(fasta, [("chr1", "ACGT" * 5), ("chr2", "T" * 8)])
    rows = build_rows(read_fai(fai), ("chr1", "chr2"), fasta, fai, window_size=8, stride=8)
    manifest = tmp_path / "phase1.parquet"
    write_parquet(rows, manifest)
    dataset = OneMaizePhase1FullGenomeMLMDataset(
        manifest, tokenizer=_tokenizer(), context_length=8, window_size=8, stride=8,
        fasta_path=fasta, allow_index_build=True, deterministic=True,
    )
    assert len(dataset) == 4
    assert dataset.coordinate(2)["chromosome"] == "chr1"
    assert dataset.coordinate(3)["chromosome"] == "chr2"
    inputs, labels, metadata = dataset[2]
    assert inputs.shape == labels.shape == (8,)
    assert metadata["valid_mask"].tolist() == [True] * 4 + [False] * 4
    assert labels[4:].tolist() == [dataset.pad_id] * 4
    assert metadata["phase1_is_tail"].item() is True
    inputs_b, labels_b, metadata_b = dataset[2]
    assert inputs.equal(inputs_b) and labels.equal(labels_b)
    assert metadata_b["phase1_region_id"].item() == 2


def test_phase1_dataset_all_n_window_uses_next_canonical_window(tmp_path):
    fasta = tmp_path / "B73.fa"
    fai = _write_fasta_with_fai(fasta, [("chr1", "N" * 8 + "ACGT" * 2)])
    rows = build_rows(read_fai(fai), ("chr1",), fasta, fai, window_size=8, stride=8)
    manifest = tmp_path / "phase1.parquet"
    write_parquet(rows, manifest)
    dataset = OneMaizePhase1FullGenomeMLMDataset(
        manifest,
        tokenizer=_tokenizer(),
        context_length=8,
        window_size=8,
        stride=8,
        fasta_path=fasta,
        allow_index_build=True,
        deterministic=True,
        reverse_complement_probability=0.0,
    )

    inputs, labels, metadata = dataset[0]
    repeated_inputs, repeated_labels, repeated_metadata = dataset[0]

    assert dataset.coordinate(0)["region_id"] == 0
    assert metadata["phase1_region_id"].item() == 1
    assert repeated_metadata["phase1_region_id"].item() == 1
    assert inputs.equal(repeated_inputs)
    assert labels.equal(repeated_labels)
    assert torch.isfinite(inputs.float()).all()
    assert torch.isfinite(labels.float()).all()

    vocab_size = max(dataset.tokenizer.get_vocab().values()) + 1
    logits = torch.zeros((dataset.context_length, vocab_size), dtype=torch.float32)
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=dataset.pad_id)
    assert torch.isfinite(loss)


def test_phase1_collate_preserves_masks_and_ids(tmp_path):
    fasta = tmp_path / "B73.fa"
    fai = _write_fasta_with_fai(fasta, [("chr1", "ACGT" * 2)])
    rows = build_rows(read_fai(fai), ("chr1",), fasta, fai, window_size=8, stride=8)
    manifest = tmp_path / "phase1.parquet"
    write_parquet(rows, manifest)
    dataset = OneMaizePhase1FullGenomeMLMDataset(manifest, tokenizer=_tokenizer(), context_length=8, window_size=8, stride=8, fasta_path=fasta, allow_index_build=True, deterministic=True)
    inputs, labels, metadata = collate_onemaize_phase1_mlm([dataset[0]], pad_token_id=dataset.pad_id)
    assert inputs.shape == labels.shape == (1, 8)
    assert metadata["attention_mask"].all()
    assert metadata["phase1_region_id"].tolist() == [0]


def test_phase1_tracker_and_ddp_arithmetic(tmp_path):
    class _Dataset:
        rows = [{"valid_bp": 8, "is_tail": False}, {"valid_bp": 4, "is_tail": True}, {"valid_bp": 8, "is_tail": False}]
        total_valid_bp = 20

    tracker = Phase1CoverageTracker(_Dataset())
    tracker.update([0, 2, 2])
    metrics = tracker.compute()
    assert metrics["train/phase1_samples_seen"].item() == 3
    assert metrics["train/phase1_unique_regions"].item() == 2
    assert metrics["train/phase1_genomic_bp_coverage"].item() == pytest.approx(16 / 20)
    assert distributed_sampler_stats(260239, 1) == {
        "per_rank_samples": 260239, "global_samples_drawn": 260239,
        "unique_regions": 260239, "duplicate_samples": 0,
    }
    assert distributed_sampler_stats(260239, 8)["per_rank_samples"] == 32530
    assert distributed_sampler_stats(260239, 8)["global_samples_drawn"] == 260240
    assert distributed_sampler_stats(260239, 8)["duplicate_samples"] == 1
