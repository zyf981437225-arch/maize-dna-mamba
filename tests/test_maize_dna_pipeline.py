from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOKENIZER_MODULE = _load_module(
    "dna_test_tokenizer", "caduceus/tokenization_caduceus.py"
)
DATASET_MODULE = _load_module(
    "dna_test_dataset", "src/dataloaders/datasets/genomic_dna_dataset.py"
)
PREPARE_MODULE = _load_module(
    "dna_test_prepare", "scripts/prepare_maize_genome.py"
)
READER_MODULE = _load_module(
    "dna_test_reader", "caduceus/memory_cross_attn.py"
)
RC_MODULE = _load_module("dna_test_rc", "src/dataloaders/utils/rc.py")

CaduceusTokenizer = TOKENIZER_MODULE.CaduceusTokenizer
IndexedGenomicDNAMLMDataset = DATASET_MODULE.IndexedGenomicDNAMLMDataset
collate_genomic_dna_mlm = DATASET_MODULE.collate_genomic_dna_mlm
prepare_genome = PREPARE_MODULE.prepare_genome
MemoryCrossAttention = READER_MODULE.MemoryCrossAttention
string_reverse_complement = RC_MODULE.string_reverse_complement


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(f">{identifier}\n{sequence}\n" for identifier, sequence in records),
        encoding="utf-8",
    )


def test_dna_tokenizer_rejects_rna_and_dna_rc_is_involutive():
    dna = CaduceusTokenizer(model_max_length=16, sequence_type="dna")

    assert list(dna.characters) == list("ACGTN")
    assert "U" not in dna.get_vocab()
    assert dna.convert_tokens_to_ids(list("ACGTN")) == [7, 8, 9, 10, 11]
    assert dna.convert_ids_to_tokens([7, 8, 9, 10, 11]) == list("ACGTN")
    assert dna.complement_map[dna.get_vocab()["A"]] == dna.get_vocab()["T"]
    with pytest.raises(ValueError, match="requires sequence_type='dna'"):
        CaduceusTokenizer(model_max_length=16, sequence_type="rna")

    sequence = "ACGTNNGCTA"
    assert string_reverse_complement(string_reverse_complement(sequence)) == sequence


def test_prepare_windows_never_cross_contigs_and_dna_mlm_is_aligned(tmp_path):
    fasta = tmp_path / "b73_subset.fa"
    _write_fasta(
        fasta,
        [
            ("chr1", "ACGTNRYACGTACGTACGTN"),
            ("chr2", "TTTTCCCCAAAA"),
            ("chr3", "GGGGAAAATTTT"),
        ],
    )
    split_config = tmp_path / "splits.json"
    split_config.write_text(
        json.dumps({"train": ["chr1"], "val": ["chr2"], "test": ["chr3"]}),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    manifest = prepare_genome(
        fasta_paths=[fasta],
        genome_names=["B73"],
        output_dir=output,
        window_size=8,
        stride=4,
        split_config=split_config,
        unassigned_policy="error",
        ambiguity_policy="map_to_n",
        max_n_fraction=1.0,
        require_all_splits=True,
    )

    assert manifest["sanity_checks"]["formal_pretraining_ready"] is True
    assert manifest["base_statistics"]["iupac_ambiguity"] == 2
    assert manifest["splits"]["train"]["windows"] == 4
    assert manifest["splits"]["val"]["windows"] == 2
    assert manifest["splits"]["test"]["windows"] == 2
    assert (output / "DATA_STATS.md").exists()

    expected_contig = {"train": "chr1", "val": "chr2", "test": "chr3"}
    for split in ("train", "val", "test"):
        with gzip.open(
            output / manifest["splits"][split]["files"]["metadata"],
            "rt",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        assert rows
        assert {row["contig"] for row in rows} == {expected_contig[split]}
        assert all(int(row["end"]) - int(row["start"]) == 8 for row in rows)

    tokenizer = CaduceusTokenizer(model_max_length=8, sequence_type="dna")
    dataset = IndexedGenomicDNAMLMDataset(
        output,
        tokenizer=tokenizer,
        split="train",
        window_size=8,
        mlm_probability=1.0,
        deterministic_mlm=True,
        seed=7,
    )
    first_a = dataset[0]
    first_b = dataset[0]
    assert first_a[0].shape == first_a[1].shape == (8,)
    assert torch.equal(first_a[0], first_b[0])
    assert torch.equal(first_a[1], first_b[1])
    predictable = {tokenizer.get_vocab()[base] for base in "ACGT"}
    assert set(first_a[1].tolist()).issubset(predictable | {tokenizer.pad_token_id})

    inputs, labels, metadata = collate_genomic_dna_mlm(
        [dataset[0], dataset[1]], pad_token_id=tokenizer.pad_token_id
    )
    assert inputs.shape == labels.shape == (2, 8)
    assert metadata["attention_mask"].shape == (2, 8)
    assert metadata["attention_mask"].all()


def test_raw_u_fails_training_gate_but_preserves_stats(tmp_path):
    fasta = tmp_path / "bad_dna.fa"
    _write_fasta(fasta, [("chr1", "ACGUACGT")])
    output = tmp_path / "prepared_bad"
    manifest = prepare_genome(
        fasta_paths=[fasta],
        genome_names=["B73"],
        output_dir=output,
        window_size=8,
        stride=8,
        ambiguity_policy="map_to_n",
    )
    assert manifest["base_statistics"]["U"] == 1
    assert manifest["sanity_checks"]["u_fraction_passed"] is False
    assert (output / "DATA_STATS.md").exists()

    tokenizer = CaduceusTokenizer(model_max_length=8, sequence_type="dna")
    with pytest.raises(ValueError, match="U-frequency sanity check"):
        IndexedGenomicDNAMLMDataset(
            output, tokenizer=tokenizer, split="train", window_size=8
        )


def test_memory_reader_broadcasts_safely_at_10240_positions():
    reader = MemoryCrossAttention(d_model=128, d_mem=64)
    hidden = torch.zeros(1, 10240, 128)
    memory = torch.randn(1, 2, 64, requires_grad=True)
    update = reader(hidden, memory)
    assert update.shape == (1, 1, 128)
    output = hidden + update
    assert output.shape == (1, 10240, 128)
    output.mean().backward()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()
