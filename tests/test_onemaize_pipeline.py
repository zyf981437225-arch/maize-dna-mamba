from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.dataloaders.datasets.onemaize_dataset import (
    OneMaizeRegionMLMDataset,
    reverse_complement,
)
from src.dataloaders.onemaize_mlm import OneMaizeDNAMLM
from src.onemaize.regions import GenomeInput, build_onemaize_index


class _Tokenizer:
    pad_token_id = 4
    mask_token_id = 3
    unk_token_id = 6

    def get_vocab(self):
        return {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            "A": 7,
            "C": 8,
            "G": 9,
            "T": 10,
            "N": 11,
        }


def _write_fixture(tmp_path: Path):
    fasta = tmp_path / "B73.fa"
    genes = tmp_path / "B73.genes.gff3"
    te = tmp_path / "B73.TE.gff3"
    sequence = ("ACGT" * 192)[:768]
    fasta.write_text(
        "".join(f">chr{i}\n{sequence}\n" for i in range(1, 4)),
        encoding="ascii",
    )
    genes.write_text(
        "##gff-version 3\n"
        + "".join(
            f"chr{i}\tNAM\tgene\t337\t368\t.\t+\t.\t"
            f"ID=gene{i};biotype=protein_coding\n"
            for i in range(1, 4)
        ),
        encoding="utf-8",
    )
    te.write_text(
        "##gff-version 3\n"
        + "".join(
            f"chr{i}\tEDTA\trepeat_region\t1\t128\t.\t+\t.\t"
            f"ID=te{i}a;Classification=LTR/Gypsy\n"
            f"chr{i}\tEDTA\trepeat_region\t641\t768\t.\t+\t.\t"
            f"ID=te{i}b;Classification=LTR/Copia\n"
            for i in range(1, 4)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "metadata"
    manifest = build_onemaize_index(
        [GenomeInput("B73", fasta, genes, te)],
        output,
        primary_context=64,
        extended_context=128,
        candidate_span=128,
        candidate_stride=64,
        gene_flank=16,
        seqid_regex=r"^chr[123]$",
        val_seqids=["chr2"],
        test_seqids=["chr3"],
    )
    return fasta, output, manifest


def test_region_builder_writes_teacher_plan_pools(tmp_path):
    _, output, manifest = _write_fixture(tmp_path)
    assert manifest["coordinate_system"] == "0-based-half-open"
    assert manifest["primary_context"] == 64
    assert manifest["extended_context"] == 128
    for split in ("train", "val", "test"):
        counts = manifest["counts"][split]["B73"]
        assert counts["gene_centered"] == 1
        assert counts["non_repeat"] > 0
        assert counts["te_rich"] > 0

    rows = pq.read_table(output / "regions.parquet").to_pylist()
    gene_rows = [row for row in rows if row["region_class"] == "gene_centered"]
    assert len(gene_rows) == 3
    assert all(row["end"] - row["start"] >= 128 for row in gene_rows)
    assert {row["split"] for row in rows} == {"train", "val", "test"}


def test_dynamic_dataset_samples_50_30_20_and_masks(tmp_path):
    fasta, output, _ = _write_fixture(tmp_path)
    dataset = OneMaizeRegionMLMDataset(
        output,
        tokenizer=_Tokenizer(),
        split="train",
        context_length=64,
        samples_per_epoch=5000,
        reverse_complement_probability=0.5,
        deterministic=True,
        allow_index_build=True,
    )
    counts = Counter(dataset.sample_metadata(i)["region_class"] for i in range(5000))
    assert abs(counts["gene_centered"] / 5000 - 0.5) < 0.03
    assert abs(counts["non_repeat"] / 5000 - 0.3) < 0.03
    assert abs(counts["te_rich"] / 5000 - 0.2) < 0.03

    input_ids, labels = dataset[17]
    assert input_ids.shape == labels.shape == (64,)
    selected = labels.ne(_Tokenizer.pad_token_id)
    assert selected.any()
    assert set(labels[selected].tolist()).issubset({7, 8, 9, 10})
    assert Path(f"{fasta}.fai").exists()


def test_extended_context_and_reverse_complement(tmp_path):
    _, output, _ = _write_fixture(tmp_path)
    dataset = OneMaizeRegionMLMDataset(
        output,
        tokenizer=_Tokenizer(),
        split="val",
        context_length=128,
        samples_per_epoch=4,
        reverse_complement_probability=0.0,
        deterministic=True,
        allow_index_build=True,
    )
    input_ids, labels = dataset[0]
    assert input_ids.shape == labels.shape == (128,)
    assert reverse_complement("AACGTN") == "NACGTT"


def test_formal_mode_rejects_single_genotype_pilot(tmp_path):
    fasta, output, _ = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="Expected 26 genotypes"):
        build_onemaize_index(
            [
                GenomeInput(
                    "B73",
                    fasta,
                    tmp_path / "B73.genes.gff3",
                    tmp_path / "B73.TE.gff3",
                )
            ],
            output.parent / "formal",
            primary_context=64,
            extended_context=128,
            candidate_span=128,
            candidate_stride=64,
            expected_genotype_count=26,
            expected_split_counts={"train": 23, "val": 1, "test": 2},
        )


def test_data_module_declares_same_position_mlm(tmp_path):
    _, output, _ = _write_fixture(tmp_path)
    module = OneMaizeDNAMLM(
        _name_="onemaize_dna_mlm",
        data_dir=output,
        context_length=64,
        train_samples_per_epoch=1,
        val_samples_per_epoch=1,
        test_samples_per_epoch=1,
        allow_index_build=True,
    )
    assert module.mlm is True
