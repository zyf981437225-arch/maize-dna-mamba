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
from src.onemaize.regions import NAM26_GENOTYPES, GenomeInput, build_onemaize_index
from src.onemaize.model_budget import estimate_caduceus_parameters
from src.onemaize.population_audit import (
    audit_population_metadata,
    write_population_audit,
)


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
    assert manifest["schema_version"] == 3
    assert manifest["fasta_alphabet_audited"] is True
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
    assert all(row["n_fraction"] == 0.0 for row in rows)


def test_builder_rejects_assembly_gap_heavy_candidates(tmp_path):
    fasta = tmp_path / "B73.fa"
    genes = tmp_path / "B73.genes.gff3"
    te = tmp_path / "B73.TE.gff3"
    sequence = "N" * 128 + ("ACGT" * 160)
    fasta.write_text(f">chr1\n{sequence}\n", encoding="ascii")
    genes.write_text(
        "##gff-version 3\n"
        "chr1\tNAM\tgene\t321\t352\t.\t+\t.\tID=gene1;biotype=protein_coding\n",
        encoding="utf-8",
    )
    te.write_text(
        "##gff-version 3\n"
        "chr1\tEDTA\trepeat_region\t513\t768\t.\t+\t.\tID=te1\n",
        encoding="utf-8",
    )
    output = tmp_path / "metadata-n"
    manifest = build_onemaize_index(
        [GenomeInput("B73", fasta, genes, te)],
        output,
        primary_context=64,
        extended_context=128,
        candidate_span=128,
        candidate_stride=64,
        gene_flank=16,
        max_n_fraction=0.1,
        seqid_regex=r"^chr1$",
        require_all_classes=False,
    )
    rows = pq.read_table(output / "regions.parquet").to_pylist()
    assert manifest["max_n_fraction"] == 0.1
    assert rows
    assert all(row["n_fraction"] <= 0.1 for row in rows)
    assert not any(row["start"] == 0 for row in rows)


def test_runtime_resamples_n_heavy_crop_inside_selected_pool(tmp_path):
    fasta = tmp_path / "B73.fa"
    genes = tmp_path / "B73.genes.gff3"
    te = tmp_path / "B73.TE.gff3"
    sequence = "N" * 128 + ("ACGT" * 224)
    fasta.write_text(f">chr1\n{sequence}\n", encoding="ascii")
    genes.write_text(
        "##gff-version 3\n"
        "chr1\tNAM\tgene\t321\t352\t.\t+\t.\tID=gene1;biotype=protein_coding\n",
        encoding="utf-8",
    )
    te.write_text(
        "##gff-version 3\n"
        "chr1\tEDTA\trepeat_region\t769\t1024\t.\t+\t.\tID=te1\n",
        encoding="utf-8",
    )
    output = tmp_path / "metadata-runtime-n"
    build_onemaize_index(
        [GenomeInput("B73", fasta, genes, te)],
        output,
        primary_context=64,
        extended_context=128,
        candidate_span=128,
        candidate_stride=64,
        gene_flank=16,
        max_n_fraction=1.0,
        seqid_regex=r"^chr1$",
        require_all_classes=True,
    )
    dataset = OneMaizeRegionMLMDataset(
        output,
        tokenizer=_Tokenizer(),
        split="train",
        context_length=64,
        samples_per_epoch=256,
        deterministic=True,
        allow_index_build=True,
        max_n_fraction=0.1,
        max_crop_attempts=64,
    )
    for index in range(len(dataset)):
        input_ids, labels = dataset[index]
        assert input_ids.shape == labels.shape == (64,)
    assert dataset.filtered_windows > 0


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


def test_formal_mode_requires_exact_panel_and_b73_in_train(tmp_path):
    fasta, output, _ = _write_fixture(tmp_path)
    records = []
    for index, genotype in enumerate(NAM26_GENOTYPES):
        split = "val" if genotype == "B73" else "test" if index in (1, 2) else "train"
        records.append(
            GenomeInput(
                genotype,
                fasta,
                tmp_path / "B73.genes.gff3",
                tmp_path / "B73.TE.gff3",
                split=split,
            )
        )
    with pytest.raises(ValueError, match="Required training genotypes"):
        build_onemaize_index(
            records,
            output.parent / "formal-b73-held-out",
            primary_context=64,
            extended_context=128,
            candidate_span=128,
            candidate_stride=64,
            expected_genotype_count=26,
            expected_split_counts={"train": 23, "val": 1, "test": 2},
            expected_genotypes=NAM26_GENOTYPES,
            required_train_genotypes=("B73",),
        )


def test_formal_nam26_panel_builds_with_23_1_2_split(tmp_path):
    fasta, output, _ = _write_fixture(tmp_path)
    records = []
    for index, genotype in enumerate(NAM26_GENOTYPES):
        split = "train" if index < 23 else "val" if index == 23 else "test"
        records.append(
            GenomeInput(
                genotype,
                fasta,
                tmp_path / "B73.genes.gff3",
                tmp_path / "B73.TE.gff3",
                split=split,
            )
        )
    manifest = build_onemaize_index(
        records,
        output.parent / "formal-nam26",
        primary_context=64,
        extended_context=128,
        candidate_span=128,
        candidate_stride=64,
        seqid_regex=r"^chr[123]$",
        expected_genotype_count=26,
        expected_split_counts={"train": 23, "val": 1, "test": 2},
        expected_genotypes=NAM26_GENOTYPES,
        required_train_genotypes=("B73",),
    )
    assert manifest["formal_split_validated"] is True
    assert manifest["expected_genotype_panel_validated"] is True
    assert manifest["genotype_split_counts"] == {"train": 23, "val": 1, "test": 2}
    assert manifest["region_count"] > 26
    index_builder = OneMaizeRegionMLMDataset(
        output.parent / "formal-nam26",
        tokenizer=_Tokenizer(),
        split="train",
        context_length=128,
        samples_per_epoch=1,
        deterministic=True,
        allow_index_build=True,
    )
    index_builder[0]
    index_builder.close()
    audit_report, audit_rows = audit_population_metadata(
        output.parent / "formal-nam26",
        context_length=128,
        formal=True,
        low_pool_warning=0,
    )
    assert audit_report["status"] == "PASS"
    assert audit_report["split_counts"] == {"train": 23, "val": 1, "test": 2}
    assert len(audit_rows) == 26 * 3


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


def test_teacher_plan_model_budgets():
    base = estimate_caduceus_parameters(d_model=864, n_layer=24)
    pilot = estimate_caduceus_parameters(d_model=512, n_layer=24)
    legacy_width = estimate_caduceus_parameters(d_model=768, n_layer=24)
    assert 110_000_000 <= base["total"] <= 130_000_000
    assert 30_000_000 <= pilot["total"] <= 50_000_000
    assert legacy_width["total"] < 100_000_000


def test_formal_model_b_config_is_schema_v3_region_aware():
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1]
    dataset = OmegaConf.load(
        root / "configs/dataset/onemaize_allcultivar_phase2_16k_region_aware.yaml"
    )
    experiment = OmegaConf.load(
        root / "configs/experiment/onemaize_allcultivar_phase2_16k_region_aware.yaml"
    )
    phase1 = OmegaConf.load(
        root / "configs/experiment/onemaize_b73_phase1_8k_full_genome.yaml"
    )
    assert dataset.mode == "region_aware"
    assert dataset.context_length == 16384
    assert dataset.mlm_probability == 0.15
    assert dataset.reverse_complement_probability == 0.5
    assert [dataset.gene_probability, dataset.non_repeat_probability, dataset.te_rich_probability] == [0.5, 0.3, 0.2]
    assert not any("variant_probability" in key for key in dataset.keys())
    assert experiment.model.config == phase1.model.config
    assert experiment.train.pretrained_model_strict_load is True


def test_population_audit_reports_pool_distribution_without_enforcing_ratio(tmp_path):
    _, output, _ = _write_fixture(tmp_path)
    index_builder = OneMaizeRegionMLMDataset(
        output,
        tokenizer=_Tokenizer(),
        split="train",
        context_length=128,
        samples_per_epoch=1,
        deterministic=True,
        allow_index_build=True,
    )
    index_builder[0]
    index_builder.close()
    report, rows = audit_population_metadata(
        output, context_length=128, formal=False, low_pool_warning=10
    )
    assert report["status"] == "PASS"
    assert report["candidate_pool_distribution_is_training_distribution"] is False
    assert report["training_sampling"]["region_class"] == {
        "gene_centered": 0.5,
        "non_repeat": 0.3,
        "te_rich": 0.2,
    }
    assert len(rows) == 3
    audit_dir = tmp_path / "population-audit"
    write_population_audit(audit_dir, report, rows)
    assert (audit_dir / "ALLCULTIVAR_INPUT_AUDIT.json").is_file()
    assert (audit_dir / "ALLCULTIVAR_INPUT_AUDIT.md").is_file()
    assert (audit_dir / "ALLCULTIVAR_CANDIDATE_COUNTS.csv").is_file()


def test_checkpoint_evaluator_base_mode_does_not_require_schema_v4(tmp_path):
    from types import SimpleNamespace
    from scripts.evaluate_onemaize_checkpoints import _base_dataset

    _, output, _ = _write_fixture(tmp_path)
    args = SimpleNamespace(
        base_data_dir=output,
        split="test",
        context_length=128,
        samples_per_class=2,
        seed=2357,
        fasta_root=None,
        variant_data_dir=None,
    )
    dataset = _base_dataset(args, _Tokenizer(), "te_rich")
    assert len(dataset) == 2
    assert {dataset.sample_metadata(index)["region_class"] for index in range(2)} == {"te_rich"}
    dataset.close()
