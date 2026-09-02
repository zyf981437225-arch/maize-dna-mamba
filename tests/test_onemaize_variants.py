from __future__ import annotations

from pathlib import Path

import json
import numpy as np
from omegaconf import OmegaConf
import pyarrow.parquet as pq
import pytest

from src.dataloaders.datasets.onemaize_variant_dataset import (
    OneMaizeVariantTEMLMDataset,
    crop_for_event,
)
from src.dataloaders.onemaize_variant_mlm import OneMaizeVariantTEDNAMLM
from src.onemaize.regions import GenomeInput, build_onemaize_index
from src.onemaize.variant_audit import audit_variant_metadata, write_audit_outputs
from src.onemaize.variants import (
    VariantEvent,
    VariantInput,
    build_variant_metadata,
    one_based_closed_to_half_open,
    one_based_insertion_after_to_boundary,
    parse_vcf,
    write_variant_events,
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


def _write_vcf(path: Path, records: list[str]) -> Path:
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        + "\n".join(records)
        + "\n",
        encoding="utf-8",
    )
    return path


def _events(tmp_path: Path, records: list[str]):
    return list(
        parse_vcf(
            _write_vcf(tmp_path / "variants.vcf", records),
            genotype="B97",
            coordinate_genotype="B97",
            reference_genotype="B73",
            split="train",
            source="fixture",
        )
    )


def test_one_based_coordinate_boundaries():
    assert one_based_closed_to_half_open(1, 1) == (0, 1)
    assert one_based_closed_to_half_open(10, 20) == (9, 20)
    assert one_based_insertion_after_to_boundary(1) == 1
    with pytest.raises(ValueError):
        one_based_closed_to_half_open(0, 1)


def test_vcf_snp_insertion_and_deletion_coordinates(tmp_path):
    snp, insertion, deletion = _events(
        tmp_path,
        [
            "chr1\t10\tsnp1\tA\tG\t.\tPASS\t.",
            "chr1\t20\tins1\tA\tAT\t.\tPASS\t.",
            "chr1\t30\tdel1\tAT\tA\t.\tPASS\t.",
        ],
    )
    assert (snp.start, snp.end, snp.variant_type) == (9, 10, "snp")
    assert (insertion.start, insertion.end, insertion.variant_length) == (20, 20, 1)
    assert (deletion.start, deletion.end, deletion.variant_length) == (30, 31, -1)


def test_vcf_large_sv_pav_and_te_coordinates(tmp_path):
    deletion, inversion, pav, te_insertion, te_deletion = _events(
        tmp_path,
        [
            "chr1\t100\tsvdel\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;END=201;SVLEN=-102",
            "chr1\t300\tinv\tN\t<INV>\t.\tPASS\tSVTYPE=INV;END=500",
            "chr1\t600\tpav\tN\t<PAV>\t.\tPASS\tSVTYPE=PAV;END=900",
            "chr1\t1000\tteins\tN\t<INS>\t.\tPASS\tSVTYPE=INS;ONEMAIZE_TYPE=te_insertion;SVLEN=5000;TE_FAMILY=Gypsy",
            "chr1\t2000\ttedel\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL;ONEMAIZE_TYPE=te_deletion;END=2500;TE_FAMILY=Copia",
        ],
    )
    assert (deletion.start, deletion.end) == (99, 201)
    assert (deletion.left_breakpoint, deletion.right_breakpoint) == (99, 201)
    assert (inversion.start, inversion.end, inversion.variant_type) == (299, 500, "inversion")
    assert (pav.start, pav.end, pav.variant_type) == (599, 900, "pav")
    assert (te_insertion.start, te_insertion.end) == (1000, 1000)
    assert te_insertion.variant_type == "te_insertion" and te_insertion.te_family == "Gypsy"
    assert (te_deletion.start, te_deletion.end, te_deletion.variant_type) == (1999, 2500, "te_deletion")


def test_schema_v4_nullable_round_trip(tmp_path):
    events = [
        VariantEvent(
            variant_id="snp1",
            genotype="B97",
            reference_genotype="B73",
            coordinate_genotype="B97",
            seqid="chr1",
            start=9,
            end=10,
            variant_type="snp",
            reference_allele="A",
            alternate_allele="G",
            variant_length=1,
            source="fixture",
            split="train",
        ),
        VariantEvent(
            variant_id="te1",
            genotype="B97",
            reference_genotype="B73",
            coordinate_genotype="B97",
            seqid="chr1",
            start=20,
            end=20,
            variant_type="te_insertion",
            reference_allele=None,
            alternate_allele=None,
            variant_length=5000,
            source="fixture",
            split="train",
            left_breakpoint=20,
            right_breakpoint=20,
            te_family="Gypsy",
            alternate_presence=True,
        ),
    ]
    output = tmp_path / "variant_regions.parquet"
    assert write_variant_events(output, events) == 2
    table = pq.read_table(output)
    assert table.schema.field("te_family").nullable
    rows = table.to_pylist()
    assert rows[0]["te_family"] is None
    assert rows[1]["alternate_presence"] is True


def test_schema_rejects_unmapped_reference_coordinates():
    event = VariantEvent(
        variant_id="bad",
        genotype="B97",
        reference_genotype="B73",
        coordinate_genotype="B73",
        seqid="chr1",
        start=1,
        end=2,
        variant_type="snp",
        reference_allele="A",
        alternate_allele="G",
        variant_length=1,
        source="fixture",
        split="train",
    )
    with pytest.raises(ValueError, match="coordinates of their genotype FASTA"):
        event.validated()


def test_variant_crop_contains_event_with_jitter_and_never_crosses_chromosome():
    event = {"start": 1000, "end": 1100, "left_breakpoint": None, "right_breakpoint": None}
    starts = set()
    for seed in range(32):
        crop = crop_for_event(
            event,
            context_length=512,
            sequence_length=2048,
            rng=np.random.default_rng(seed),
            jitter=200,
        )
        starts.add(crop["crop_start"])
        assert crop["crop_start"] <= event["start"]
        assert event["end"] <= crop["crop_end"]
        assert 0 <= crop["crop_start"] < crop["crop_end"] <= 2048
    assert len(starts) > 1


def test_large_sv_uses_a_breakpoint_context():
    event = {
        "start": 1000,
        "end": 9000,
        "left_breakpoint": 1000,
        "right_breakpoint": 9000,
    }
    observed = set()
    for seed in range(16):
        crop = crop_for_event(
            event,
            context_length=1024,
            sequence_length=10000,
            rng=np.random.default_rng(seed),
            jitter=128,
        )
        observed.add(crop["sampling_subtype"])
        assert 0 <= crop["crop_start"] < crop["crop_end"] <= 10000
        assert crop["crop_start"] <= crop["target_start"] < crop["crop_end"]
    assert observed == {"left_breakpoint", "right_breakpoint"}


def _write_fasta_and_index(path: Path, sequence: str) -> None:
    header = b">chr1\n"
    body = sequence.encode("ascii") + b"\n"
    path.write_bytes(header + body)
    Path(f"{path}.fai").write_text(
        f"chr1\t{len(sequence)}\t{len(header)}\t{len(sequence)}\t{len(sequence) + 1}\n",
        encoding="ascii",
    )


def _variant_dataset_fixture(tmp_path: Path):
    inputs = []
    events = []
    splits = {"G1": "train", "G2": "train", "G3": "val", "G4": "test"}
    for genotype, split in splits.items():
        fasta = tmp_path / f"{genotype}.fa"
        genes = tmp_path / f"{genotype}.genes.gff3"
        te = tmp_path / f"{genotype}.te.gff3"
        _write_fasta_and_index(fasta, ("ACGT" * 512)[:2048])
        genes.write_text(
            "##gff-version 3\n"
            "chr1\tNAM\tgene\t900\t1000\t.\t+\t.\tID=gene1;biotype=protein_coding\n",
            encoding="utf-8",
        )
        te.write_text(
            "##gff-version 3\n"
            "chr1\tEDTA\trepeat_region\t1\t512\t.\t+\t.\tID=te1\n"
            "chr1\tEDTA\trepeat_region\t1537\t2048\t.\t+\t.\tID=te2\n",
            encoding="utf-8",
        )
        inputs.append(GenomeInput(genotype, fasta, genes, te, split=split))
        for variant_type, start, end, length in (
            ("snp", 700, 701, 1),
            ("inversion", 800, 1200, 400),
            ("te_insertion", 1300, 1300, 300),
        ):
            events.append(
                VariantEvent(
                    variant_id=f"{genotype}:{variant_type}",
                    genotype=genotype,
                    reference_genotype="B73",
                    coordinate_genotype=genotype,
                    seqid="chr1",
                    start=start,
                    end=end,
                    variant_type=variant_type,
                    reference_allele=None,
                    alternate_allele=None,
                    variant_length=length,
                    source="fixture",
                    split=split,
                    left_breakpoint=start if variant_type == "inversion" else None,
                    right_breakpoint=end if variant_type == "inversion" else None,
                )
            )
    base_dir = tmp_path / "base"
    build_onemaize_index(
        inputs,
        base_dir,
        primary_context=64,
        extended_context=128,
        candidate_span=512,
        candidate_stride=256,
        gene_flank=32,
        repeat_threshold=0.5,
        seqid_regex=r"^chr1$",
    )
    variant_dir = tmp_path / "variants"
    variant_dir.mkdir()
    write_variant_events(variant_dir / "variant_regions.parquet", events)
    (variant_dir / "variant_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "base_schema_version": 3,
                "coordinate_system": "0-based-half-open",
                "files": {"variants": "variant_regions.parquet"},
            }
        ),
        encoding="utf-8",
    )
    return base_dir, variant_dir


def _make_dataset(base_dir, variant_dir, **overrides):
    kwargs = {
        "gene_probability": 0.0,
        "non_repeat_probability": 0.0,
        "te_rich_probability": 0.0,
        "small_variant_probability": 1.0,
        "structural_variant_probability": 0.0,
        "te_variant_probability": 0.0,
    }
    kwargs.update(overrides)
    return OneMaizeVariantTEMLMDataset(
        base_dir,
        variant_dir,
        tokenizer=_Tokenizer(),
        split="train",
        context_length=128,
        samples_per_epoch=2000,
        deterministic=True,
        allow_index_build=False,
        variant_jitter=32,
        return_metadata=True,
        **kwargs,
    )


def test_variant_sampler_is_uniform_by_genotype_and_deterministic(tmp_path):
    base_dir, variant_dir = _variant_dataset_fixture(tmp_path)
    dataset = _make_dataset(base_dir, variant_dir)
    counts = {"G1": 0, "G2": 0}
    for index in range(len(dataset)):
        metadata = dataset.sample_metadata(index)
        counts[metadata["genotype"]] += 1
        assert metadata["sampling_class"] == "small_variant"
    assert abs(counts["G1"] / len(dataset) - 0.5) < 0.04
    first = dataset[7]
    second = dataset[7]
    assert first[0].equal(second[0]) and first[1].equal(second[1])
    assert first[2] == second[2]
    assert first[2]["crop_start"] <= first[2]["event_start"] < first[2]["crop_end"]


def test_variant_validation_split_is_deterministic(tmp_path):
    base_dir, variant_dir = _variant_dataset_fixture(tmp_path)
    dataset = OneMaizeVariantTEMLMDataset(
        base_dir,
        variant_dir,
        tokenizer=_Tokenizer(),
        split="val",
        context_length=128,
        samples_per_epoch=8,
        deterministic=True,
        allow_index_build=False,
        return_metadata=True,
        variant_jitter=32,
    )
    first = [dataset[index] for index in range(len(dataset))]
    second = [dataset[index] for index in range(len(dataset))]
    assert all(a[0].equal(b[0]) and a[1].equal(b[1]) and a[2] == b[2] for a, b in zip(first, second))
    assert {item[2]["genotype"] for item in first} == {"G3"}


def test_variant_sampler_probability_validation_and_missing_class_policy(tmp_path):
    base_dir, variant_dir = _variant_dataset_fixture(tmp_path)
    with pytest.raises(ValueError, match="sum to 1"):
        _make_dataset(base_dir, variant_dir, small_variant_probability=0.9)
    table = pq.read_table(variant_dir / "variant_regions.parquet")
    filtered = table.filter(
        __import__("pyarrow.compute").compute.not_equal(
            table["variant_type"], "te_insertion"
        )
    )
    pq.write_table(filtered, variant_dir / "variant_regions.parquet")
    with pytest.raises(ValueError, match="Missing required genotype/class pools"):
        _make_dataset(
            base_dir,
            variant_dir,
            small_variant_probability=0.5,
            te_variant_probability=0.5,
        )
    dataset = _make_dataset(
        base_dir,
        variant_dir,
        small_variant_probability=0.5,
        te_variant_probability=0.5,
        missing_class_policy="renormalize",
    )
    assert dataset.sample_metadata(0)["sampling_class"] == "small_variant"


def test_variant_sampler_rejects_train_test_leakage(tmp_path):
    base_dir, variant_dir = _variant_dataset_fixture(tmp_path)
    table = pq.read_table(variant_dir / "variant_regions.parquet")
    rows = table.to_pylist()
    rows[0]["split"] = "test"
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), variant_dir / "variant_regions.parquet")
    with pytest.raises(ValueError, match="leakage"):
        _make_dataset(base_dir, variant_dir)


def test_variant_builder_and_audit_end_to_end(tmp_path):
    base_dir, _ = _variant_dataset_fixture(tmp_path)
    vcf = _write_vcf(
        tmp_path / "G1.vcf",
        [
            "chr1\t100\ts1\tA\tG\t.\tPASS\t.",
            "chr1\t500\tsv1\tN\t<INV>\t.\tPASS\tSVTYPE=INV;END=800",
            "chr1\t1200\tte1\tN\t<INS>\t.\tPASS\tSVTYPE=INS;ONEMAIZE_TYPE=te_insertion;SVLEN=200",
        ],
    )
    second_vcf = _write_vcf(
        tmp_path / "G1.second.vcf",
        ["chr1\t150\ts1\tC\tT\t.\tPASS\t."],
    )
    output = tmp_path / "built-v4"
    manifest = build_variant_metadata(
        base_dir,
        [
            VariantInput("G1", vcf, "fixture", "G1"),
            VariantInput("G1", second_vcf, "fixture-second", "G1"),
        ],
        output,
    )
    assert manifest["schema_version"] == 4
    assert manifest["variant_count"] == 4
    assert {row["variant_type"] for row in pq.read_table(output / "variant_regions.parquet").to_pylist()} == {
        "snp",
        "inversion",
        "te_insertion",
    }

    report, rows = audit_variant_metadata(
        base_dir,
        output,
        context_length=128,
        formal=False,
    )
    assert report["status"] == "PASS"
    assert report["variant_count"] == 4
    assert rows
    audit_output = tmp_path / "audit"
    write_audit_outputs(audit_output, report, rows)
    assert (audit_output / "VARIANT_INPUT_AUDIT.json").is_file()
    assert (audit_output / "VARIANT_INPUT_AUDIT.md").is_file()
    assert (audit_output / "VARIANT_SAMPLER_AUDIT.md").is_file()
    assert (audit_output / "VARIANT_SAMPLER_COUNTS.csv").is_file()


def test_variant_datamodule_setup_keeps_mlm_contract(tmp_path):
    base_dir, variant_dir = _variant_dataset_fixture(tmp_path)
    module = OneMaizeVariantTEDNAMLM(
        _name_="onemaize_variant_te_mlm",
        data_dir=base_dir,
        variant_data_dir=variant_dir,
        context_length=128,
        train_samples_per_epoch=2,
        val_samples_per_epoch=2,
        test_samples_per_epoch=2,
        allow_index_build=False,
        batch_size=1,
        variant_jitter=32,
    )
    module.setup()
    batch = next(iter(module.train_dataloader(num_workers=0)))
    assert module.mlm is True
    assert batch[0].shape == batch[1].shape == (1, 128)
    assert batch[2]["attention_mask"].shape == (1, 128)


def test_phase1_and_variant_phase2_model_configs_are_strictly_compatible():
    root = Path(__file__).resolve().parents[1]
    phase1 = OmegaConf.load(root / "configs/experiment/onemaize_b73_phase1_8k_full_genome.yaml")
    model_b = OmegaConf.load(root / "configs/experiment/onemaize_allcultivar_phase2_variant_te_16k.yaml")
    assert OmegaConf.to_container(phase1.model, resolve=False) == OmegaConf.to_container(
        model_b.model, resolve=False
    )
    assert model_b.train.pretrained_model_strict_load is True
