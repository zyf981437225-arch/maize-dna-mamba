"""Explicit variant- and transposon-aware OneMaize Phase-II dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from src.onemaize.variants import (
    SMALL_VARIANT_TYPES,
    STRUCTURAL_VARIANT_TYPES,
    TE_VARIANT_TYPES,
    read_fai_lengths,
)

from .onemaize_dataset import OneMaizeRegionMLMDataset, reverse_complement


SAMPLING_CLASSES = (
    "gene_centered",
    "non_repeat",
    "te_rich",
    "small_variant",
    "structural_variant",
    "te_variant",
)
BASE_SAMPLING_CLASSES = SAMPLING_CLASSES[:3]
VARIANT_SAMPLING_CLASSES = SAMPLING_CLASSES[3:]


def event_sampling_class(event: dict, small_variant_max_length: int = 50) -> str:
    variant_type = str(event["variant_type"])
    if variant_type in TE_VARIANT_TYPES:
        return "te_variant"
    if (
        variant_type in SMALL_VARIANT_TYPES
        and abs(int(event["variant_length"])) <= int(small_variant_max_length)
    ):
        return "small_variant"
    if variant_type in STRUCTURAL_VARIANT_TYPES:
        return "structural_variant"
    raise ValueError(
        f"Variant {event.get('variant_id')} cannot be assigned to a sampling class: "
        f"type={variant_type!r}"
    )


def crop_for_event(
    event: dict,
    *,
    context_length: int,
    sequence_length: int,
    rng: np.random.Generator,
    jitter: int,
) -> dict:
    """Choose a chromosome-safe crop that retains the event or one SV breakpoint."""

    context_length = int(context_length)
    sequence_length = int(sequence_length)
    if sequence_length < context_length:
        raise ValueError("Sequence is shorter than the requested context")
    start, end = int(event["start"]), int(event["end"])
    if start < 0 or end < start or (end > sequence_length):
        raise ValueError(f"Invalid event coordinates: {start}-{end}/{sequence_length}")

    span = end - start
    if span > context_length:
        breakpoints = [
            int(event.get("left_breakpoint") if event.get("left_breakpoint") is not None else start),
            int(event.get("right_breakpoint") if event.get("right_breakpoint") is not None else end),
        ]
        side = int(rng.integers(0, 2))
        boundary = breakpoints[side]
        anchor = min(max(0, boundary), sequence_length - 1)
        target_start = target_end = anchor
        subtype = "left_breakpoint" if side == 0 else "right_breakpoint"
    else:
        target_start, target_end = start, end
        anchor = start if start == end else (start + end - 1) // 2
        subtype = "point" if start == end else "span"

    if target_start == target_end:
        lower = max(0, anchor - context_length + 1)
        upper = min(anchor, sequence_length - context_length)
    else:
        lower = max(0, target_end - context_length)
        upper = min(target_start, sequence_length - context_length)
    if lower > upper:
        raise ValueError(
            f"Event cannot fit in context: event={start}-{end}, context={context_length}"
        )

    centered = anchor - context_length // 2
    offset = int(rng.integers(-int(jitter), int(jitter) + 1)) if jitter else 0
    crop_start = min(max(centered + offset, lower), upper)
    crop_end = crop_start + context_length
    if not (0 <= crop_start < crop_end <= sequence_length):
        raise AssertionError("Internal crop crossed a chromosome boundary")
    if target_start == target_end:
        contains_target = crop_start <= anchor < crop_end
    else:
        contains_target = crop_start <= target_start and target_end <= crop_end
    if not contains_target:
        raise AssertionError("Variant-aware crop lost its target event")
    return {
        "crop_start": crop_start,
        "crop_end": crop_end,
        "sampling_subtype": subtype,
        "target_start": target_start,
        "target_end": target_end,
    }


class OneMaizeVariantTEMLMDataset(OneMaizeRegionMLMDataset):
    """Uniform-genotype Phase-II sampling with explicit variant event pools."""

    def __init__(
        self,
        data_dir,
        variant_data_dir,
        tokenizer,
        split: str,
        context_length: int = 16384,
        samples_per_epoch: int = 100_000,
        gene_probability: float = 0.20,
        non_repeat_probability: float = 0.15,
        te_rich_probability: float = 0.15,
        small_variant_probability: float = 0.20,
        structural_variant_probability: float = 0.20,
        te_variant_probability: float = 0.10,
        missing_class_policy: str = "error",
        small_variant_max_length: int = 50,
        variant_jitter: int = 4096,
        reverse_complement_probability: float = 0.5,
        mlm_probability: float = 0.15,
        deterministic: bool = False,
        seed: int = 2357,
        allow_index_build: bool = False,
        fasta_root: Optional[str] = None,
        max_n_fraction: Optional[float] = None,
        max_crop_attempts: int = 16,
        return_metadata: bool = False,
        variant_type_filter: Optional[Iterable[str]] = None,
    ) -> None:
        probabilities = np.asarray(
            [
                gene_probability,
                non_repeat_probability,
                te_rich_probability,
                small_variant_probability,
                structural_variant_probability,
                te_variant_probability,
            ],
            dtype=np.float64,
        )
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Sampling probabilities must be non-negative and sum to 1")
        base = probabilities[:3]
        normalized_base = base / base.sum() if base.sum() else np.asarray([1.0, 0.0, 0.0])
        super().__init__(
            data_dir,
            tokenizer=tokenizer,
            split=split,
            context_length=context_length,
            samples_per_epoch=samples_per_epoch,
            gene_probability=float(normalized_base[0]),
            non_repeat_probability=float(normalized_base[1]),
            te_rich_probability=float(normalized_base[2]),
            reverse_complement_probability=reverse_complement_probability,
            mlm_probability=mlm_probability,
            deterministic=deterministic,
            seed=seed,
            allow_index_build=allow_index_build,
            fasta_root=fasta_root,
            max_n_fraction=max_n_fraction,
            max_crop_attempts=max_crop_attempts,
        )
        self.sampling_probabilities = probabilities
        self.missing_class_policy = str(missing_class_policy).lower()
        if self.missing_class_policy not in {"error", "renormalize"}:
            raise ValueError("missing_class_policy must be error or renormalize")
        self.small_variant_max_length = int(small_variant_max_length)
        if self.small_variant_max_length <= 0:
            raise ValueError("small_variant_max_length must be positive")
        self.variant_jitter = int(variant_jitter)
        if not 0 <= self.variant_jitter < self.context_length:
            raise ValueError("variant_jitter must be in [0, context_length)")
        self.return_metadata = bool(return_metadata)

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Reading variant metadata requires pyarrow") from exc

        self.variant_data_dir = Path(variant_data_dir).expanduser().resolve()
        variant_manifest = json.loads(
            (self.variant_data_dir / "variant_manifest.json").read_text(encoding="utf-8")
        )
        if int(variant_manifest.get("schema_version", -1)) != 4:
            raise ValueError("Variant-aware sampling requires schema-v4 metadata")
        if int(variant_manifest.get("base_schema_version", -1)) != 3:
            raise ValueError("schema-v4 metadata must reference schema-v3 candidates")
        if variant_manifest.get("coordinate_system") != "0-based-half-open":
            raise ValueError("Variant metadata must use zero-based half-open coordinates")

        variant_columns = [
            "variant_id",
            "genotype",
            "reference_genotype",
            "coordinate_genotype",
            "seqid",
            "start",
            "end",
            "variant_type",
            "variant_length",
            "split",
            "left_breakpoint",
            "right_breakpoint",
            "te_family",
        ]
        variant_path = self.variant_data_dir / variant_manifest["files"]["variants"]
        all_variants = pq.read_table(
            variant_path,
            columns=["variant_id", "genotype", "coordinate_genotype", "split"],
        ).to_pylist()
        split_by_genotype = {
            row["genotype"]: row["default_split"]
            for row in pq.read_table(
                self.data_dir / self.manifest["files"]["genomes"],
                columns=["genotype", "default_split"],
            ).to_pylist()
        }
        leakage = [
            row["variant_id"]
            for row in all_variants
            if row["coordinate_genotype"] != row["genotype"]
            or split_by_genotype.get(row["genotype"]) != row["split"]
        ]
        if leakage:
            raise ValueError(
                "Variant coordinate/split leakage detected: " + ", ".join(leakage[:10])
            )

        table = pq.read_table(
            variant_path,
            columns=variant_columns,
            filters=[("split", "=", self.split)],
        )
        allowed_variant_types = (
            None
            if variant_type_filter is None
            else {str(item) for item in variant_type_filter}
        )
        self.variant_rows = [
            row
            for row in table.to_pylist()
            if allowed_variant_types is None or row["variant_type"] in allowed_variant_types
        ]
        self.variant_grouped_indices: dict[tuple[str, str], np.ndarray] = {}
        for genotype in self.genotypes:
            for sampling_class in VARIANT_SAMPLING_CLASSES:
                indices = np.asarray(
                    [
                        index
                        for index, row in enumerate(self.variant_rows)
                        if row["genotype"] == genotype
                        and event_sampling_class(row, self.small_variant_max_length)
                        == sampling_class
                    ],
                    dtype=np.int64,
                )
                if indices.size:
                    self.variant_grouped_indices[(genotype, sampling_class)] = indices

        self.sequence_lengths = {}
        for genotype, fasta_path in self._fasta_store.paths.items():
            fai = Path(f"{fasta_path}.fai")
            if not fai.is_file():
                raise FileNotFoundError(fai)
            self.sequence_lengths[genotype] = read_fai_lengths(fai)

        self.class_probabilities_by_genotype = {}
        missing = []
        for genotype in self.genotypes:
            available = np.asarray(
                [self._pool_exists(genotype, item) for item in SAMPLING_CLASSES],
                dtype=bool,
            )
            active_missing = [
                sampling_class
                for sampling_class, probability, exists in zip(
                    SAMPLING_CLASSES, probabilities, available
                )
                if probability > 0 and not exists
            ]
            if active_missing and self.missing_class_policy == "error":
                missing.extend(f"{genotype}/{item}" for item in active_missing)
                continue
            adjusted = probabilities * available
            if adjusted.sum() <= 0:
                raise ValueError(f"No enabled sampling class for genotype {genotype}")
            self.class_probabilities_by_genotype[genotype] = adjusted / adjusted.sum()
        if missing:
            raise ValueError(
                "Missing required genotype/class pools: " + ", ".join(missing)
            )
        self.variant_source_counts = {
            sampling_class: sum(
                len(self.variant_grouped_indices.get((genotype, sampling_class), ()))
                for genotype in self.genotypes
            )
            for sampling_class in VARIANT_SAMPLING_CLASSES
        }

    def _pool_exists(self, genotype: str, sampling_class: str) -> bool:
        if sampling_class in BASE_SAMPLING_CLASSES:
            return (genotype, sampling_class) in self.grouped_indices
        return (genotype, sampling_class) in self.variant_grouped_indices

    def _sample_pool(self, rng: np.random.Generator) -> tuple[str, str]:
        genotype = self.genotypes[int(rng.integers(0, len(self.genotypes)))]
        probabilities = self.class_probabilities_by_genotype[genotype]
        class_index = int(rng.choice(len(SAMPLING_CLASSES), p=probabilities))
        return genotype, SAMPLING_CLASSES[class_index]

    def _sample_spec(
        self,
        rng: np.random.Generator,
        genotype: str,
        sampling_class: str,
    ) -> dict:
        if sampling_class in BASE_SAMPLING_CLASSES:
            row = self._sample_region_from_pool(rng, genotype, sampling_class)
            max_start = int(row["end"]) - self.context_length
            crop_start = int(rng.integers(int(row["start"]), max_start + 1))
            return {
                "genotype": genotype,
                "seqid": row["seqid"],
                "sampling_class": sampling_class,
                "crop_start": crop_start,
                "crop_end": crop_start + self.context_length,
                "region_id": row["region_id"],
                "variant_id": None,
                "variant_type": None,
                "sampling_subtype": "dynamic_crop",
                "target_start": None,
                "target_end": None,
            }

        pool = self.variant_grouped_indices[(genotype, sampling_class)]
        event = self.variant_rows[int(pool[int(rng.integers(0, len(pool)))])]
        seqid = event["seqid"]
        if seqid not in self.sequence_lengths[genotype]:
            raise ValueError(
                f"Variant {event['variant_id']} seqid {seqid!r} is absent from FASTA index"
            )
        crop = crop_for_event(
            event,
            context_length=self.context_length,
            sequence_length=self.sequence_lengths[genotype][seqid],
            rng=rng,
            jitter=self.variant_jitter,
        )
        return {
            "genotype": genotype,
            "seqid": seqid,
            "sampling_class": sampling_class,
            "region_id": None,
            "variant_id": event["variant_id"],
            "variant_type": event["variant_type"],
            "event_start": int(event["start"]),
            "event_end": int(event["end"]),
            "te_family": event.get("te_family"),
            **crop,
        }

    def sample_metadata(self, index: int) -> dict:
        rng = self._rng_for_index(index)
        genotype, sampling_class = self._sample_pool(rng)
        return self._sample_spec(rng, genotype, sampling_class)

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = self._rng_for_index(index)
        genotype, sampling_class = self._sample_pool(rng)
        sequence = None
        metadata = None
        last_location = None
        for _ in range(self.max_crop_attempts):
            metadata = self._sample_spec(rng, genotype, sampling_class)
            candidate = self._fasta_store.fetch(
                metadata["genotype"],
                metadata["seqid"],
                metadata["crop_start"],
                metadata["crop_end"],
            )
            last_location = (
                f"{metadata['genotype']}/{metadata['seqid']}:"
                f"{metadata['crop_start']}-{metadata['crop_end']}"
            )
            if len(candidate) != self.context_length:
                raise ValueError(
                    f"Expected {self.context_length} bp, fetched {len(candidate)} "
                    f"for {last_location}"
                )
            candidate = candidate.upper()
            if set(candidate) - set("ACGTN"):
                raise ValueError(f"Non-ACGTN sequence fetched for {last_location}")
            if candidate.count("N") / self.context_length <= self.max_n_fraction:
                sequence = candidate
                break
            self.filtered_windows += 1
        if sequence is None or metadata is None:
            raise ValueError(
                f"Could not sample a valid crop after {self.max_crop_attempts} attempts "
                f"from {genotype}/{sampling_class}; last={last_location}"
            )
        reverse_complemented = bool(
            rng.random() < self.reverse_complement_probability
        )
        if reverse_complemented:
            sequence = reverse_complement(sequence)
        raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        token_ids = self._byte_to_token[raw]
        if np.any(token_ids == int(self.tokenizer.unk_token_id)):
            raise ValueError("Normalized OneMaize sample contains an unknown token")
        corrupted, labels = self._mask(token_ids, rng)
        if not self.return_metadata:
            return corrupted.long(), labels.long()
        metadata = dict(metadata)
        metadata["reverse_complemented"] = reverse_complemented
        return corrupted.long(), labels.long(), metadata
