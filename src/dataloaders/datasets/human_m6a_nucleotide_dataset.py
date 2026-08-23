"""Full-mRNA datasets for nucleotide-level m6A classification."""

from __future__ import annotations

import bisect
import gzip
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

IGNORE_INDEX = -100.0


def collate_full_transcripts(
    batch,
    pad_token_id: int,
    pad_to_multiple: int = 8,
):
    """Dynamically right-pad complete transcripts and return model metadata."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if pad_to_multiple <= 0:
        raise ValueError("pad_to_multiple must be positive")
    input_items, label_items = zip(*batch)
    lengths = torch.tensor([item.numel() for item in input_items], dtype=torch.long)
    padded_length = int(lengths.max().item())
    remainder = padded_length % int(pad_to_multiple)
    if remainder:
        padded_length += int(pad_to_multiple) - remainder

    input_ids = torch.full(
        (len(batch), padded_length), int(pad_token_id), dtype=torch.long
    )
    labels = torch.full(
        (len(batch), padded_length), float(IGNORE_INDEX), dtype=torch.float32
    )
    attention_mask = torch.zeros((len(batch), padded_length), dtype=torch.bool)
    for row, (item, target) in enumerate(zip(input_items, label_items)):
        length = item.numel()
        if target.numel() != length:
            raise ValueError("Input and target lengths must match")
        input_ids[row, :length] = item
        labels[row, :length] = target
        attention_mask[row, :length] = True
    return input_ids, labels, {"attention_mask": attention_mask, "lengths": lengths}


def window_starts(sequence_length: int, window_length: int, stride: int) -> List[int]:
    """Generate starts while guaranteeing that one window covers the tail."""

    if sequence_length <= 0:
        return []
    if window_length <= 0 or stride <= 0:
        raise ValueError("window_length and stride must be positive")
    if sequence_length <= window_length:
        return [0]
    starts = list(range(0, sequence_length - window_length + 1, stride))
    tail_start = sequence_length - window_length
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def window_ownership(
    sequence_length: int,
    window_length: int,
    stride: int,
) -> List[Tuple[int, int, int, int]]:
    """Return windows plus a non-overlapping loss interval for each window.

    Overlap supplies context, while midpoint boundaries make every nucleotide
    contribute to the training loss exactly once.
    """

    if stride > window_length:
        raise ValueError("stride must not exceed window_length")
    starts = window_starts(sequence_length, window_length, stride)
    if not starts:
        return []
    ends = [min(start + window_length, sequence_length) for start in starts]
    centers = [(start + end) / 2.0 for start, end in zip(starts, ends)]
    boundaries = [0]
    boundaries.extend(int((centers[index - 1] + centers[index]) / 2.0) for index in range(1, len(starts)))
    boundaries.append(sequence_length)

    layout = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        loss_start = boundaries[index]
        loss_end = boundaries[index + 1]
        if not (start <= loss_start <= loss_end <= end):
            raise RuntimeError("Window overlap is insufficient to define unique loss ownership")
        layout.append((start, end, loss_start, loss_end))
    return layout


class HumanM6ANucleotideDataset(torch.utils.data.Dataset):
    """Lazily construct fixed-length token windows and A-only binary labels."""

    def __init__(
        self,
        jsonl_file,
        tokenizer,
        window_length: int = 1024,
        stride: int = 512,
        require_mrna_coordinate_reliable: bool = True,
        max_transcripts: Optional[int] = None,
        max_windows: Optional[int] = None,
    ):
        self.jsonl_file = Path(jsonl_file)
        self.tokenizer = tokenizer
        self.window_length = int(window_length)
        self.max_length = self.window_length
        self.stride = int(stride)
        self.require_mrna_coordinate_reliable = bool(require_mrna_coordinate_reliable)
        if self.window_length <= 0 or self.stride <= 0 or self.stride > self.window_length:
            raise ValueError("Require 0 < stride <= window_length")

        self.records: List[Mapping[str, object]] = []
        opener = gzip.open if self.jsonl_file.suffix == ".gz" else open
        with opener(self.jsonl_file, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_record(record, line_number)
                if self.require_mrna_coordinate_reliable and not bool(
                    record["mrna_coordinate_system_reliable"]
                ):
                    continue
                self.records.append(record)
                if max_transcripts is not None and len(self.records) >= int(max_transcripts):
                    break

        self.window_index: List[Tuple[int, int, int, int, int]] = []
        for record_index, record in enumerate(self.records):
            sequence = str(record["sequence"])
            for start, end, loss_start, loss_end in window_ownership(
                len(sequence), self.window_length, self.stride
            ):
                if "A" in sequence[loss_start:loss_end]:
                    self.window_index.append((record_index, start, end, loss_start, loss_end))
        if max_windows is not None:
            if int(max_windows) <= 0:
                raise ValueError("max_windows must be positive")
            self.window_index = self.window_index[: int(max_windows)]

        self.candidate_count, self.positive_count = self._label_counts()
        self.negative_count = self.candidate_count - self.positive_count
        self.positive_weight = self.negative_count / max(1, self.positive_count)

        vocab = tokenizer.get_vocab()
        self._byte_to_token = np.full(256, int(tokenizer.unk_token_id), dtype=np.int64)
        for base in "ACGUN":
            self._byte_to_token[ord(base)] = int(vocab[base])

    @staticmethod
    def _validate_record(record: Mapping[str, object], line_number: int) -> None:
        required = {
            "transcript_id",
            "gene_id",
            "sequence",
            "m6a_positions",
            "mrna_coordinate_system_reliable",
        }
        missing = required - set(record)
        if missing:
            raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
        sequence = str(record["sequence"])
        invalid = set(sequence) - set("ACGUN")
        if invalid:
            raise ValueError(f"Line {line_number} contains invalid RNA symbols: {sorted(invalid)}")
        positions = list(record["m6a_positions"])
        if positions != sorted(set(positions)):
            raise ValueError(f"Line {line_number} has unsorted or repeated m6A positions")
        if any(not isinstance(position, int) or position < 0 or position >= len(sequence) for position in positions):
            raise ValueError(f"Line {line_number} has an out-of-range m6A position")
        if any(sequence[position] != "A" for position in positions):
            raise ValueError(f"Line {line_number} maps m6A to a non-A base")

    def _label_counts(self) -> Tuple[int, int]:
        candidates = positives = 0
        for record_index, _, _, loss_start, loss_end in self.window_index:
            record = self.records[record_index]
            sequence = str(record["sequence"])
            positions = record["m6a_positions"]
            candidates += sequence[loss_start:loss_end].count("A")
            positives += bisect.bisect_left(positions, loss_end) - bisect.bisect_left(
                positions, loss_start
            )
        return candidates, positives

    def __len__(self) -> int:
        return len(self.window_index)

    def window_metadata(self, index: int) -> Dict[str, object]:
        record_index, start, end, loss_start, loss_end = self.window_index[index]
        record = self.records[record_index]
        return {
            "transcript_id": record["transcript_id"],
            "gene_id": record["gene_id"],
            "start": start,
            "end": end,
            "valid_length": end - start,
            "loss_start": loss_start,
            "loss_end": loss_end,
        }

    def __getitem__(self, index: int):
        record_index, start, end, loss_start, loss_end = self.window_index[index]
        record = self.records[record_index]
        sequence = str(record["sequence"])
        window = sequence[start:end]

        input_ids = torch.full(
            (self.window_length,), int(self.tokenizer.pad_token_id), dtype=torch.long
        )
        raw = np.frombuffer(window.encode("ascii"), dtype=np.uint8)
        input_ids[: len(window)] = torch.from_numpy(self._byte_to_token[raw].copy())

        labels = torch.full((self.window_length,), IGNORE_INDEX, dtype=torch.float32)
        owned = sequence[loss_start:loss_end]
        owned_raw = np.frombuffer(owned.encode("ascii"), dtype=np.uint8)
        candidate_local = np.flatnonzero(owned_raw == ord("A")) + (loss_start - start)
        if candidate_local.size:
            labels[torch.from_numpy(candidate_local.astype(np.int64))] = 0.0

        positions = record["m6a_positions"]
        left = bisect.bisect_left(positions, loss_start)
        right = bisect.bisect_left(positions, loss_end)
        for position in positions[left:right]:
            labels[int(position) - start] = 1.0

        return input_ids, labels


class HumanM6AFullTranscriptDataset(torch.utils.data.Dataset):
    """Return one complete, untruncated mRNA per item.

    Transcripts longer than ``max_sequence_length`` are excluded or rejected;
    they are never silently truncated. Padding is deferred to the data-module
    collator so every item retains its true biological length.
    """

    REQUIRED_REGION_FIELDS = {
        "utr5_start",
        "utr5_end",
        "cds_start",
        "cds_end",
        "utr3_start",
        "utr3_end",
        "cds_boundary_reliable",
    }

    def __init__(
        self,
        jsonl_file,
        tokenizer,
        max_sequence_length: Optional[int] = 10240,
        overlength_policy: str = "exclude",
        require_mrna_coordinate_reliable: bool = True,
        require_cds_boundary_reliable: bool = True,
        max_transcripts: Optional[int] = None,
    ):
        self.jsonl_file = Path(jsonl_file)
        self.tokenizer = tokenizer
        self.max_sequence_length = (
            None if max_sequence_length is None else int(max_sequence_length)
        )
        self.max_length = self.max_sequence_length
        self.overlength_policy = str(overlength_policy).lower()
        self.require_mrna_coordinate_reliable = bool(require_mrna_coordinate_reliable)
        self.require_cds_boundary_reliable = bool(require_cds_boundary_reliable)
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive or null")
        if self.overlength_policy not in {"exclude", "error"}:
            raise ValueError("overlength_policy must be 'exclude' or 'error'")

        self.records: List[Mapping[str, object]] = []
        self.excluded_overlength = 0
        self.excluded_unreliable_mrna = 0
        self.excluded_unreliable_cds = 0
        opener = gzip.open if self.jsonl_file.suffix == ".gz" else open
        with opener(self.jsonl_file, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_record(record, line_number)
                if self.require_mrna_coordinate_reliable and not bool(
                    record["mrna_coordinate_system_reliable"]
                ):
                    self.excluded_unreliable_mrna += 1
                    continue
                if self.require_cds_boundary_reliable and not bool(
                    record["cds_boundary_reliable"]
                ):
                    self.excluded_unreliable_cds += 1
                    continue
                sequence_length = len(str(record["sequence"]))
                if (
                    self.max_sequence_length is not None
                    and sequence_length > self.max_sequence_length
                ):
                    if self.overlength_policy == "error":
                        raise ValueError(
                            f"Line {line_number} transcript {record['transcript_id']} has length "
                            f"{sequence_length} > max_sequence_length={self.max_sequence_length}"
                        )
                    self.excluded_overlength += 1
                    continue
                if "A" not in str(record["sequence"]):
                    continue
                self.records.append(record)
                if max_transcripts is not None and len(self.records) >= int(max_transcripts):
                    break

        self.sequence_lengths = [len(str(record["sequence"])) for record in self.records]
        self.candidate_count = sum(str(record["sequence"]).count("A") for record in self.records)
        self.positive_count = sum(len(record["m6a_positions"]) for record in self.records)
        self.negative_count = self.candidate_count - self.positive_count
        self.positive_weight = self.negative_count / max(1, self.positive_count)

        vocab = tokenizer.get_vocab()
        self._byte_to_token = np.full(256, int(tokenizer.unk_token_id), dtype=np.int64)
        for base in "ACGUN":
            self._byte_to_token[ord(base)] = int(vocab[base])

    @classmethod
    def _validate_record(cls, record: Mapping[str, object], line_number: int) -> None:
        HumanM6ANucleotideDataset._validate_record(record, line_number)
        missing = cls.REQUIRED_REGION_FIELDS - set(record)
        if missing:
            raise ValueError(f"Line {line_number} is missing region fields: {sorted(missing)}")
        sequence_length = len(str(record["sequence"]))
        boundaries = [
            int(record["utr5_start"]),
            int(record["utr5_end"]),
            int(record["cds_start"]),
            int(record["cds_end"]),
            int(record["utr3_start"]),
            int(record["utr3_end"]),
        ]
        if not (
            boundaries[0] == 0
            and boundaries[1] == boundaries[2]
            and boundaries[3] == boundaries[4]
            and boundaries[5] == sequence_length
            and boundaries == sorted(boundaries)
        ):
            raise ValueError(f"Line {line_number} has inconsistent full-mRNA region boundaries")

    def __len__(self) -> int:
        return len(self.records)

    def transcript_metadata(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        return {
            "transcript_id": record["transcript_id"],
            "gene_id": record["gene_id"],
            "length": len(str(record["sequence"])),
            "utr5": (record["utr5_start"], record["utr5_end"]),
            "cds": (record["cds_start"], record["cds_end"]),
            "utr3": (record["utr3_start"], record["utr3_end"]),
        }

    def __getitem__(self, index: int):
        record = self.records[index]
        sequence = str(record["sequence"])
        raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        input_ids = torch.from_numpy(self._byte_to_token[raw].copy())

        labels = torch.full((len(sequence),), IGNORE_INDEX, dtype=torch.float32)
        candidate_positions = np.flatnonzero(raw == ord("A"))
        if candidate_positions.size:
            labels[torch.from_numpy(candidate_positions.astype(np.int64))] = 0.0
        for position in record["m6a_positions"]:
            labels[int(position)] = 1.0
        return input_ids, labels
