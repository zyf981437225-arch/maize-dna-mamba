"""Sliding-window dataset for observed human m6A site counts."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch


def window_starts(sequence_length: int, window_length: int, stride: int) -> List[int]:
    """Generate starts while guaranteeing one window covers the sequence tail."""

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


class HumanM6AWindowDataset(torch.utils.data.Dataset):
    """Build fixed-length windows lazily from transcript-level JSONL records.

    Each target is the count of observed, uniquely mapped m6A sites in the
    window.  Site positions and window coordinates are zero-based and the end
    coordinate is exclusive.
    """

    def __init__(
        self,
        jsonl_file,
        tokenizer,
        window_length: int = 128,
        stride: int = 64,
        target_transform: str = "none",
        include_tail: bool = True,
        max_windows: Optional[int] = None,
    ):
        self.jsonl_file = Path(jsonl_file)
        self.tokenizer = tokenizer
        self.window_length = int(window_length)
        self.max_length = self.window_length
        self.stride = int(stride)
        self.target_transform = target_transform
        self.include_tail = bool(include_tail)

        if self.window_length <= 0 or self.stride <= 0:
            raise ValueError("window_length and stride must be positive")
        if target_transform not in {"none", "log1p"}:
            raise ValueError("target_transform must be 'none' or 'log1p'")

        self.records: List[Mapping[str, object]] = []
        self.window_index: List[Tuple[int, int]] = []
        with self.jsonl_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate_record(record, line_number)
                record_index = len(self.records)
                self.records.append(record)
                sequence_length = len(record["sequence"])
                starts = window_starts(sequence_length, self.window_length, self.stride)
                if not self.include_tail and sequence_length > self.window_length:
                    starts = list(range(0, sequence_length - self.window_length + 1, self.stride))
                self.window_index.extend((record_index, start) for start in starts)

        if max_windows is not None:
            if max_windows <= 0:
                raise ValueError("max_windows must be positive when provided")
            self.window_index = self.window_index[: int(max_windows)]

    @staticmethod
    def _validate_record(record: Mapping[str, object], line_number: int) -> None:
        required = {"transcript_id", "gene_id", "sequence", "observed_m6a_positions"}
        missing = required - set(record)
        if missing:
            raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
        sequence = str(record["sequence"])
        invalid = sorted(set(sequence) - set("ACGUN"))
        if invalid:
            raise ValueError(f"Line {line_number} contains invalid RNA symbols: {invalid}")
        positions = list(record["observed_m6a_positions"])
        if positions != sorted(set(positions)):
            raise ValueError(f"Line {line_number} has unsorted or repeated site positions")
        if any(not isinstance(position, int) or position < 0 or position >= len(sequence) for position in positions):
            raise ValueError(f"Line {line_number} has an out-of-range site position")
        if any(sequence[position] != "A" for position in positions):
            raise ValueError(f"Line {line_number} maps an m6A site to a non-A base")

    def __len__(self) -> int:
        return len(self.window_index)

    def _site_count(self, record: Mapping[str, object], start: int, end: int) -> int:
        positions: Sequence[int] = record["observed_m6a_positions"]
        left = bisect.bisect_left(positions, start)
        right = bisect.bisect_left(positions, end)
        return right - left

    def window_metadata(self, index: int) -> Dict[str, object]:
        """Return auditable coordinates and the untransformed target."""

        record_index, start = self.window_index[index]
        record = self.records[record_index]
        end = min(start + self.window_length, len(record["sequence"]))
        return {
            "transcript_id": record["transcript_id"],
            "gene_id": record["gene_id"],
            "start": start,
            "end": end,
            "valid_length": end - start,
            "observed_m6a_count": self._site_count(record, start, end),
        }

    def __getitem__(self, index: int):
        record_index, start = self.window_index[index]
        record = self.records[record_index]
        sequence = str(record["sequence"])
        end = min(start + self.window_length, len(sequence))
        window = sequence[start:end]
        valid_length = len(window)
        observed_count = self._site_count(record, start, end)

        encoded = self.tokenizer(
            window,
            add_special_tokens=False,
            padding="max_length",
            max_length=self.window_length,
            truncation=True,
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        target = float(observed_count)
        if self.target_transform == "log1p":
            target = math.log1p(target)

        return (
            input_ids,
            torch.tensor([target], dtype=torch.float32),
            torch.tensor(valid_length, dtype=torch.long),
        )
