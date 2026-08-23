"""Indexed genomic-DNA windows for same-position masked language modelling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class IndexedGenomicDNAMLMDataset(torch.utils.data.Dataset):
    """Read fixed-length, contig-safe DNA windows from a memory-mapped corpus."""

    SUPPORTED_SCHEMA_VERSIONS = {1}

    def __init__(
        self,
        data_dir,
        tokenizer,
        split: str,
        window_size: Optional[int] = None,
        mlm_probability: float = 0.15,
        deterministic_mlm: bool = False,
        seed: int = 2357,
        max_windows: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = str(split).lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not 0.0 < float(mlm_probability) <= 1.0:
            raise ValueError("mlm_probability must be in (0, 1]")

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing DNA corpus manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        schema_version = int(manifest.get("schema_version", -1))
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported DNA corpus schema {schema_version}; expected one of "
                f"{sorted(self.SUPPORTED_SCHEMA_VERSIONS)}"
            )
        if str(manifest.get("sequence_type", "")).lower() != "dna":
            raise ValueError("Prepared corpus is not marked as sequence_type=dna")
        if manifest.get("alphabet") != "ACGTN":
            raise ValueError("Prepared DNA corpus must declare alphabet ACGTN")
        sanity_checks = manifest.get("sanity_checks", {})
        if not bool(sanity_checks.get("u_fraction_passed", False)):
            raise ValueError(
                "Prepared DNA corpus failed the U-frequency sanity check; inspect "
                f"{self.data_dir / 'DATA_STATS.md'} before training"
            )
        if self.split not in manifest.get("splits", {}):
            raise ValueError(f"Manifest does not define split {self.split!r}")

        self.manifest = manifest
        self.max_sequence_length = int(manifest["window_size"])
        self.max_length = self.max_sequence_length
        if window_size is not None and int(window_size) != self.max_sequence_length:
            raise ValueError(
                f"Prepared window size is {self.max_sequence_length}, but the dataset "
                f"config requests {int(window_size)}"
            )

        split_info = manifest["splits"][self.split]
        files = split_info["files"]
        self.sequence_path = self.data_dir / files["sequences"]
        self.offset_path = self.data_dir / files["offsets"]
        self.metadata_path = self.data_dir / files["metadata"]
        for path in (self.sequence_path, self.offset_path, self.metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing indexed DNA corpus file: {path}")

        self._record_count = int(split_info["windows"])
        expected_offset_bytes = (self._record_count + 1) * np.dtype("<u8").itemsize
        if self.offset_path.stat().st_size != expected_offset_bytes:
            raise ValueError(
                f"Invalid offset index size for {self.split}: expected "
                f"{expected_offset_bytes}, got {self.offset_path.stat().st_size}"
            )
        with self.offset_path.open("rb") as handle:
            handle.seek(-np.dtype("<u8").itemsize, 2)
            indexed_sequence_bytes = int.from_bytes(handle.read(8), "little")
        if indexed_sequence_bytes != self.sequence_path.stat().st_size:
            raise ValueError(
                f"Invalid final offset for {self.split}: index reports "
                f"{indexed_sequence_bytes}, sequence file has "
                f"{self.sequence_path.stat().st_size} bytes"
            )
        expected_sequence_bytes = self._record_count * self.max_sequence_length
        if indexed_sequence_bytes != expected_sequence_bytes:
            raise ValueError(
                f"Split {self.split} must contain fixed {self.max_sequence_length}-bp "
                f"windows; expected {expected_sequence_bytes} sequence bytes, got "
                f"{indexed_sequence_bytes}"
            )

        self.tokenizer = tokenizer
        self.mlm_probability = float(mlm_probability)
        self.deterministic_mlm = bool(deterministic_mlm)
        self.seed = int(seed)
        self.pad_id = int(tokenizer.pad_token_id)
        self.mask_id = int(tokenizer.mask_token_id)
        vocab = tokenizer.get_vocab()
        self.random_token_ids = torch.tensor(
            [vocab[base] for base in "ACGTN" if base in vocab], dtype=torch.long
        )
        self.predictable_token_ids = torch.tensor(
            [vocab[base] for base in "ACGT" if base in vocab], dtype=torch.long
        )
        if self.random_token_ids.numel() != 5 or self.predictable_token_ids.numel() != 4:
            raise ValueError("DNA tokenizer must contain A, C, G, T, and N")
        if "U" in vocab:
            raise ValueError("DNA tokenizer must not contain U")

        self._byte_to_token = np.full(256, int(tokenizer.unk_token_id), dtype=np.int64)
        for base in "ACGTN":
            self._byte_to_token[ord(base)] = int(vocab[base])

        self._sequence_bytes = None
        self._offsets = None
        self._selection = None
        if max_windows is not None and int(max_windows) < self._record_count:
            limit = int(max_windows)
            if limit < 0:
                raise ValueError("max_windows cannot be negative")
            split_offset = {"train": 0, "val": 1, "test": 2}[self.split]
            rng = np.random.default_rng(self.seed + split_offset)
            self._selection = rng.choice(
                self._record_count, size=limit, replace=False
            ).astype(np.int64)

        self.source_counts = {"genomic_dna": self._record_count}
        self.nucleotides = int(split_info.get("nucleotides", expected_sequence_bytes))
        self.filtered_windows = int(split_info.get("filtered_windows", 0))

    def _ensure_open(self) -> None:
        if self._offsets is None:
            self._offsets = np.memmap(self.offset_path, mode="r", dtype="<u8")
        if self._sequence_bytes is None:
            if self.sequence_path.stat().st_size:
                self._sequence_bytes = np.memmap(
                    self.sequence_path, mode="r", dtype="u1"
                )
            else:
                self._sequence_bytes = np.empty(0, dtype="u1")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sequence_bytes"] = None
        state["_offsets"] = None
        return state

    def __len__(self) -> int:
        if self._selection is not None:
            return int(self._selection.size)
        return self._record_count

    def _global_index(self, index: int) -> int:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._selection is not None:
            return int(self._selection[index])
        return index

    def __getitem__(self, index: int):
        self._ensure_open()
        global_index = self._global_index(index)
        start = int(self._offsets[global_index])
        end = int(self._offsets[global_index + 1])
        if end - start != self.max_sequence_length:
            raise ValueError(
                f"DNA window {self.split}[{global_index}] has invalid byte bounds "
                f"{start}:{end}"
            )
        raw = np.asarray(self._sequence_bytes[start:end], dtype=np.uint8)
        original = torch.from_numpy(self._byte_to_token[raw].copy()).long()
        if (original == int(self.tokenizer.unk_token_id)).any():
            raise ValueError(
                f"DNA window {self.split}[{global_index}] contains a non-ACGTN byte"
            )

        labels = torch.full_like(original, self.pad_id)
        generator = None
        if self.deterministic_mlm:
            generator = torch.Generator().manual_seed(self.seed + global_index)
        eligible = torch.zeros_like(original, dtype=torch.bool)
        for token_id in self.predictable_token_ids:
            eligible |= original == token_id
        if not eligible.any():
            raise ValueError(
                f"DNA window {self.split}[{global_index}] has no canonical DNA base"
            )
        selected = (
            torch.rand(original.shape, generator=generator) < self.mlm_probability
        ) & eligible
        if not selected.any():
            selected[torch.nonzero(eligible, as_tuple=False)[0, 0]] = True
        labels[selected] = original[selected]

        corrupted = original.clone()
        replacement_draw = torch.rand(original.shape, generator=generator)
        replace_with_mask = selected & (replacement_draw < 0.8)
        corrupted[replace_with_mask] = self.mask_id
        replace_with_random = selected & (replacement_draw >= 0.8) & (
            replacement_draw < 0.9
        )
        random_indices = torch.randint(
            self.random_token_ids.numel(), original.shape, generator=generator
        )
        random_tokens = self.random_token_ids[random_indices]
        corrupted[replace_with_random] = random_tokens[replace_with_random]
        return corrupted, labels


def collate_genomic_dna_mlm(batch, pad_token_id: int):
    """Stack fixed-length DNA windows and expose the valid-token mask."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    input_ids = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    attention_mask = input_ids.ne(int(pad_token_id))
    return input_ids, labels, {"attention_mask": attention_mask}
