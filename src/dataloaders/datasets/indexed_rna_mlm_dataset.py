"""Memory-mapped RNA corpus for million-scale same-position MLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class IndexedRNAMLMDataset(torch.utils.data.Dataset):
    """Read complete RNA sequences from an indexed, memory-mapped corpus.

    The sequence bytes stay on disk and are mapped lazily in each DataLoader
    worker.  This avoids materialising millions of Python strings in every DDP
    process while retaining random access for the distributed sampler.
    """

    SUPPORTED_SCHEMA_VERSIONS = {1, 2}

    def __init__(
        self,
        data_dir,
        tokenizer,
        split: str,
        max_sequence_length: Optional[int] = None,
        mlm_probability: float = 0.15,
        deterministic_mlm: bool = False,
        seed: int = 2357,
        max_sequences: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = str(split).lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not 0.0 < float(mlm_probability) <= 1.0:
            raise ValueError("mlm_probability must be in (0, 1]")

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing indexed-corpus manifest: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        schema_version = int(manifest.get("schema_version", -1))
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported indexed-corpus schema {manifest.get('schema_version')}; "
                f"expected one of {sorted(self.SUPPORTED_SCHEMA_VERSIONS)}"
            )
        if self.split not in manifest.get("splits", {}):
            raise ValueError(f"Manifest does not define split {self.split!r}")

        split_info = manifest["splits"][self.split]
        files = split_info["files"]
        self.sequence_path = self.data_dir / files["sequences"]
        self.offset_path = self.data_dir / files["offsets"]
        self.source_path = self.data_dir / files["sources"]
        for path in (self.sequence_path, self.offset_path, self.source_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing indexed-corpus file: {path}")

        self._record_count = int(split_info["records"])
        expected_offset_bytes = (self._record_count + 1) * np.dtype("<u8").itemsize
        if self.offset_path.stat().st_size != expected_offset_bytes:
            raise ValueError(
                f"Invalid offset index size for {self.split}: "
                f"expected {expected_offset_bytes}, got {self.offset_path.stat().st_size}"
            )
        if self.source_path.stat().st_size != self._record_count:
            raise ValueError(
                f"Invalid source index size for {self.split}: "
                f"expected {self._record_count}, got {self.source_path.stat().st_size}"
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

        self.tokenizer = tokenizer
        self.mlm_probability = float(mlm_probability)
        self.deterministic_mlm = bool(deterministic_mlm)
        self.seed = int(seed)
        self.pad_id = int(tokenizer.pad_token_id)
        self.mask_id = int(tokenizer.mask_token_id)
        vocab = tokenizer.get_vocab()
        self.random_token_ids = torch.tensor(
            [vocab[base] for base in "ACGUN" if base in vocab], dtype=torch.long
        )
        self.predictable_token_ids = torch.tensor(
            [vocab[base] for base in "ACGU" if base in vocab], dtype=torch.long
        )
        if self.random_token_ids.numel() == 0:
            raise ValueError("Tokenizer has no RNA nucleotide tokens")

        self._byte_to_token = np.full(256, int(tokenizer.unk_token_id), dtype=np.int64)
        for base in "ACGUN":
            if base in vocab:
                self._byte_to_token[ord(base)] = int(vocab[base])

        self._sequence_bytes = None
        self._offsets = None
        self._sources = None
        self._selection = None
        if max_sequences is not None and int(max_sequences) < self._record_count:
            limit = int(max_sequences)
            if limit < 0:
                raise ValueError("max_sequences cannot be negative")
            split_offset = {"train": 0, "val": 1, "test": 2}[self.split]
            rng = np.random.default_rng(self.seed + split_offset)
            self._selection = rng.choice(
                self._record_count, size=limit, replace=False
            ).astype(np.int64)

        self.source_counts = dict(split_info.get("source_class_counts", {}))
        self.source_type_counts = dict(split_info.get("source_type_counts", {}))
        self.nucleotides = int(split_info.get("nucleotides", 0))
        self.max_sequence_length = int(manifest["selection"]["max_length"])
        if (
            max_sequence_length is not None
            and self.max_sequence_length > int(max_sequence_length)
        ):
            raise ValueError(
                f"Prepared corpus permits {self.max_sequence_length}-nt records, "
                f"but the dataset config permits only {int(max_sequence_length)} nt"
            )
        self.max_length = self.max_sequence_length

    def _ensure_open(self) -> None:
        if self._offsets is None:
            self._offsets = np.memmap(self.offset_path, mode="r", dtype="<u8")
        if self._sources is None:
            self._sources = np.memmap(self.source_path, mode="r", dtype="u1")
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
        state["_sources"] = None
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
        if not 0 <= start < end <= len(self._sequence_bytes):
            raise ValueError(
                f"Invalid indexed sequence bounds for {self.split}[{global_index}]: "
                f"{start}:{end}"
            )
        raw = np.asarray(self._sequence_bytes[start:end], dtype=np.uint8)
        original = torch.from_numpy(self._byte_to_token[raw].copy()).long()

        labels = torch.full_like(original, self.pad_id)
        generator = None
        if self.deterministic_mlm:
            generator = torch.Generator().manual_seed(self.seed + global_index)
        eligible = torch.zeros_like(original, dtype=torch.bool)
        for token_id in self.predictable_token_ids:
            eligible |= original == token_id
        if not eligible.any():
            raise ValueError(
                f"Indexed record {self.split}[{global_index}] has no canonical RNA base"
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
