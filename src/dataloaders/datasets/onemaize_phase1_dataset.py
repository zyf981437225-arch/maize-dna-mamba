"""Deterministic exhaustive B73 Phase-I full-genome MLM dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .onemaize_dataset import _FastaStore, reverse_complement


PHASE1_COLUMNS = (
    "region_id",
    "genotype",
    "chromosome",
    "start",
    "end",
    "valid_bp",
    "padded_bp",
    "is_tail",
    "window_size",
    "stride",
)


class OneMaizePhase1FullGenomeMLMDataset(torch.utils.data.Dataset):
    """One fixed index per non-overlapping 8K window, including chromosome tails."""

    def __init__(
        self,
        manifest_path,
        tokenizer,
        context_length: int = 8192,
        window_size: int = 8192,
        stride: int = 8192,
        tail_policy: str = "pad",
        genotype: str = "B73",
        reverse_complement_probability: float = 0.5,
        mlm_probability: float = 0.15,
        deterministic: bool = False,
        seed: int = 2357,
        fasta_path: Optional[str] = None,
        fasta_root: Optional[str] = None,
        allow_index_build: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyarrow is required for the Phase-I manifest") from exc
        table = pq.read_table(self.manifest_path, columns=list(PHASE1_COLUMNS) + ["fasta"])
        missing = [name for name in PHASE1_COLUMNS if name not in table.column_names]
        if missing:
            raise ValueError(f"Phase-I manifest missing columns: {missing}")
        self.rows = table.to_pylist()
        if not self.rows:
            raise ValueError("Phase-I manifest is empty")

        self.context_length = int(context_length)
        self.max_sequence_length = self.context_length
        self.max_length = self.context_length
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.context_length != self.window_size or self.stride != self.window_size:
            raise ValueError("Phase-I requires context_length == window_size == stride")
        if str(tail_policy).lower() not in {"pad", "keep", "keep-pad"}:
            raise ValueError("Phase-I tail_policy must be pad/keep")
        self.tail_policy = "pad"
        self.genotype = str(genotype)
        self.mlm = True
        self.mlm_probability = float(mlm_probability)
        self.reverse_complement_probability = float(reverse_complement_probability)
        if not 0.0 < self.mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in (0, 1]")
        if not 0.0 <= self.reverse_complement_probability <= 1.0:
            raise ValueError("reverse_complement_probability must be in [0, 1]")
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._random_rng = None
        self.tokenizer = tokenizer
        self.pad_id = int(tokenizer.pad_token_id)
        self.mask_id = int(tokenizer.mask_token_id)
        vocab = tokenizer.get_vocab()
        if any(base not in vocab for base in "ACGTN") or "U" in vocab:
            raise ValueError("OneMaize tokenizer must contain A/C/G/T/N and exclude U")
        self.random_token_ids = np.asarray([vocab[base] for base in "ACGTN"], dtype=np.int64)
        self.predictable_token_ids = np.asarray([vocab[base] for base in "ACGT"], dtype=np.int64)
        self._byte_to_token = np.full(256, int(tokenizer.unk_token_id), dtype=np.int64)
        for base in "ACGTN":
            self._byte_to_token[ord(base)] = int(vocab[base])

        self._validate_rows()
        if max_samples is None:
            self._length = len(self.rows)
        else:
            if int(max_samples) <= 0:
                raise ValueError("max_samples must be positive")
            self._length = min(len(self.rows), int(max_samples))
        if fasta_path is None:
            fasta_values = {str(row.get("fasta", "")) for row in self.rows if row.get("fasta")}
            if len(fasta_values) != 1:
                raise ValueError("Pass fasta_path when manifest does not contain one FASTA path")
            fasta_path = next(iter(fasta_values))
        fasta = Path(fasta_path).expanduser()
        if fasta_root is not None and not fasta.is_absolute():
            fasta = Path(fasta_root).expanduser() / fasta
        fasta = fasta.resolve()
        self.fasta_path = fasta
        self._fasta_store = _FastaStore({self.genotype: fasta}, allow_index_build=allow_index_build)
        self.total_valid_bp = int(sum(int(row["valid_bp"]) for row in self.rows))
        self.tail_count = int(sum(bool(row["is_tail"]) for row in self.rows))
        self.tail_valid_bp = int(sum(int(row["valid_bp"]) for row in self.rows if row["is_tail"]))
        self.nucleotides = self._length * self.context_length

    def _validate_rows(self) -> None:
        expected_id = 0
        previous_by_chrom: dict[str, tuple[int, int]] = {}
        for row in self.rows:
            if int(row["region_id"]) != expected_id:
                raise ValueError("Phase-I region_id must be contiguous and zero-based")
            expected_id += 1
            if str(row["genotype"]) != self.genotype:
                raise ValueError(f"Unexpected genotype in manifest: {row['genotype']}")
            start, end = int(row["start"]), int(row["end"])
            valid, padded = int(row["valid_bp"]), int(row["padded_bp"])
            if end <= start or end - start != valid:
                raise ValueError(f"Invalid interval row {row}")
            if valid + padded != self.window_size or not 0 <= padded < self.window_size:
                raise ValueError(f"Invalid valid/padded bp in row {row}")
            if int(row["window_size"]) != self.window_size or int(row["stride"]) != self.stride:
                raise ValueError("Manifest window/stride does not match dataset configuration")
            if bool(row["is_tail"]) != (valid < self.window_size):
                raise ValueError("is_tail disagrees with valid_bp")
            chrom = str(row["chromosome"])
            previous = previous_by_chrom.get(chrom)
            if previous is None and start != 0:
                raise ValueError(f"First window on {chrom} must start at zero")
            if previous is not None and start != previous[1]:
                raise ValueError(f"Non-contiguous or overlapping windows on {chrom}")
            previous_by_chrom[chrom] = (start, end)

    def __len__(self) -> int:
        return self._length

    def coordinate(self, index: int) -> dict:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return dict(self.rows[index])

    def _rng_for_index(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(self.seed + int(index))
        if self._random_rng is None:
            self._random_rng = np.random.default_rng(int(torch.initial_seed()) + self.seed)
        return self._random_rng

    def _mask(self, token_ids: np.ndarray, valid_mask: np.ndarray, rng: np.random.Generator):
        eligible = np.isin(token_ids, self.predictable_token_ids) & valid_mask
        selected = (rng.random(token_ids.shape[0]) < self.mlm_probability) & eligible
        if not selected.any():
            eligible_positions = np.flatnonzero(eligible)
            if eligible_positions.size == 0:
                raise ValueError("Sample contains no canonical A/C/G/T position")
            selected[eligible_positions[0]] = True
        labels = np.full(token_ids.shape, self.pad_id, dtype=np.int64)
        labels[selected] = token_ids[selected]
        corrupted = token_ids.copy()
        draw = rng.random(token_ids.shape[0])
        replace_mask = selected & (draw < 0.8)
        replace_random = selected & (draw >= 0.8) & (draw < 0.9)
        corrupted[replace_mask] = self.mask_id
        random_indices = rng.integers(0, len(self.random_token_ids), size=token_ids.shape[0])
        corrupted[replace_random] = self.random_token_ids[random_indices[replace_random]]
        return torch.from_numpy(corrupted), torch.from_numpy(labels)

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        for _ in range(len(self)):
            row = self.rows[index]
            valid_bp = int(row["valid_bp"])
            padded_bp = int(row["padded_bp"])

            sequence = self._fasta_store.fetch(
                self.genotype,
                str(row["chromosome"]),
                int(row["start"]),
                int(row["end"]),
            )

            if len(sequence) != valid_bp:
                raise ValueError(
                    f"Expected {valid_bp} bp, fetched {len(sequence)} for region {index}"
                )

            sequence = sequence.upper()

            if set(sequence) - set("ACGTN"):
                raise ValueError(f"Non-ACGTN sequence fetched for region {index}")

            if any(base in sequence for base in "ACGT"):
                break

            index = (index + 1) % len(self)
        else:
            raise ValueError("No trainable A/C/G/T-containing Phase-I window found")

        rng = self._rng_for_index(index)

        reverse_complemented = bool(
            rng.random() < self.reverse_complement_probability
        )
        if reverse_complemented:
            sequence = reverse_complement(sequence)

        raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        real_tokens = self._byte_to_token[raw]

        if np.any(real_tokens == int(self.tokenizer.unk_token_id)):
            raise ValueError("Normalized Phase-I sample contains an unknown token")

        token_ids = np.pad(
            real_tokens,
            (0, padded_bp),
            constant_values=self.pad_id,
        )

        valid_mask = np.arange(self.context_length) < valid_bp

        corrupted, labels = self._mask(
            token_ids,
            valid_mask,
            rng,
        )

        metadata = {
            "attention_mask": torch.from_numpy(valid_mask.copy()),
            "valid_mask": torch.from_numpy(valid_mask.copy()),
            "phase1_region_id": torch.tensor(
                int(row["region_id"]), dtype=torch.long
            ),
            "phase1_valid_bp": torch.tensor(valid_bp, dtype=torch.long),
            "phase1_is_tail": torch.tensor(
                bool(row["is_tail"]), dtype=torch.bool
            ),
            "phase1_reverse_complemented": torch.tensor(
                reverse_complemented, dtype=torch.bool
            ),
        }

        return corrupted.long(), labels.long(), metadata

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_random_rng"] = None
        return state

    def close(self) -> None:
        self._fasta_store.close()


def collate_onemaize_phase1_mlm(batch, pad_token_id: int):
    if not batch:
        raise ValueError("Cannot collate an empty Phase-I batch")
    input_ids = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    keys = batch[0][2].keys()
    metadata = {key: torch.stack([item[2][key] for item in batch], dim=0) for key in keys}
    metadata["attention_mask"] = input_ids.ne(int(pad_token_id))
    return input_ids, labels, metadata
