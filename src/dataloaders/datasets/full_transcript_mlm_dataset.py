"""Long-context MLM data for complete mRNA and non-coding RNA sequences."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def _normalize_rna(sequence: object) -> str:
    sequence = str(sequence).strip().upper().replace("T", "U")
    if (
        not sequence
        or any(base not in "AUCGN" for base in sequence)
        or not any(base in "AUCG" for base in sequence)
    ):
        return ""
    return sequence


def _stable_source_split(identifier: str, seed: int) -> str:
    """Assign an external RNA record reproducibly to an 80/10/10 split."""

    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def _iter_fasta(path: Path) -> Iterable[tuple[str, str]]:
    identifier: Optional[str] = None
    chunks: list[str] = []
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    yield identifier, "".join(chunks)
                identifier = line[1:].split(maxsplit=1)[0]
                chunks = []
            else:
                chunks.append(line)
    if identifier is not None:
        yield identifier, "".join(chunks)


class FullTranscriptMLMDataset(torch.utils.data.Dataset):
    """Return one untruncated RNA sequence with same-position MLM labels."""

    def __init__(
        self,
        mrna_jsonl,
        tokenizer,
        split: str,
        max_sequence_length: int = 10240,
        ncrna_fasta=None,
        include_ncrna: bool = True,
        require_mrna_coordinate_reliable: bool = True,
        require_cds_boundary_reliable: bool = True,
        mlm_probability: float = 0.15,
        deterministic_mlm: bool = False,
        seed: int = 2357,
        max_sequences: Optional[int] = None,
    ) -> None:
        split = str(split).lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if int(max_sequence_length) <= 0:
            raise ValueError("max_sequence_length must be positive")
        if not 0.0 < float(mlm_probability) <= 1.0:
            raise ValueError("mlm_probability must be in (0, 1]")

        self.split = split
        self.max_sequence_length = int(max_sequence_length)
        self.max_length = self.max_sequence_length
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

        self.sequences: list[str] = []
        self.sources: list[str] = []
        self.excluded_overlength = 0
        self.excluded_invalid = 0
        self.excluded_unreliable_mrna = 0
        self.excluded_unreliable_cds = 0

        self._load_mrna(
            Path(mrna_jsonl),
            require_mrna_coordinate_reliable=bool(require_mrna_coordinate_reliable),
            require_cds_boundary_reliable=bool(require_cds_boundary_reliable),
        )
        if include_ncrna:
            if ncrna_fasta is None:
                raise ValueError("include_ncrna=true requires ncrna_fasta")
            self._load_ncrna(Path(ncrna_fasta))

        if max_sequences is not None and len(self.sequences) > int(max_sequences):
            order = list(range(len(self.sequences)))
            random.Random(self.seed + {"train": 0, "val": 1, "test": 2}[split]).shuffle(order)
            order = order[: int(max_sequences)]
            self.sequences = [self.sequences[index] for index in order]
            self.sources = [self.sources[index] for index in order]

        self.source_counts = {
            source: self.sources.count(source) for source in sorted(set(self.sources))
        }
        self.nucleotides = sum(map(len, self.sequences))

    def _append(self, sequence: object, source: str) -> None:
        sequence = _normalize_rna(sequence)
        if not sequence:
            self.excluded_invalid += 1
            return
        if len(sequence) > self.max_sequence_length:
            self.excluded_overlength += 1
            return
        self.sequences.append(sequence)
        self.sources.append(source)

    def _load_mrna(
        self,
        path: Path,
        require_mrna_coordinate_reliable: bool,
        require_cds_boundary_reliable: bool,
    ) -> None:
        with _open_text(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                for field in (
                    "sequence",
                    "mrna_coordinate_system_reliable",
                    "cds_boundary_reliable",
                    "utr5_start",
                    "utr5_end",
                    "cds_start",
                    "cds_end",
                    "utr3_start",
                    "utr3_end",
                ):
                    if field not in record:
                        raise ValueError(f"{path}:{line_number} is missing {field}")
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
                    raise ValueError(
                        f"{path}:{line_number} has inconsistent full-mRNA boundaries"
                    )
                if require_mrna_coordinate_reliable and not bool(
                    record["mrna_coordinate_system_reliable"]
                ):
                    self.excluded_unreliable_mrna += 1
                    continue
                if require_cds_boundary_reliable and not bool(record["cds_boundary_reliable"]):
                    self.excluded_unreliable_cds += 1
                    continue
                self._append(record["sequence"], "mRNA")

    def _load_ncrna(self, path: Path) -> None:
        for identifier, sequence in _iter_fasta(path):
            if _stable_source_split(identifier, self.seed) == self.split:
                self._append(sequence, "ncRNA")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        sequence = self.sequences[index]
        raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        original = torch.from_numpy(self._byte_to_token[raw].copy()).long()
        labels = torch.full_like(original, self.pad_id)

        generator = None
        if self.deterministic_mlm:
            generator = torch.Generator().manual_seed(self.seed + int(index))
        eligible = torch.zeros_like(original, dtype=torch.bool)
        for token_id in self.predictable_token_ids:
            eligible |= original == token_id
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

        replace_with_random = selected & (replacement_draw >= 0.8) & (replacement_draw < 0.9)
        random_indices = torch.randint(
            self.random_token_ids.numel(), original.shape, generator=generator
        )
        random_tokens = self.random_token_ids[random_indices]
        corrupted[replace_with_random] = random_tokens[replace_with_random]
        return corrupted, labels


def collate_full_transcript_mlm(
    batch,
    pad_token_id: int,
    pad_to_multiple: int = 8,
):
    """Dynamically right-pad a variable-length MLM batch."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([item[0].numel() for item in batch], dtype=torch.long)
    padded_length = int(lengths.max().item())
    remainder = padded_length % int(pad_to_multiple)
    if remainder:
        padded_length += int(pad_to_multiple) - remainder

    input_ids = torch.full(
        (len(batch), padded_length), int(pad_token_id), dtype=torch.long
    )
    labels = torch.full_like(input_ids, int(pad_token_id))
    attention_mask = torch.zeros((len(batch), padded_length), dtype=torch.bool)
    for row, (tokens, targets) in enumerate(batch):
        length = tokens.numel()
        input_ids[row, :length] = tokens
        labels[row, :length] = targets
        attention_mask[row, :length] = True
    return input_ids, labels, {"attention_mask": attention_mask}
