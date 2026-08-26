"""Dynamic annotation-aware OneMaize MLM dataset."""

from __future__ import annotations

import json
import struct
from bisect import bisect_right
from pathlib import Path
from typing import Optional

import numpy as np
import torch


REGION_CLASSES = ("gene_centered", "non_repeat", "te_rich")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


class _IndexedBgzfFasta:
    """Small `.fai` + `.gzi` BGZF reader used when pysam is unavailable."""

    def __init__(self, path: Path) -> None:
        try:
            from Bio import bgzf
        except ImportError as exc:
            raise RuntimeError(
                "BGZF FASTA access requires pysam or biopython"
            ) from exc
        self.path = path
        self.records = {}
        with Path(f"{path}.fai").open("rt", encoding="utf-8") as handle:
            for raw in handle:
                fields = raw.rstrip("\n").split("\t")
                if len(fields) < 5:
                    raise ValueError(f"Malformed FASTA index line: {raw!r}")
                self.records[fields[0]] = tuple(int(value) for value in fields[1:5])

        with Path(f"{path}.gzi").open("rb") as handle:
            count_raw = handle.read(8)
            if len(count_raw) != 8:
                raise ValueError("Malformed BGZF .gzi header")
            count = struct.unpack("<Q", count_raw)[0]
            payload = handle.read()
        if len(payload) != count * 16:
            raise ValueError("Malformed BGZF .gzi payload")
        compressed_offsets = [0]
        uncompressed_offsets = [0]
        for index in range(count):
            compressed, uncompressed = struct.unpack_from("<QQ", payload, index * 16)
            compressed_offsets.append(compressed)
            uncompressed_offsets.append(uncompressed)
        self.compressed_offsets = compressed_offsets
        self.uncompressed_offsets = uncompressed_offsets
        self.reader = bgzf.BgzfReader(str(path), "rb")

    def _virtual_offset(self, uncompressed_offset: int) -> int:
        index = bisect_right(self.uncompressed_offsets, uncompressed_offset) - 1
        within_block = uncompressed_offset - self.uncompressed_offsets[index]
        if not 0 <= within_block <= 0xFFFF:
            raise ValueError("BGZF .gzi does not map the requested offset")
        return (self.compressed_offsets[index] << 16) | within_block

    def fetch(self, seqid: str, start: int, end: int) -> str:
        if seqid not in self.records:
            raise KeyError(seqid)
        length, offset, line_bases, line_width = self.records[seqid]
        if start < 0 or end < start or end > length:
            raise ValueError(f"Invalid FASTA interval {seqid}:{start}-{end}")
        if start == end:
            return ""

        def byte_offset(position: int) -> int:
            return offset + (position // line_bases) * line_width + position % line_bases

        first = byte_offset(start)
        last = byte_offset(end - 1)
        self.reader.seek(self._virtual_offset(first))
        raw = self.reader.read(last - first + 1)
        sequence = raw.replace(b"\n", b"").replace(b"\r", b"")
        if len(sequence) != end - start:
            raise ValueError(
                f"FASTA index returned {len(sequence)} bp, expected {end - start}"
            )
        return sequence.decode("ascii").upper()

    def close(self) -> None:
        self.reader.close()


class _FastaStore:
    def __init__(
        self,
        paths: dict[str, Path],
        *,
        allow_index_build: bool = False,
    ) -> None:
        self.paths = paths
        self.allow_index_build = bool(allow_index_build)
        self._handles = {}

    def _open(self, genotype: str):
        path = self.paths[genotype]
        fai = Path(f"{path}.fai")
        if not fai.exists() and not self.allow_index_build:
            raise FileNotFoundError(
                f"Missing FASTA index {fai}. Download the matching .fai/.gzi files "
                "for BGZF FASTA or run samtools faidx before training."
            )
        try:
            if path.suffix.lower() == ".gz":
                try:
                    import pysam
                except ImportError:
                    pysam = None
                gzi = Path(f"{path}.gzi")
                if not gzi.exists():
                    raise FileNotFoundError(
                        f"Missing BGZF index {gzi}; download the matching .gzi"
                    )
                if pysam is not None:
                    handle = ("pysam", pysam.FastaFile(str(path)))
                else:
                    handle = ("bgzf", _IndexedBgzfFasta(path))
            else:
                try:
                    from pyfaidx import Fasta
                except ImportError as exc:
                    raise RuntimeError(
                        "Plain FASTA random access requires pyfaidx"
                    ) from exc
                handle = (
                    "pyfaidx",
                    Fasta(
                        str(path),
                        as_raw=True,
                        sequence_always_upper=True,
                        rebuild=self.allow_index_build,
                        build_index=self.allow_index_build,
                    ),
                )
        except Exception as exc:
            raise RuntimeError(
                f"Could not open indexed FASTA for {genotype}: {path}. Compressed "
                "inputs must be BGZF and have matching .fai and .gzi indexes."
            ) from exc
        self._handles[genotype] = handle
        return handle

    def fetch(self, genotype: str, seqid: str, start: int, end: int) -> str:
        handle = self._handles.get(genotype)
        if handle is None:
            handle = self._open(genotype)
        backend, reader = handle
        try:
            if backend == "pysam":
                return str(reader.fetch(seqid, int(start), int(end))).upper()
            if backend == "bgzf":
                return reader.fetch(seqid, int(start), int(end))
            return str(reader[seqid][int(start) : int(end)]).upper()
        except Exception as exc:
            raise RuntimeError(
                f"FASTA fetch failed for {genotype}/{seqid}:{start}-{end}"
            ) from exc

    def close(self) -> None:
        for _, reader in self._handles.values():
            close = getattr(reader, "close", None)
            if close is not None:
                close()
        self._handles = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state


class _ArrowRegionRows:
    """Mapping-like row access without materializing millions of Python dicts."""

    def __init__(self, table, columns: list[str]) -> None:
        self.table = table
        self.columns = tuple(columns)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return {
            name: self.table.column(name)[index].as_py() for name in self.columns
        }


class OneMaizeRegionMLMDataset(torch.utils.data.Dataset):
    """Hierarchically sample genotype, region class, interval, and dynamic crop."""

    SUPPORTED_SCHEMA_VERSIONS = {3}

    def __init__(
        self,
        data_dir,
        tokenizer,
        split: str,
        context_length: int = 8192,
        samples_per_epoch: int = 100_000,
        gene_probability: float = 0.5,
        non_repeat_probability: float = 0.3,
        te_rich_probability: float = 0.2,
        reverse_complement_probability: float = 0.5,
        mlm_probability: float = 0.15,
        deterministic: bool = False,
        seed: int = 2357,
        allow_index_build: bool = False,
        fasta_root: Optional[str] = None,
        max_n_fraction: Optional[float] = None,
        max_crop_attempts: int = 16,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.split = str(split).lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.context_length = int(context_length)
        self.max_sequence_length = self.context_length
        self.max_length = self.context_length
        self.samples_per_epoch = int(samples_per_epoch)
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.mlm_probability = float(mlm_probability)
        self.reverse_complement_probability = float(reverse_complement_probability)
        if not 0.0 < self.mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be in (0, 1]")
        if not 0.0 <= self.reverse_complement_probability <= 1.0:
            raise ValueError("reverse_complement_probability must be in [0, 1]")
        probabilities = np.asarray(
            [gene_probability, non_repeat_probability, te_rich_probability],
            dtype=np.float64,
        )
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("Region probabilities must be non-negative and sum to 1")
        self.region_probabilities = probabilities
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._random_rng = None

        manifest_path = self.data_dir / "manifest.json"
        with manifest_path.open("rt", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        schema_version = int(self.manifest.get("schema_version", -1))
        if schema_version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported OneMaize schema {schema_version}; expected "
                f"{sorted(self.SUPPORTED_SCHEMA_VERSIONS)}"
            )
        if self.manifest.get("coordinate_system") != "0-based-half-open":
            raise ValueError("OneMaize metadata must use zero-based half-open coordinates")
        if self.manifest.get("alphabet") != "ACGTN":
            raise ValueError("OneMaize manifest must declare alphabet ACGTN")
        allowed_contexts = {
            int(self.manifest["primary_context"]),
            int(self.manifest["extended_context"]),
        }
        if self.context_length not in allowed_contexts:
            raise ValueError(
                f"context_length must be one of {sorted(allowed_contexts)}, got "
                f"{self.context_length}"
            )
        manifest_n_fraction = float(self.manifest.get("max_n_fraction", 1.0))
        self.max_n_fraction = (
            manifest_n_fraction
            if max_n_fraction is None
            else float(max_n_fraction)
        )
        self.max_crop_attempts = int(max_crop_attempts)
        if not 0.0 <= self.max_n_fraction <= 1.0:
            raise ValueError("max_n_fraction must be in [0, 1]")
        if self.max_crop_attempts <= 0:
            raise ValueError("max_crop_attempts must be positive")

        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Reading OneMaize metadata requires pyarrow") from exc
        genomes_table = pq.read_table(self.data_dir / self.manifest["files"]["genomes"])
        genomes = genomes_table.to_pylist()
        root = None if fasta_root is None else Path(fasta_root).expanduser().resolve()
        fasta_paths = {}
        for row in genomes:
            path = Path(row["fasta"])
            if root is not None:
                path = root / path.name
            fasta_paths[row["genotype"]] = path
        self._fasta_store = _FastaStore(
            fasta_paths, allow_index_build=allow_index_build
        )

        region_columns = [
            "region_id",
            "genotype",
            "split",
            "seqid",
            "start",
            "end",
            "region_class",
            "repeat_fraction",
            "n_fraction",
            "gene_id",
        ]
        table = pq.read_table(
            self.data_dir / self.manifest["files"]["regions"],
            columns=region_columns,
            filters=[("split", "=", self.split)],
        )
        self.rows = _ArrowRegionRows(table, region_columns)
        if not self.rows:
            raise ValueError(f"OneMaize split {self.split!r} has no candidate regions")
        lengths = pc.subtract(table["end"], table["start"])
        minimum_length = int(pc.min(lengths).as_py())
        if minimum_length < self.context_length:
            raise ValueError(
                f"Shortest candidate is {minimum_length} bp, below context_length"
            )

        genotype_values = table["genotype"].combine_chunks()
        class_values = table["region_class"].combine_chunks()
        self.genotypes = sorted(pc.unique(genotype_values).to_pylist())
        genotype_codes = pc.index_in(
            genotype_values, value_set=pa.array(self.genotypes)
        ).to_numpy(zero_copy_only=False)
        class_codes = pc.index_in(
            class_values, value_set=pa.array(REGION_CLASSES)
        ).to_numpy(zero_copy_only=False)
        self.grouped_indices = {}
        for genotype_index, genotype in enumerate(self.genotypes):
            for class_index, region_class in enumerate(REGION_CLASSES):
                indices = np.flatnonzero(
                    (genotype_codes == genotype_index) & (class_codes == class_index)
                ).astype(np.int64, copy=False)
                if indices.size:
                    self.grouped_indices[(genotype, region_class)] = indices
        missing = [
            f"{genotype}/{region_class}"
            for genotype in self.genotypes
            for region_class in REGION_CLASSES
            if (genotype, region_class) not in self.grouped_indices
        ]
        if missing:
            raise ValueError(
                f"Split {self.split} lacks required genotype/class pools: "
                + ", ".join(missing)
            )

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
        self.source_counts = {
            region_class: sum(
                len(self.grouped_indices[(genotype, region_class)])
                for genotype in self.genotypes
            )
            for region_class in REGION_CLASSES
        }
        self.nucleotides = self.samples_per_epoch * self.context_length
        self.filtered_windows = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng_for_index(self, index: int) -> np.random.Generator:
        if self.deterministic:
            split_offset = {"train": 0, "val": 1, "test": 2}[self.split]
            return np.random.default_rng(self.seed + split_offset * 10_000_000 + int(index))
        if self._random_rng is None:
            worker_seed = int(torch.initial_seed())
            self._random_rng = np.random.default_rng(worker_seed + self.seed)
        return self._random_rng

    def _sample_pool(self, rng: np.random.Generator) -> tuple[str, str]:
        genotype = self.genotypes[int(rng.integers(0, len(self.genotypes)))]
        class_index = int(rng.choice(len(REGION_CLASSES), p=self.region_probabilities))
        region_class = REGION_CLASSES[class_index]
        return genotype, region_class

    def _sample_region_from_pool(
        self,
        rng: np.random.Generator,
        genotype: str,
        region_class: str,
    ) -> dict:
        pool = self.grouped_indices[(genotype, region_class)]
        row_index = int(pool[int(rng.integers(0, len(pool)))])
        return self.rows[row_index]

    def _sample_region(self, rng: np.random.Generator) -> dict:
        return self._sample_region_from_pool(rng, *self._sample_pool(rng))

    def sample_metadata(self, index: int) -> dict:
        """Return the sampled region/crop metadata without reading sequence data."""

        rng = self._rng_for_index(index)
        row = dict(self._sample_region(rng))
        max_start = int(row["end"]) - self.context_length
        crop_start = int(rng.integers(int(row["start"]), max_start + 1))
        row["crop_start"] = crop_start
        row["crop_end"] = crop_start + self.context_length
        row["reverse_complemented"] = (
            rng.random() < self.reverse_complement_probability
        )
        return row

    def _mask(self, token_ids: np.ndarray, rng: np.random.Generator):
        eligible = np.isin(token_ids, self.predictable_token_ids)
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
        random_tokens = self.random_token_ids[random_indices]
        corrupted[replace_random] = random_tokens[replace_random]
        return torch.from_numpy(corrupted), torch.from_numpy(labels)

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = self._rng_for_index(index)
        genotype, region_class = self._sample_pool(rng)
        sequence = None
        last_location = None
        for _ in range(self.max_crop_attempts):
            row = self._sample_region_from_pool(rng, genotype, region_class)
            max_start = int(row["end"]) - self.context_length
            crop_start = int(rng.integers(int(row["start"]), max_start + 1))
            crop_end = crop_start + self.context_length
            candidate = self._fasta_store.fetch(
                row["genotype"], row["seqid"], crop_start, crop_end
            )
            last_location = (
                f"{row['genotype']}/{row['seqid']}:{crop_start}-{crop_end}"
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
        if sequence is None:
            raise ValueError(
                f"Could not sample a crop with N fraction <= {self.max_n_fraction:.3f} "
                f"after {self.max_crop_attempts} attempts from "
                f"{genotype}/{region_class}; last={last_location}"
            )
        if rng.random() < self.reverse_complement_probability:
            sequence = reverse_complement(sequence)
        raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        token_ids = self._byte_to_token[raw]
        if np.any(token_ids == int(self.tokenizer.unk_token_id)):
            raise ValueError("Normalized OneMaize sample contains an unknown token")
        corrupted, labels = self._mask(token_ids, rng)
        return corrupted.long(), labels.long()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_random_rng"] = None
        return state

    def close(self) -> None:
        self._fasta_store.close()


def collate_onemaize_mlm(batch, pad_token_id: int):
    if not batch:
        raise ValueError("Cannot collate an empty OneMaize batch")
    input_ids = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    attention_mask = input_ids.ne(int(pad_token_id))
    return input_ids, labels, {"attention_mask": attention_mask}
