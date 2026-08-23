"""Data module for human m6A sliding-window fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hydra.utils import to_absolute_path

from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.dataloaders.base import SequenceDataset
from src.dataloaders.datasets.human_m6a_window_dataset import HumanM6AWindowDataset
from src.dataloaders.datasets.human_m6a_nucleotide_dataset import (
    HumanM6AFullTranscriptDataset,
    HumanM6ANucleotideDataset,
    collate_full_transcripts,
)


class HumanM6A(SequenceDataset):
    """Transcript-grouped train/validation/test m6A window data module."""

    _name_ = "human_m6a"
    _collate_arg_names = ["lengths"]
    l_output = 0
    d_output = 1

    def __init__(
        self,
        _name_: str,
        data_dir,
        tokenizer_name: str = "char",
        window_length: int = 128,
        stride: int = 64,
        target_transform: str = "none",
        batch_size: int = 8,
        batch_size_eval: Optional[int] = None,
        shuffle: bool = True,
        max_train_windows: Optional[int] = None,
        max_val_windows: Optional[int] = None,
        max_test_windows: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(_name_=_name_, data_dir=to_absolute_path(str(data_dir)))
        self.tokenizer_name = tokenizer_name
        self.window_length = int(window_length)
        self.max_length = self.window_length
        self.stride = int(stride)
        self.target_transform = target_transform
        self.batch_size = int(batch_size)
        self.batch_size_eval = int(batch_size_eval or batch_size)
        self.shuffle = bool(shuffle)
        self.max_train_windows = max_train_windows
        self.max_val_windows = max_val_windows
        self.max_test_windows = max_test_windows
        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None):
        if self.tokenizer_name != "char":
            raise NotImplementedError("Human m6A fine-tuning currently supports only the RNA character tokenizer")
        self.tokenizer = CaduceusTokenizer(
            model_max_length=self.window_length,
            add_special_tokens=False,
            padding_side="right",
        )
        self.vocab_size = len(self.tokenizer)

        common = {
            "tokenizer": self.tokenizer,
            "window_length": self.window_length,
            "stride": self.stride,
            "target_transform": self.target_transform,
        }
        data_dir = Path(self.data_dir)
        self.dataset_train = HumanM6AWindowDataset(
            data_dir / "train.jsonl", max_windows=self.max_train_windows, **common
        )
        self.dataset_val = HumanM6AWindowDataset(
            data_dir / "val.jsonl", max_windows=self.max_val_windows, **common
        )
        self.dataset_test = HumanM6AWindowDataset(
            data_dir / "test.jsonl", max_windows=self.max_test_windows, **common
        )

    def train_dataloader(self, **kwargs: Any):
        kwargs.setdefault("drop_last", False)
        return self._dataloader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs,
        )

    def val_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(
            self.dataset_val,
            batch_size=self.batch_size_eval,
            **kwargs,
        )

    def test_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(
            self.dataset_test,
            batch_size=self.batch_size_eval,
            **kwargs,
        )


class HumanM6ANucleotide(SequenceDataset):
    """Gene-split full-mRNA windows with per-adenosine binary labels."""

    _name_ = "human_m6a_nt"
    _collate_arg_names = []
    l_output = None
    d_output = 1

    def __init__(
        self,
        _name_: str,
        data_dir,
        tokenizer_name: str = "char",
        window_length: int = 1024,
        stride: int = 512,
        require_mrna_coordinate_reliable: bool = True,
        batch_size: int = 8,
        batch_size_eval: Optional[int] = None,
        shuffle: bool = True,
        max_train_transcripts: Optional[int] = None,
        max_val_transcripts: Optional[int] = None,
        max_test_transcripts: Optional[int] = None,
        max_train_windows: Optional[int] = None,
        max_val_windows: Optional[int] = None,
        max_test_windows: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(_name_=_name_, data_dir=to_absolute_path(str(data_dir)))
        self.tokenizer_name = tokenizer_name
        self.window_length = int(window_length)
        self.max_length = self.window_length
        self.stride = int(stride)
        self.require_mrna_coordinate_reliable = bool(require_mrna_coordinate_reliable)
        self.batch_size = int(batch_size)
        self.batch_size_eval = int(batch_size_eval or batch_size)
        self.shuffle = bool(shuffle)
        self.max_train_transcripts = max_train_transcripts
        self.max_val_transcripts = max_val_transcripts
        self.max_test_transcripts = max_test_transcripts
        self.max_train_windows = max_train_windows
        self.max_val_windows = max_val_windows
        self.max_test_windows = max_test_windows
        self.tokenizer = None
        self.vocab_size = 0
        self.positive_weight = 1.0

    def _split_path(self, split: str) -> Path:
        compressed = Path(self.data_dir) / f"{split}.jsonl.gz"
        if compressed.exists():
            return compressed
        plain = Path(self.data_dir) / f"{split}.jsonl"
        if plain.exists():
            return plain
        raise FileNotFoundError(f"Missing {compressed} (or uncompressed {plain})")

    def setup(self, stage=None):
        if self.tokenizer_name != "char":
            raise NotImplementedError("Nucleotide-level m6A training supports only the RNA character tokenizer")
        self.tokenizer = CaduceusTokenizer(
            model_max_length=self.window_length,
            add_special_tokens=False,
            padding_side="right",
        )
        self.vocab_size = len(self.tokenizer)

        common = {
            "tokenizer": self.tokenizer,
            "window_length": self.window_length,
            "stride": self.stride,
            "require_mrna_coordinate_reliable": self.require_mrna_coordinate_reliable,
        }
        self.dataset_train = HumanM6ANucleotideDataset(
            self._split_path("train"),
            max_transcripts=self.max_train_transcripts,
            max_windows=self.max_train_windows,
            **common,
        )
        self.dataset_val = HumanM6ANucleotideDataset(
            self._split_path("val"),
            max_transcripts=self.max_val_transcripts,
            max_windows=self.max_val_windows,
            **common,
        )
        self.dataset_test = HumanM6ANucleotideDataset(
            self._split_path("test"),
            max_transcripts=self.max_test_transcripts,
            max_windows=self.max_test_windows,
            **common,
        )
        self.positive_weight = self.dataset_train.positive_weight
        print(
            "[HumanM6ANucleotide] "
            f"train_windows={len(self.dataset_train)} "
            f"candidate_A={self.dataset_train.candidate_count} "
            f"positive={self.dataset_train.positive_count} "
            f"pos_weight={self.positive_weight:.4f}"
        )

    def train_dataloader(self, **kwargs: Any):
        kwargs.setdefault("drop_last", False)
        return self._dataloader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs,
        )

    def val_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(self.dataset_val, batch_size=self.batch_size_eval, **kwargs)

    def test_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(self.dataset_test, batch_size=self.batch_size_eval, **kwargs)


class HumanM6AFullTranscript(SequenceDataset):
    """One complete 5'UTR+CDS+3'UTR sequence per training item."""

    _name_ = "human_m6a_full"
    _collate_arg_names = ["attention_mask", "lengths"]
    l_output = None
    d_output = 1

    def __init__(
        self,
        _name_: str,
        data_dir,
        tokenizer_name: str = "char",
        max_sequence_length: int = 10240,
        overlength_policy: str = "exclude",
        require_mrna_coordinate_reliable: bool = True,
        require_cds_boundary_reliable: bool = True,
        pad_to_multiple: int = 8,
        batch_size: int = 1,
        batch_size_eval: Optional[int] = None,
        shuffle: bool = True,
        max_train_transcripts: Optional[int] = None,
        max_val_transcripts: Optional[int] = None,
        max_test_transcripts: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(_name_=_name_, data_dir=to_absolute_path(str(data_dir)))
        self.tokenizer_name = tokenizer_name
        self.max_sequence_length = int(max_sequence_length)
        self.max_length = self.max_sequence_length
        self.overlength_policy = str(overlength_policy)
        self.require_mrna_coordinate_reliable = bool(require_mrna_coordinate_reliable)
        self.require_cds_boundary_reliable = bool(require_cds_boundary_reliable)
        self.pad_to_multiple = int(pad_to_multiple)
        self.batch_size = int(batch_size)
        self.batch_size_eval = int(batch_size_eval or batch_size)
        self.shuffle = bool(shuffle)
        self.max_train_transcripts = max_train_transcripts
        self.max_val_transcripts = max_val_transcripts
        self.max_test_transcripts = max_test_transcripts
        self.tokenizer = None
        self.vocab_size = 0
        self.positive_weight = 1.0
        if self.pad_to_multiple <= 0:
            raise ValueError("pad_to_multiple must be positive")

    def _split_path(self, split: str) -> Path:
        compressed = Path(self.data_dir) / f"{split}.jsonl.gz"
        if compressed.exists():
            return compressed
        plain = Path(self.data_dir) / f"{split}.jsonl"
        if plain.exists():
            return plain
        raise FileNotFoundError(f"Missing {compressed} (or uncompressed {plain})")

    def setup(self, stage=None):
        if self.tokenizer_name != "char":
            raise NotImplementedError("Full-mRNA m6A training supports only the RNA character tokenizer")
        self.tokenizer = CaduceusTokenizer(
            model_max_length=self.max_sequence_length,
            add_special_tokens=False,
            padding_side="right",
        )
        self.vocab_size = len(self.tokenizer)
        common = {
            "tokenizer": self.tokenizer,
            "max_sequence_length": self.max_sequence_length,
            "overlength_policy": self.overlength_policy,
            "require_mrna_coordinate_reliable": self.require_mrna_coordinate_reliable,
            "require_cds_boundary_reliable": self.require_cds_boundary_reliable,
        }
        self.dataset_train = HumanM6AFullTranscriptDataset(
            self._split_path("train"),
            max_transcripts=self.max_train_transcripts,
            **common,
        )
        self.dataset_val = HumanM6AFullTranscriptDataset(
            self._split_path("val"),
            max_transcripts=self.max_val_transcripts,
            **common,
        )
        self.dataset_test = HumanM6AFullTranscriptDataset(
            self._split_path("test"),
            max_transcripts=self.max_test_transcripts,
            **common,
        )
        self.positive_weight = self.dataset_train.positive_weight
        for split, dataset in (
            ("train", self.dataset_train),
            ("val", self.dataset_val),
            ("test", self.dataset_test),
        ):
            print(
                "[HumanM6AFullTranscript] "
                f"split={split} transcripts={len(dataset)} "
                f"candidate_A={dataset.candidate_count} positive={dataset.positive_count} "
                f"excluded_overlength={dataset.excluded_overlength} "
                f"excluded_unreliable_mrna={dataset.excluded_unreliable_mrna} "
                f"excluded_unreliable_cds={dataset.excluded_unreliable_cds}"
            )
        print(f"[HumanM6AFullTranscript] train_pos_weight={self.positive_weight:.4f}")

    def _collate_fn(self, batch, *args, **kwargs):
        """Dynamically pad only to the longest transcript in this batch."""
        return collate_full_transcripts(
            batch,
            pad_token_id=int(self.tokenizer.pad_token_id),
            pad_to_multiple=self.pad_to_multiple,
        )

    def train_dataloader(self, **kwargs: Any):
        kwargs.setdefault("drop_last", False)
        return self._dataloader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            **kwargs,
        )

    def val_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(self.dataset_val, batch_size=self.batch_size_eval, **kwargs)

    def test_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(self.dataset_test, batch_size=self.batch_size_eval, **kwargs)
