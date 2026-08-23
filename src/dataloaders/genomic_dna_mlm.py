"""Data module for fixed-window maize genomic-DNA MLM."""

from __future__ import annotations

from typing import Any, Optional

from hydra.utils import to_absolute_path

from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.dataloaders.base import SequenceDataset
from src.dataloaders.datasets.genomic_dna_dataset import (
    IndexedGenomicDNAMLMDataset,
    collate_genomic_dna_mlm,
)


class GenomicDNAMLM(SequenceDataset):
    """Leakage-safe genomic windows prepared at contig boundaries."""

    _name_ = "genomic_dna_mlm"
    _collate_arg_names = ["attention_mask"]

    def __init__(
        self,
        _name_: str,
        data_dir,
        sequence_type: str = "dna",
        species: str = "maize",
        tokenizer_name: str = "char",
        window_size: int = 10240,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        batch_size: int = 1,
        batch_size_eval: Optional[int] = None,
        shuffle: bool = True,
        split_seed: int = 2357,
        require_nonempty_splits: bool = True,
        max_train_windows: Optional[int] = None,
        max_val_windows: Optional[int] = None,
        max_test_windows: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(_name_=_name_, data_dir=to_absolute_path(str(data_dir)))
        if str(sequence_type).lower() != "dna":
            raise ValueError("GenomicDNAMLM requires sequence_type=dna")
        if not mlm:
            raise ValueError("GenomicDNAMLM requires mlm=true")
        self.sequence_type = "dna"
        self.species = str(species)
        self.tokenizer_name = str(tokenizer_name)
        self.window_size = int(window_size)
        self.max_sequence_length = self.window_size
        self.max_length = self.window_size
        self.mlm = True
        self.mlm_probability = float(mlm_probability)
        self.batch_size = int(batch_size)
        self.batch_size_eval = int(batch_size_eval or batch_size)
        self.shuffle = bool(shuffle)
        self.split_seed = int(split_seed)
        self.require_nonempty_splits = bool(require_nonempty_splits)
        self.max_train_windows = max_train_windows
        self.max_val_windows = max_val_windows
        self.max_test_windows = max_test_windows
        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None) -> None:
        if self.tokenizer_name != "char":
            raise NotImplementedError("Maize DNA MLM supports only char tokenization")
        self.tokenizer = CaduceusTokenizer(
            model_max_length=self.window_size,
            sequence_type="dna",
            add_special_tokens=False,
            padding_side="right",
        )
        self.vocab_size = len(self.tokenizer)

        def make_dataset(split, deterministic_mlm, max_windows):
            return IndexedGenomicDNAMLMDataset(
                self.data_dir,
                tokenizer=self.tokenizer,
                split=split,
                window_size=self.window_size,
                mlm_probability=self.mlm_probability,
                deterministic_mlm=deterministic_mlm,
                seed=self.split_seed,
                max_windows=max_windows,
            )

        self.dataset_train = make_dataset(
            "train", deterministic_mlm=False, max_windows=self.max_train_windows
        )
        self.dataset_val = make_dataset(
            "val", deterministic_mlm=True, max_windows=self.max_val_windows
        )
        self.dataset_test = make_dataset(
            "test", deterministic_mlm=True, max_windows=self.max_test_windows
        )
        datasets = {
            "train": self.dataset_train,
            "val": self.dataset_val,
            "test": self.dataset_test,
        }
        if self.require_nonempty_splits:
            empty = [split for split, dataset in datasets.items() if len(dataset) == 0]
            if empty:
                raise ValueError(
                    "Formal DNA training requires non-empty chromosome/contig splits; "
                    f"empty splits: {', '.join(empty)}"
                )
        for split, dataset in datasets.items():
            print(
                "[GenomicDNAMLM] "
                f"split={split} windows={len(dataset)} "
                f"window_size={dataset.max_sequence_length} "
                f"nucleotides={dataset.nucleotides} "
                f"filtered_windows={dataset.filtered_windows}"
            )

    def _collate_fn(self, batch, *args, **kwargs):
        return collate_genomic_dna_mlm(
            batch, pad_token_id=int(self.tokenizer.pad_token_id)
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
            self.dataset_val, batch_size=self.batch_size_eval, **kwargs
        )

    def test_dataloader(self, **kwargs: Any):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._dataloader(
            self.dataset_test, batch_size=self.batch_size_eval, **kwargs
        )
