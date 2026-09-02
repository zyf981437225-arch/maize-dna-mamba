"""Data module for explicit variant/TE-aware OneMaize Phase-II MLM."""

from __future__ import annotations

from typing import Any, Optional

from hydra.utils import to_absolute_path

from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.dataloaders.base import SequenceDataset
from src.dataloaders.datasets.onemaize_dataset import collate_onemaize_mlm
from src.dataloaders.datasets.onemaize_variant_dataset import (
    OneMaizeVariantTEMLMDataset,
)


class OneMaizeVariantTEDNAMLM(SequenceDataset):
    """Phase-II data module; Model B still uses the unchanged MLM model."""

    _name_ = "onemaize_variant_te_mlm"
    _collate_arg_names = ["attention_mask"]

    def __init__(
        self,
        _name_: str,
        data_dir,
        variant_data_dir,
        context_length: int = 16384,
        mlm_probability: float = 0.15,
        reverse_complement_probability: float = 0.5,
        gene_probability: float = 0.20,
        non_repeat_probability: float = 0.15,
        te_rich_probability: float = 0.15,
        small_variant_probability: float = 0.20,
        structural_variant_probability: float = 0.20,
        te_variant_probability: float = 0.10,
        missing_class_policy: str = "error",
        small_variant_max_length: int = 50,
        variant_jitter: int = 4096,
        train_samples_per_epoch: int = 100_000,
        val_samples_per_epoch: int = 2_048,
        test_samples_per_epoch: int = 2_048,
        batch_size: int = 1,
        batch_size_eval: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 2357,
        allow_index_build: bool = False,
        fasta_root: Optional[str] = None,
        max_n_fraction: Optional[float] = None,
        max_crop_attempts: int = 16,
        sequence_type: str = "dna",
        tokenizer_name: str = "char",
        species: str = "Zea mays",
        **kwargs,
    ) -> None:
        super().__init__(_name_=_name_, data_dir=to_absolute_path(str(data_dir)))
        if sequence_type.lower() != "dna" or tokenizer_name != "char":
            raise ValueError("Variant-aware OneMaize requires char-tokenized DNA")
        self.variant_data_dir = to_absolute_path(str(variant_data_dir))
        self.context_length = int(context_length)
        self.max_sequence_length = self.context_length
        self.max_length = self.context_length
        self.mlm = True
        self.mlm_probability = float(mlm_probability)
        self.reverse_complement_probability = float(reverse_complement_probability)
        self.probabilities = {
            "gene_probability": float(gene_probability),
            "non_repeat_probability": float(non_repeat_probability),
            "te_rich_probability": float(te_rich_probability),
            "small_variant_probability": float(small_variant_probability),
            "structural_variant_probability": float(structural_variant_probability),
            "te_variant_probability": float(te_variant_probability),
        }
        self.missing_class_policy = missing_class_policy
        self.small_variant_max_length = int(small_variant_max_length)
        self.variant_jitter = int(variant_jitter)
        self.train_samples_per_epoch = int(train_samples_per_epoch)
        self.val_samples_per_epoch = int(val_samples_per_epoch)
        self.test_samples_per_epoch = int(test_samples_per_epoch)
        self.batch_size = int(batch_size)
        self.batch_size_eval = int(batch_size_eval or batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.allow_index_build = bool(allow_index_build)
        self.fasta_root = fasta_root
        self.max_n_fraction = max_n_fraction
        self.max_crop_attempts = int(max_crop_attempts)
        self.species = species
        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None) -> None:
        self.tokenizer = CaduceusTokenizer(
            model_max_length=self.context_length,
            sequence_type="dna",
            add_special_tokens=False,
            padding_side="right",
        )
        self.vocab_size = len(self.tokenizer)

        def make(split: str, samples: int, deterministic: bool, rc_probability: float):
            return OneMaizeVariantTEMLMDataset(
                self.data_dir,
                self.variant_data_dir,
                tokenizer=self.tokenizer,
                split=split,
                context_length=self.context_length,
                samples_per_epoch=samples,
                reverse_complement_probability=rc_probability,
                mlm_probability=self.mlm_probability,
                deterministic=deterministic,
                seed=self.seed,
                allow_index_build=self.allow_index_build,
                fasta_root=self.fasta_root,
                max_n_fraction=self.max_n_fraction,
                max_crop_attempts=self.max_crop_attempts,
                missing_class_policy=self.missing_class_policy,
                small_variant_max_length=self.small_variant_max_length,
                variant_jitter=self.variant_jitter,
                **self.probabilities,
            )

        self.dataset_train = make(
            "train", self.train_samples_per_epoch, False, self.reverse_complement_probability
        )
        self.dataset_val = make("val", self.val_samples_per_epoch, True, 0.0)
        self.dataset_test = make("test", self.test_samples_per_epoch, True, 0.0)

    def _collate_fn(self, batch, *args, **kwargs):
        return collate_onemaize_mlm(
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
        kwargs.update(drop_last=False, shuffle=False)
        return self._dataloader(
            self.dataset_val, batch_size=self.batch_size_eval, **kwargs
        )

    def test_dataloader(self, **kwargs: Any):
        kwargs.update(drop_last=False, shuffle=False)
        return self._dataloader(
            self.dataset_test, batch_size=self.batch_size_eval, **kwargs
        )
