"""Data module for OneMaize population-aware dynamic DNA MLM."""

from __future__ import annotations

from typing import Any, Optional

from hydra.utils import to_absolute_path

from caduceus.tokenization_caduceus import CaduceusTokenizer
from src.dataloaders.base import SequenceDataset
from src.dataloaders.datasets.onemaize_dataset import (
    OneMaizeRegionMLMDataset,
    collate_onemaize_mlm,
)


class OneMaizeDNAMLM(SequenceDataset):
    """26-genotype hierarchical sampling with B73 pilot compatibility."""

    _name_ = "onemaize_dna_mlm"
    _collate_arg_names = ["attention_mask"]

    def __init__(
        self,
        _name_: str,
        data_dir,
        sequence_type: str = "dna",
        species: str = "Zea mays",
        tokenizer_name: str = "char",
        context_length: int = 8192,
        mlm_probability: float = 0.15,
        reverse_complement_probability: float = 0.5,
        gene_probability: float = 0.5,
        non_repeat_probability: float = 0.3,
        te_rich_probability: float = 0.2,
        train_samples_per_epoch: int = 100_000,
        val_samples_per_epoch: int = 2_048,
        test_samples_per_epoch: int = 2_048,
        batch_size: int = 1,
        batch_size_eval: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 2357,
        allow_index_build: bool = False,
        fasta_root: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(_name_=_name_, data_dir=to_absolute_path(str(data_dir)))
        if str(sequence_type).lower() != "dna":
            raise ValueError("OneMaizeDNAMLM requires sequence_type=dna")
        if tokenizer_name != "char":
            raise NotImplementedError("OneMaize v1 uses single-nucleotide tokenization")
        self.sequence_type = "dna"
        self.species = str(species)
        self.tokenizer_name = tokenizer_name
        self.context_length = int(context_length)
        self.max_sequence_length = self.context_length
        self.max_length = self.context_length
        self.mlm = True
        self.mlm_probability = float(mlm_probability)
        self.reverse_complement_probability = float(reverse_complement_probability)
        self.gene_probability = float(gene_probability)
        self.non_repeat_probability = float(non_repeat_probability)
        self.te_rich_probability = float(te_rich_probability)
        self.train_samples_per_epoch = int(train_samples_per_epoch)
        self.val_samples_per_epoch = int(val_samples_per_epoch)
        self.test_samples_per_epoch = int(test_samples_per_epoch)
        self.batch_size = int(batch_size)
        self.batch_size_eval = int(batch_size_eval or batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.allow_index_build = bool(allow_index_build)
        self.fasta_root = fasta_root
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

        def make_dataset(split: str, samples: int, deterministic: bool, rc_p: float):
            return OneMaizeRegionMLMDataset(
                self.data_dir,
                tokenizer=self.tokenizer,
                split=split,
                context_length=self.context_length,
                samples_per_epoch=samples,
                gene_probability=self.gene_probability,
                non_repeat_probability=self.non_repeat_probability,
                te_rich_probability=self.te_rich_probability,
                reverse_complement_probability=rc_p,
                mlm_probability=self.mlm_probability,
                deterministic=deterministic,
                seed=self.seed,
                allow_index_build=self.allow_index_build,
                fasta_root=self.fasta_root,
            )

        self.dataset_train = make_dataset(
            "train",
            self.train_samples_per_epoch,
            deterministic=False,
            rc_p=self.reverse_complement_probability,
        )
        self.dataset_val = make_dataset(
            "val", self.val_samples_per_epoch, deterministic=True, rc_p=0.0
        )
        self.dataset_test = make_dataset(
            "test", self.test_samples_per_epoch, deterministic=True, rc_p=0.0
        )
        for split, dataset in (
            ("train", self.dataset_train),
            ("val", self.dataset_val),
            ("test", self.dataset_test),
        ):
            print(
                "[OneMaizeDNAMLM] "
                f"split={split} genotypes={len(dataset.genotypes)} "
                f"candidate_regions={len(dataset.rows)} samples={len(dataset)} "
                f"context={self.context_length} pools={dataset.source_counts}"
            )

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
