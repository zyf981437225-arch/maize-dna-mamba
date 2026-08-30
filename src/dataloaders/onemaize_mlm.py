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
from src.dataloaders.datasets.onemaize_phase1_dataset import (
    OneMaizePhase1FullGenomeMLMDataset,
    collate_onemaize_phase1_mlm,
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
        mode: str = "region_aware",
        full_genome_manifest: Optional[str] = None,
        fasta_path: Optional[str] = None,
        window_size: int = 8192,
        stride: int = 8192,
        tail_policy: str = "pad",
        genotype: str = "B73",
        mlm_probability: float = 0.15,
        reverse_complement_probability: float = 0.5,
        gene_probability: float = 0.5,
        non_repeat_probability: float = 0.3,
        te_rich_probability: float = 0.2,
        train_samples_per_epoch: Optional[int] = 100_000,
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
        self.mode = str(mode).lower()
        if self.mode not in {"region_aware", "full_genome"}:
            raise ValueError("mode must be region_aware or full_genome")
        self.full_genome_manifest = full_genome_manifest
        self.fasta_path = fasta_path
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.tail_policy = str(tail_policy)
        self.genotype = str(genotype)
        self.max_sequence_length = self.context_length
        self.max_length = self.context_length
        self.mlm = True
        self.mlm_probability = float(mlm_probability)
        self.reverse_complement_probability = float(reverse_complement_probability)
        self.gene_probability = float(gene_probability)
        self.non_repeat_probability = float(non_repeat_probability)
        self.te_rich_probability = float(te_rich_probability)
        self.train_samples_per_epoch = (
            None if train_samples_per_epoch is None else int(train_samples_per_epoch)
        )
        if self.mode == "region_aware" and self.train_samples_per_epoch is None:
            raise ValueError("region_aware mode requires train_samples_per_epoch")
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

        if self.mode == "full_genome":
            if self.full_genome_manifest is None:
                raise ValueError("full_genome mode requires full_genome_manifest")

            def make_phase1(split: str, max_samples: Optional[int], rc_p: float):
                return OneMaizePhase1FullGenomeMLMDataset(
                    self.full_genome_manifest,
                    tokenizer=self.tokenizer,
                    context_length=self.context_length,
                    window_size=self.window_size,
                    stride=self.stride,
                    tail_policy=self.tail_policy,
                    genotype=self.genotype,
                    reverse_complement_probability=rc_p,
                    mlm_probability=self.mlm_probability,
                    deterministic=(split != "train"),
                    seed=self.seed + {"train": 0, "val": 1, "test": 2}[split] * 10_000_000,
                    fasta_path=self.fasta_path,
                    fasta_root=self.fasta_root,
                    allow_index_build=self.allow_index_build,
                    max_samples=max_samples,
                )

            self.dataset_train = make_phase1("train", None, self.reverse_complement_probability)
            self.dataset_val = make_phase1("val", self.val_samples_per_epoch, 0.0)
            self.dataset_test = make_phase1("test", self.test_samples_per_epoch, 0.0)
            for split, dataset in (("train", self.dataset_train), ("val", self.dataset_val), ("test", self.dataset_test)):
                print(
                    "[OneMaizeDNAMLM] "
                    f"mode=full_genome split={split} windows={len(dataset)} "
                    f"valid_bp={dataset.total_valid_bp} tails={dataset.tail_count} "
                    f"context={self.context_length} rc={dataset.reverse_complement_probability}"
                )
            return

        if self.train_samples_per_epoch is None:
            raise ValueError("region_aware mode requires train_samples_per_epoch")

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
                max_n_fraction=self.max_n_fraction,
                max_crop_attempts=self.max_crop_attempts,
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
        if self.mode == "full_genome":
            return collate_onemaize_phase1_mlm(
                batch, pad_token_id=int(self.tokenizer.pad_token_id)
            )
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
