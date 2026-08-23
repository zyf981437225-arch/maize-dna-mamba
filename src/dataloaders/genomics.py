"""Dataloaders for genomics datasets, including pretraining and downstream tasks.

    - Adapted from:
        https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
    - Adapted from:
        https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""

import copy
from typing import Any
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, Subset

from caduceus.tokenization_caduceus import CaduceusTokenizer
import src.utils.train
from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.datasets.genomic_bench_dataset import GenomicBenchmarkDataset
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.kmer_tokenizer import KmerTokenizer
from src.dataloaders.datasets.hg38_dataset import HG38Dataset
from src.dataloaders.datasets.bacteria_txt_dataset import BacteriaTxtDataset
from src.dataloaders.datasets.rnacentral_fasta_dataset import RNACentralFastaDataset
from src.dataloaders.datasets.mixed_rna_dataset import MixedRNADataset
from src.dataloaders.datasets.nucleotide_transformer_dataset import NucleotideTransformerDataset
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler

logger = src.utils.train.get_logger(__name__)


class HG38(SequenceDataset):
    _name_ = "hg38"

    def __init__(
        self,
        bed_file,
        fasta_file,
        tokenizer_name=None,
        dataset_config_name=None,
        max_length=1024,
        d_output=2,
        rc_aug=True,
        max_length_val=None,
        max_length_test=None,
        val_ratio=0.0005,
        val_split_seed=2357,
        add_eos=True,
        detokenize=False,
        val_only=False,
        batch_size=32,
        batch_size_eval=None,
        shuffle=False,
        num_workers=1,
        fault_tolerant=False,
        ddp=False,
        fast_forward_epochs=None,
        fast_forward_batches=None,
        mlm=True,
        mlm_probability=0.15,
        *args,
        **kwargs,
    ):
        # extra dataset config fields
        self.dataset_name = kwargs.pop("dataset_name", None)
        self.text_file = kwargs.pop("text_file", None)
        self.rna_fasta_file = kwargs.pop("rna_fasta_file", None)
        if self.text_file is not None:
            self.text_file = to_absolute_path(self.text_file)
        if self.rna_fasta_file is not None:
            self.rna_fasta_file = to_absolute_path(self.rna_fasta_file)
        self.ignore_id = kwargs.pop("ignore_id", None)
        self.kmer = kwargs.pop("kmer", 1)
        self.frame = kwargs.pop("frame", 0)

        # optional controls for mixed dataset
        self.max_text_sequences = kwargs.pop("max_text_sequences", None)
        self.max_fasta_sequences = kwargs.pop("max_fasta_sequences", None)

        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.bed_file = bed_file
        self.fasta_file = fasta_file

        if self.bed_file is None:
            self.bed_file = default_data_path / self._name_ / "human-sequences.bed"
        if self.fasta_file is None:
            self.fasta_file = default_data_path / self._name_ / "hg38.ml.fa"

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant

        if ddp:
            assert fault_tolerant
        self.ddp = ddp

        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        self.mlm = mlm
        self.mlm_probability = mlm_probability

        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None):
        if self.tokenizer_name == "char":
            logger.info("**Using Char-level tokenizer**")
            self.tokenizer = CaduceusTokenizer(
                model_max_length=self.max_length,
                add_special_tokens=False,
            )
        elif self.tokenizer_name == "kmer3":
            logger.info("**Using 3-mer tokenizer (non-overlap, random frame in dataset)**")
            self.tokenizer = KmerTokenizer(
                k=3,
                alphabet="ACGU",
                model_max_length=self.max_length,
            )
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()

    def _make_splits(self, n: int):
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        n_test = n - n_train - n_val

        g = torch.Generator().manual_seed(self.val_split_seed)
        perm = torch.randperm(n, generator=g).tolist()

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
        return train_idx, val_idx, test_idx

    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)."""

        # cleanup old hg38 datasets if present
        if hasattr(self, "dataset_train"):
            if hasattr(self.dataset_train, "fasta"):
                self.dataset_train.fasta.seqs.close()
                del self.dataset_train.fasta.seqs

        if hasattr(self, "dataset_test"):
            if hasattr(self.dataset_test, "fasta"):
                self.dataset_test.fasta.seqs.close()
                del self.dataset_test.fasta.seqs

        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        ignore_id = getattr(self, "ignore_id", pad_id)

        # ------------------------------------------------------------------
        # 1) bacteria txt only
        # ------------------------------------------------------------------
        if self.dataset_name == "bacteria_txt" and self.text_file is not None:
            base_dataset = BacteriaTxtDataset(
                text_file=self.text_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_eos=self.add_eos,
                mlm=True,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
            )

            train_idx, val_idx, test_idx = self._make_splits(len(base_dataset))

            train_ds = BacteriaTxtDataset(
                text_file=self.text_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_eos=self.add_eos,
                mlm=self.mlm,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
            )

            val_ds = BacteriaTxtDataset(
                text_file=self.text_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length_val,
                add_eos=self.add_eos,
                mlm=self.mlm,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
            )

            test_ds = BacteriaTxtDataset(
                text_file=self.text_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length_test,
                add_eos=self.add_eos,
                mlm=self.mlm,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
            )

            self.dataset_train = Subset(train_ds, train_idx)
            self.dataset_val = Subset(val_ds, val_idx)
            self.dataset_test = Subset(test_ds, test_idx)
            return

        # ------------------------------------------------------------------
        # 2) rnacentral fasta only
        # ------------------------------------------------------------------
        if self.dataset_name == "rnacentral_fasta" and self.rna_fasta_file is not None:
            base_dataset = RNACentralFastaDataset(
                fasta_file=self.rna_fasta_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_eos=self.add_eos,
                mlm=True,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
            )

            train_idx, val_idx, test_idx = self._make_splits(len(base_dataset))

            train_ds = RNACentralFastaDataset(
                fasta_file=self.rna_fasta_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_eos=self.add_eos,
                mlm=self.mlm,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
            )

            val_ds = RNACentralFastaDataset(
                fasta_file=self.rna_fasta_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length_val,
                add_eos=self.add_eos,
                mlm=self.mlm,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
            )

            test_ds = RNACentralFastaDataset(
                fasta_file=self.rna_fasta_file,
                tokenizer=self.tokenizer,
                max_length=self.max_length_test,
                add_eos=self.add_eos,
                mlm=self.mlm,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
            )

            self.dataset_train = Subset(train_ds, train_idx)
            self.dataset_val = Subset(val_ds, val_idx)
            self.dataset_test = Subset(test_ds, test_idx)

            return

        # ------------------------------------------------------------------
        # 3) mixed txt + fasta
        # ------------------------------------------------------------------
        if self.dataset_name == "mixed_rna":
            if self.text_file is None and self.rna_fasta_file is None:
                raise ValueError("For dataset_name='mixed_rna', provide text_file and/or rna_fasta_file.")

            base_dataset = MixedRNADataset(
                tokenizer=self.tokenizer,
                text_file=self.text_file,
                fasta_file=self.rna_fasta_file,
                max_length=self.max_length,
                add_eos=self.add_eos,
                mlm=True,
                mlm_probability=self.mlm_probability,
                ignore_id=ignore_id,
                kmer=self.kmer,
                frame=self.frame,
                max_text_sequences=self.max_text_sequences,
                max_fasta_sequences=self.max_fasta_sequences,
            )

            train_idx, val_idx, test_idx = self._make_splits(len(base_dataset))

            # Shallow copies share the immutable sequence/source lists. This
            # avoids parsing and storing the complete corpus three more times
            # in every DDP process.
            train_ds = copy.copy(base_dataset)
            val_ds = copy.copy(base_dataset)
            test_ds = copy.copy(base_dataset)

            train_ds.max_length = self.max_length
            train_ds.mlm = self.mlm
            train_ds.deterministic_mlm = False

            val_ds.max_length = self.max_length_val
            val_ds.mlm = self.mlm
            val_ds.deterministic_mlm = True
            val_ds.mlm_seed = self.val_split_seed + 10_000

            test_ds.max_length = self.max_length_test
            test_ds.mlm = self.mlm
            test_ds.deterministic_mlm = True
            test_ds.mlm_seed = self.val_split_seed + 20_000

            self.dataset_train = Subset(train_ds, train_idx)
            self.dataset_val = Subset(val_ds, val_idx)
            self.dataset_test = Subset(test_ds, test_idx)

            return

        # ------------------------------------------------------------------
        # 4) original hg38 fallback
        # ------------------------------------------------------------------
        self.dataset_train, self.dataset_val, self.dataset_test = [
            HG38Dataset(
                split=split,
                bed_file=self.bed_file,
                fasta_file=self.fasta_file,
                max_length=max_len,
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                add_eos=self.add_eos,
                return_seq_indices=False,
                rc_aug=self.rc_aug,
                return_augs=False,
                mlm=self.mlm,
                mlm_probability=self.mlm_probability,
            )
            for split, max_len in zip(
                ["train", "valid", "test"],
                [self.max_length, self.max_length_val, self.max_length_test],
            )
        ]

    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            sampler = (
                FaultTolerantDistributedSampler(
                    self.dataset_train,
                    **distributed_sampler_kwargs
                ) if self.ddp else RandomFaultTolerantSampler(self.dataset_train)
            )

            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    "epoch": self.fast_forward_epochs,
                    "counter": self.fast_forward_batches * self.batch_size,
                })
        else:
            shuffle = self.shuffle
            sampler = None

        return self._data_loader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            max_length=getattr(self, "max_length_train", self.max_length),
            **kwargs,
        )

    def val_dataloader(self, **kwargs):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._data_loader(
            self.dataset_val,
            batch_size=self.batch_size_eval,
            max_length=self.max_length_val,
            **kwargs,
        )

    def test_dataloader(self, **kwargs):
        kwargs["drop_last"] = False
        kwargs["shuffle"] = False
        return self._data_loader(
            self.dataset_test,
            batch_size=self.batch_size_eval,
            max_length=self.max_length_test,
            **kwargs,
        )

    def _data_loader(self, dataset, batch_size: int, shuffle: bool = False, sampler=None, max_length=None, **kwargs):
        if max_length is None:
            max_length = self.max_length

        real_dataset = dataset
        while isinstance(real_dataset, Subset):
            real_dataset = real_dataset.dataset

        if hasattr(real_dataset, "max_length"):
            real_dataset.max_length = max_length

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
            self.fast_forward_batches = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]["completed"]


class GenomicBenchmark(HG38):
    _name_ = "genomic_benchmark"
    l_output = 0

    def __init__(
        self,
        dataset_name,
        train_val_split_seed,
        dest_path=None,
        tokenizer_name="char",
        d_output=None,
        rc_aug=True,
        conjoin_train=False,
        conjoin_test=False,
        max_length=1024,
        use_padding=True,
        max_length_val=None,
        max_length_test=None,
        padding_side="left",
        val_ratio=0.0005,
        val_split_seed=2357,
        add_eos=False,
        detokenize=False,
        val_only=False,
        batch_size=32,
        batch_size_eval=None,
        num_workers=1,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        fault_tolerant=False,
        ddp=False,
        fast_forward_epochs=None,
        fast_forward_batches=None,
        *args,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.train_val_split_seed = train_val_split_seed
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == "char":
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,
                add_special_tokens=False,
                padding_side=self.padding_side,
            )

        self.dataset_train, self.dataset_test = [
            GenomicBenchmarkDataset(
                split=split,
                max_length=max_len,
                dataset_name=self.dataset_name,
                tokenizer=self.tokenizer,
                tokenizer_name=self.tokenizer_name,
                use_padding=self.use_padding,
                d_output=self.d_output,
                add_eos=self.add_eos,
                dest_path=self.dest_path,
                rc_aug=self.rc_aug,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False,
            )
            for split, max_len in zip(["train", "test"], [self.max_length, self.max_length_val])
        ]

        val_data, train_data = torch.utils.data.random_split(
            list(zip(self.dataset_train.all_seqs, self.dataset_train.all_labels)),
            lengths=[0.1, 0.9],
            generator=torch.Generator().manual_seed(self.train_val_split_seed),
        )
        self.dataset_val = copy.deepcopy(self.dataset_train)
        self.dataset_train.all_seqs = [train_data[i][0] for i in range(len(train_data))]
        self.dataset_train.all_labels = [train_data[i][1] for i in range(len(train_data))]
        self.dataset_val.all_seqs = [val_data[i][0] for i in range(len(val_data))]
        self.dataset_val.all_labels = [val_data[i][1] for i in range(len(val_data))]
        self.dataset_val.split = "val"


class NucleotideTransformer(HG38):
    _name_ = "nucleotide_transformer"
    l_output = 0

    def __init__(
        self,
        dataset_name,
        train_val_split_seed,
        tokenizer_name="char",
        d_output=None,
        rc_aug=True,
        conjoin_train=False,
        conjoin_test=False,
        max_length=1024,
        use_padding=True,
        max_length_val=None,
        max_length_test=None,
        padding_side="left",
        val_ratio=0.0005,
        val_split_seed=2357,
        add_eos=False,
        detokenize=False,
        val_only=False,
        batch_size=32,
        batch_size_eval=None,
        num_workers=1,
        shuffle=True,
        shuffle_eval=None,
        pin_memory=False,
        drop_last=False,
        fault_tolerant=False,
        ddp=False,
        fast_forward_epochs=None,
        fast_forward_batches=None,
        *args,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.train_val_split_seed = train_val_split_seed
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shuffle_eval = shuffle_eval if shuffle_eval is not None else shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        if self.tokenizer_name == "char":
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,
                add_special_tokens=False,
                padding_side=self.padding_side,
            )

        self.dataset_train, self.dataset_test = [
            NucleotideTransformerDataset(
                split=split,
                max_length=max_len,
                tokenizer=self.tokenizer,
                dataset_name=self.dataset_name,
                tokenizer_name=self.tokenizer_name,
                use_padding=self.use_padding,
                d_output=self.d_output,
                add_eos=self.add_eos,
                rc_aug=self.rc_aug,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False,
            )
            for split, max_len in zip(["train", "test"], [self.max_length, self.max_length_val])
        ]

        ds_train_val_split = self.dataset_train.seqs.train_test_split(
            test_size=0.1,
            seed=self.train_val_split_seed,
        )
        self.dataset_val = copy.deepcopy(self.dataset_train)
        self.dataset_train.seqs = ds_train_val_split["train"]
        self.dataset_val.split = "val"
        self.dataset_val.seqs = ds_train_val_split["test"]
