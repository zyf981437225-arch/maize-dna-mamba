from pathlib import Path

import torch


def test_mixed_rna_mlm_is_deterministic_and_uses_bases_only():
    from caduceus.tokenization_caduceus import CaduceusTokenizer
    from src.dataloaders.datasets.mixed_rna_dataset import MixedRNADataset

    root = Path(__file__).resolve().parents[2]
    tokenizer = CaduceusTokenizer(model_max_length=64)
    dataset = MixedRNADataset(
        tokenizer=tokenizer,
        text_file=str(root / "tests/fixtures/mixed_rna_small.txt"),
        fasta_file=str(root / "tests/fixtures/mixed_rna_small.fasta"),
        max_length=64,
        add_eos=True,
        mlm=True,
        mlm_probability=1.0,
        deterministic_mlm=True,
        mlm_seed=123,
    )

    input_a, labels_a = dataset[0]
    input_b, labels_b = dataset[0]

    assert torch.equal(input_a, input_b)
    assert torch.equal(labels_a, labels_b)

    vocab = tokenizer.get_vocab()
    allowed = {
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        *(vocab[token] for token in ("A", "C", "G", "U", "N")),
    }
    assert set(input_a.tolist()).issubset(allowed)

    valid_labels = labels_a[labels_a != tokenizer.pad_token_id]
    assert valid_labels.numel() > 0
    assert set(valid_labels.tolist()).issubset(
        {vocab[token] for token in ("A", "C", "G", "U", "N")}
    )
