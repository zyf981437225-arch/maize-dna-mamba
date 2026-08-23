"""Regression tests for valid-length sequence pooling."""

import torch

from src.tasks.decoders import SequenceDecoder


def test_pooling_ignores_right_padding_with_lengths():
    decoder = SequenceDecoder(
        d_model=2,
        d_output=1,
        l_output=0,
        use_lengths=True,
        mode="pool",
    )
    with torch.no_grad():
        decoder.output_transform.weight.fill_(1.0)
        decoder.output_transform.bias.zero_()

    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]],
        ]
    )
    lengths = torch.tensor([2, 3])

    output = decoder(hidden, lengths=lengths)

    assert output.shape == (2, 1)
    assert torch.allclose(output, torch.tensor([[5.0], [8.0]]))


def test_pooling_without_lengths_uses_sequence_axis():
    decoder = SequenceDecoder(
        d_model=2,
        d_output=None,
        l_output=0,
        use_lengths=False,
        mode="pool",
    )
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])

    output = decoder(hidden)

    assert output.shape == (1, 2)
    assert torch.allclose(output, torch.tensor([[2.0, 4.0]]))
