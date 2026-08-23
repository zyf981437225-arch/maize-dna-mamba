import pytest
import torch

from caduceus.modeling_caduceus import _reverse_valid_prefix


def test_reverse_valid_prefix_keeps_right_padding_at_the_end():
    hidden = torch.arange(10, dtype=torch.float32).reshape(2, 5, 1)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    reversed_hidden = _reverse_valid_prefix(hidden, mask)
    assert reversed_hidden[:, :, 0].tolist() == [
        [2.0, 1.0, 0.0, 3.0, 4.0],
        [9.0, 8.0, 7.0, 6.0, 5.0],
    ]
    assert torch.equal(_reverse_valid_prefix(reversed_hidden, mask), hidden)


def test_reverse_valid_prefix_rejects_noncontiguous_masks():
    hidden = torch.zeros(1, 4, 2)
    mask = torch.tensor([[True, False, True, False]])
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        _reverse_valid_prefix(hidden, mask, validate=True)
