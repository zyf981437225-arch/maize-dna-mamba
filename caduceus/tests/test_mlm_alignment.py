from types import SimpleNamespace

import torch


class _Encoder:
    def __call__(self, x, **_kwargs):
        return x, {}


class _Decoder:
    def __call__(self, x, **_kwargs):
        return x, {}


class _Model(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, _input_ids):
        return SimpleNamespace(logits=self.logits)


def test_mlm_task_keeps_same_position_alignment():
    from src.tasks.tasks import LMTask

    batch_size, length, vocab_size = 2, 5, 12
    input_ids = torch.randint(0, vocab_size, (batch_size, length))
    labels = torch.tensor(
        [[4, 7, 4, 9, 4], [8, 4, 6, 4, 10]],
        dtype=torch.long,
    )
    logits = torch.randn(batch_size, length, vocab_size)

    task = LMTask.__new__(LMTask)
    task.dataset = SimpleNamespace(mlm=True)
    task._state = None
    flattened_logits, flattened_labels, _ = task.forward(
        (input_ids, labels),
        _Encoder(),
        _Model(logits),
        _Decoder(),
        None,
    )

    assert flattened_logits.shape == (batch_size * length, vocab_size)
    assert torch.equal(flattened_logits, logits.reshape(-1, vocab_size))
    assert torch.equal(flattened_labels, labels.reshape(-1))
