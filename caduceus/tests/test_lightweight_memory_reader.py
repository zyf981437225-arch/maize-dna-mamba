import torch


def test_lightweight_reader_broadcasts_one_memory_summary():
    from caduceus.memory_cross_attn import MemoryCrossAttention

    reader = MemoryCrossAttention(d_model=32, d_mem=8)
    hidden = torch.randn(2, 128, 32)
    memory = torch.randn(2, 3, 8, requires_grad=True)

    output = reader(hidden, memory)
    assert output.shape == (2, 1, 32)

    output.square().mean().backward()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()
    assert memory.grad.abs().sum() > 0
