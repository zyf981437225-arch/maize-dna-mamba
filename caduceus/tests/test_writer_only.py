import torch


def test_writer_only_is_padding_aware_and_pools_before_projection():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from caduceus.memory.writer import BidirectionalMemoryWriter

    batch, length, width = 2, 16, 64
    writer = BidirectionalMemoryWriter(
        d_model=width,
        d_sum=32,
        d_mem=128,
    ).to(device).eval()

    hidden = torch.randn(batch, length, width, device=device)
    attention_mask = torch.ones(batch, length, device=device)
    attention_mask[:, -4:] = 0
    changed_padding = hidden.clone()
    changed_padding[:, -4:] = torch.randn_like(changed_padding[:, -4:]) * 100

    projection_shapes = []
    handle = writer.shared_projection.register_forward_pre_hook(
        lambda _module, args: projection_shapes.append(args[0].shape)
    )
    with torch.no_grad():
        entry, aux = writer(hidden, hidden, attn_mask=attention_mask)
        changed_entry, _ = writer(
            changed_padding,
            changed_padding,
            attn_mask=attention_mask,
        )
    handle.remove()

    assert entry.shape == (batch, 128)
    assert set(aux) == {"s_fwd", "s_bwd", "gate", "s"}
    assert projection_shapes == [
        torch.Size([2 * batch, width]),
        torch.Size([2 * batch, width]),
    ]
    assert torch.allclose(entry, changed_entry, atol=1e-5, rtol=1e-5)
