import torch

def test_memory_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from caduceus.modeling_caduceus import CaduceusMixerModel
    from caduceus.configuration_caduceus import CaduceusConfig

    config = CaduceusConfig(
        d_model=64,
        n_layer=4,
        vocab_size=128,
        fused_add_norm=False,
        use_memory=True,
        memory_d_sum=32,
        memory_d_mem=16,
        memory_n_heads=4,
    )
    model = CaduceusMixerModel(config).to(device).train()

    model.memory_read_stride = 1
    model.memory_write_stride = 2

    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)

    out, _ = model(input_ids=input_ids)

    loss = out.float().pow(2).mean()
    loss.backward()

    memory_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if "memory_writer" in name or "memory_attn" in name
    }
    assert memory_parameters
    for name, parameter in memory_parameters.items():
        assert parameter.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"Invalid gradient in {name}"

    assert sum(
        parameter.grad.abs().sum() for parameter in memory_parameters.values()
    ) > 0
