import torch


def test_memory_read_before_write():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    from caduceus.configuration_caduceus import CaduceusConfig
    from caduceus.modeling_caduceus import CaduceusMixerModel

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
    model = CaduceusMixerModel(config).to(device).eval()
    model.memory_read_stride = 1
    model.memory_write_stride = 1
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)

    read_calls = []
    handle = model.memory_attn.register_forward_hook(
        lambda _module, _inputs, _output: read_calls.append(1)
    )
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    handle.remove()

    # Layer 0 has no earlier memory to read. Every later layer can read.
    assert len(read_calls) == len(model.layers) - 1
