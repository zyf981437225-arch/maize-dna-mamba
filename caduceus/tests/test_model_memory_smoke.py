import torch

def test_model_memory_smoke():
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

    model = CaduceusMixerModel(config).to(device).eval()

    # 先固定住 stride，避免复杂度
    model.memory_write_stride = 2
    model.memory_read_stride = 1

    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    attn_mask = torch.ones(B, T, device=device)

    with torch.no_grad():
        out, all_h = model(input_ids=input_ids)

    print("output shape:", out.shape)
    assert out.shape[0] == B
    assert out.shape[1] == T


    # 只要能跑通 + shape 对，就算过
    if hasattr(out, "last_hidden_state"):
        x = out.last_hidden_state
    elif isinstance(out, (tuple, list)):
        x = out[0]
    else:
        x = out

    print("output shape:", x.shape)
    assert x.shape[0] == B
    assert x.shape[1] == T
