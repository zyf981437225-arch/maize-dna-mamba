import torch


def test_memory_eval_isolation():
    """Independent evaluation batches must not share hidden memory."""
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
        memory_persist_across_batches=False,
    )
    model = CaduceusMixerModel(config).to(device).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)

    with torch.no_grad():
        first, _ = model(input_ids=input_ids)
        second, _ = model(input_ids=input_ids)

    assert torch.equal(first, second)
