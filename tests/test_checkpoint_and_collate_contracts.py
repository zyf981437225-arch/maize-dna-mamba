import warnings

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from src.dataloaders.base import DefaultCollateMixin
from train import SequenceLightningModule


def _tiny_lightning_module():
    module = SequenceLightningModule.__new__(SequenceLightningModule)
    pl.LightningModule.__init__(module)
    module.model = torch.nn.Linear(2, 2)
    module.decoder = torch.nn.Linear(2, 1)
    module.save_hyperparameters(
        OmegaConf.create(
            {
                "train": {
                    "pretrained_model_state_hook": {
                        "_name_": "load_backbone",
                        "freeze_backbone": False,
                    }
                }
            }
        ),
        logger=False,
    )
    return module


def test_exact_checkpoint_restore_keeps_downstream_head():
    module = _tiny_lightning_module()
    restored = {
        name: torch.full_like(value, index + 1.0)
        for index, (name, value) in enumerate(module.state_dict().items())
    }
    module.load_state_dict(restored, strict=True)
    for name, value in module.state_dict().items():
        assert torch.equal(value, restored[name])


def test_warm_start_hook_still_leaves_downstream_head_fresh(tmp_path):
    module = _tiny_lightning_module()
    original_decoder = {
        name: value.detach().clone()
        for name, value in module.decoder.state_dict().items()
    }
    checkpoint_path = tmp_path / "pretrained.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.weight": torch.full_like(module.model.weight, 3.0),
                "model.bias": torch.full_like(module.model.bias, 4.0),
            }
        },
        checkpoint_path,
    )
    module.load_compatible_pretrained(checkpoint_path)
    assert torch.equal(module.model.weight, torch.full_like(module.model.weight, 3.0))
    assert torch.equal(module.model.bias, torch.full_like(module.model.bias, 4.0))
    for name, value in module.decoder.state_dict().items():
        assert torch.equal(value, original_decoder[name])


def test_matching_backbone_hook_ignores_same_shaped_decoder(tmp_path):
    module = _tiny_lightning_module()
    module.hparams.train.pretrained_model_state_hook._name_ = "load_matching_backbone"
    original_decoder = {
        name: value.detach().clone()
        for name, value in module.decoder.state_dict().items()
    }
    checkpoint_path = tmp_path / "downstream.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.weight": torch.full_like(module.model.weight, 5.0),
                "model.bias": torch.full_like(module.model.bias, 6.0),
                "decoder.weight": torch.full_like(module.decoder.weight, 7.0),
                "decoder.bias": torch.full_like(module.decoder.bias, 8.0),
            }
        },
        checkpoint_path,
    )
    module.load_compatible_pretrained(checkpoint_path)
    assert torch.equal(module.model.weight, torch.full_like(module.model.weight, 5.0))
    assert torch.equal(module.model.bias, torch.full_like(module.model.bias, 6.0))
    for name, value in module.decoder.state_dict().items():
        assert torch.equal(value, original_decoder[name])


def test_worker_collate_preallocates_final_output_shape(monkeypatch):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    batch = (torch.arange(4), torch.arange(4) + 10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = DefaultCollateMixin._collate(batch)
    assert output.shape == (2, 4)
    assert not any("was resized" in str(item.message) for item in caught)
    assert not any("TypedStorage is deprecated" in str(item.message) for item in caught)
