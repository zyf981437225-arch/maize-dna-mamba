from pathlib import Path

import pytest

from src.onemaize.checkpoint_contracts import (
    validate_allcultivar_resume_config,
    validate_phase1_initialization_config,
)


def test_model_b_initialization_requires_b73_phase1():
    validate_phase1_initialization_config(
        {"dataset": {"mode": "full_genome", "context_length": 8192, "genotype": "B73"}}
    )
    with pytest.raises(ValueError, match="Not a B73 Phase-I"):
        validate_phase1_initialization_config(
            {"dataset": {"mode": "region_aware", "context_length": 16384, "genotype": "B73"}}
        )


def test_model_a_checkpoint_cannot_be_used_as_model_b_resume(tmp_path):
    model_b_data = tmp_path / "nam26-schema-v3"
    model_a_data = tmp_path / "b73-schema-v3"
    model_b_data.mkdir()
    model_a_data.mkdir()
    validate_allcultivar_resume_config(
        {"dataset": {"mode": "region_aware", "context_length": 16384, "data_dir": str(model_b_data)}},
        model_b_data,
    )
    with pytest.raises(ValueError, match="Not an exact all-cultivar Model-B resume"):
        validate_allcultivar_resume_config(
            {"dataset": {"mode": "region_aware", "context_length": 16384, "data_dir": str(model_a_data)}},
            model_b_data,
        )
