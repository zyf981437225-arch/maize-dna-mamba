"""Hugging Face config, model, and tokenizer for Caduceus.

"""

from .configuration_caduceus import CaduceusConfig
from .tokenization_caduceus import CaduceusTokenizer

__all__ = [
    "CaduceusConfig",
    "CaduceusTokenizer",
    "Caduceus",
    "CaduceusForMaskedLM",
    "CaduceusForSequenceClassification",
]


def __getattr__(name):
    """Load CUDA/Mamba model code only when a model class is requested.

    Tokenization and data-preparation utilities remain usable on CPU machines
    that do not install the Linux-only ``mamba_ssm`` kernels.
    """

    if name in {
        "Caduceus",
        "CaduceusForMaskedLM",
        "CaduceusForSequenceClassification",
    }:
        from .modeling_caduceus import (
            Caduceus,
            CaduceusForMaskedLM,
            CaduceusForSequenceClassification,
        )

        return {
            "Caduceus": Caduceus,
            "CaduceusForMaskedLM": CaduceusForMaskedLM,
            "CaduceusForSequenceClassification": CaduceusForSequenceClassification,
        }[name]
    raise AttributeError(name)
