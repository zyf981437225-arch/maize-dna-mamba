"""Caduceus model for Hugging Face.

"""
from __future__ import annotations

from caduceus.memory_pool import MemoryPool
from caduceus.memory_cross_attn import MemoryCrossAttention
from .memory.writer import BidirectionalMemoryWriter
import inspect
import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.modules.mamba_simple import Block  # Legacy mambav1 file structure
except ImportError:
    from mamba_ssm.modules.block import Block  # mambav2 file structure
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention, MaskedLMOutput, SequenceClassifierOutput

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn  # Legacy mambav1 file structure
except ImportError:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn  # mambav2 file structure
    except ImportError:
        RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .configuration_caduceus import CaduceusConfig
from .modeling_rcps import RCPSAddNormWrapper, RCPSEmbedding, RCPSLMHead, RCPSMambaBlock


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        bidirectional=True,
        bidirectional_strategy="add",
        bidirectional_weight_tie=True,
        rcps=False,
        device=None,
        dtype=None,
):
    """Create Caduceus block.

    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    mixer_cls = partial(BiMambaWrapper, layer_idx=layer_idx, **ssm_cfg, **bidirectional_kwargs, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block_cls = RCPSMambaBlock if rcps else Block
    # mambav2 compatibility
    if "mlp_cls" in inspect.signature(block_cls.__init__).parameters:
        block = block_cls(
            d_model,
            mixer_cls,
            mlp_cls=nn.Identity,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    else:
        block = block_cls(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    block.layer_idx = layer_idx
    return block


def _get_bimamba_from_layer(layer):
    if hasattr(layer, "mixer") and hasattr(layer.mixer, "last_fwd"):
        return layer.mixer
    if hasattr(layer, "mixer") and hasattr(layer.mixer, "mixer") and hasattr(layer.mixer.mixer, "last_fwd"):
        return layer.mixer.mixer
    return None


def _valid_prefix_reverse_indices(attention_mask, validate=False):
    """Build the self-inverse gather indices for right-padded sequences."""

    mask = attention_mask.to(dtype=torch.bool)
    positions = torch.arange(mask.shape[1], device=mask.device)
    lengths = mask.sum(dim=1)
    expected = positions.unsqueeze(0) < lengths.unsqueeze(1)
    if validate and not torch.equal(mask, expected):
        raise ValueError("BiMamba requires attention_mask to be a contiguous valid prefix")
    return torch.where(
        expected,
        lengths.unsqueeze(1) - 1 - positions.unsqueeze(0),
        positions.unsqueeze(0),
    )


def _reverse_valid_prefix(
    hidden_states,
    attention_mask,
    reverse_indices=None,
    validate=False,
):
    """Reverse only each sample's contiguous non-PAD prefix.

    A plain sequence flip moves right padding in front of the biological
    sequence and contaminates the reverse Mamba state. This gather is
    self-inverse: applying it before and after the reverse mixer restores the
    original right-padded layout without allowing PAD tokens to precede valid
    nucleotides.
    """

    if attention_mask is None:
        return hidden_states.flip(dims=(1,))
    if attention_mask.ndim != 2 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            "attention_mask must have shape [batch, length] matching hidden_states"
        )
    if reverse_indices is None:
        reverse_indices = _valid_prefix_reverse_indices(
            attention_mask.to(device=hidden_states.device), validate=validate
        )
    gather_indices = reverse_indices.unsqueeze(-1).expand_as(hidden_states)
    return hidden_states.gather(1, gather_indices)


class BiMambaWrapper(nn.Module):
    def __init__(
            self,
            d_model,
            bidirectional=False,
            bidirectional_strategy="add",
            bidirectional_weight_tie=False,
            layer_idx=None,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional_strategy is None:
            bidirectional_strategy = "add"
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(
                f"`{bidirectional_strategy}` strategy for bidirectionality is not implemented!"
            )

        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy

        # 注意：mamba_kwargs 里会包含 layer_idx/device/dtype 以及 ssm_cfg 展开的参数
        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)

        self.mamba_rev = None
        if bidirectional:
            self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs)
            if bidirectional_weight_tie:
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias

        # 缓存两路输出给 memory sidecar
        self.last_fwd = None
        self.last_bwd = None
        self.attention_mask = None
        self.reverse_indices = None

    def forward(self, hidden_states, inference_params=None):
        out_fwd = self.mamba_fwd(
            hidden_states,
            inference_params=inference_params,
        )
        out = out_fwd
        out_rev = None

        if self.bidirectional:
            reversed_states = _reverse_valid_prefix(
                hidden_states, self.attention_mask, self.reverse_indices
            )
            out_rev = self.mamba_rev(
                reversed_states,
                inference_params=inference_params,
            )
            out_rev = _reverse_valid_prefix(
                out_rev, self.attention_mask, self.reverse_indices
            )

            if self.bidirectional_strategy == "add":
                out = out_fwd + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out_fwd * out_rev
            else:
                raise NotImplementedError(
                    f"Unknown bidirectional strategy {self.bidirectional_strategy}"
                )

        # cache for memory sidecar
        self.last_fwd = out_fwd
        self.last_bwd = out_rev if out_rev is not None else out_fwd

        return out


class CaduceusEmbeddings(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config.rcps:
            self.word_embeddings = RCPSEmbedding(
                config.vocab_size, config.d_model, config.complement_map, **factory_kwargs
            )
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, **factory_kwargs)

    def forward(self, input_ids):
        """
            input_ids: (batch, seqlen)
        """
        return self.word_embeddings(input_ids)




class MemoryWriterWrapper(nn.Module):

    def __init__(self, d_model: int, d_sum: int = 64, d_mem: int = 64,
                 pool: str = "mean", gate: str = "vector"):
        super().__init__()
        self.writer = BidirectionalMemoryWriter(
            d_model=d_model, d_sum=d_sum, d_mem=d_mem, pool=pool, gate=gate
        )

    def forward(self, hidden_states: torch.Tensor, attn_mask: torch.Tensor | None = None):
        entry, aux = self.writer(hidden_states, hidden_states, attn_mask=attn_mask)
        return entry, aux


class CaduceusMixerModel(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.config = config
        self.fused_add_norm = config.fused_add_norm
        self.rcps = config.rcps
        self.residual_in_fp32 = config.residual_in_fp32

        self.embeddings = CaduceusEmbeddings(config, **factory_kwargs)

        # Mamba changes the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    rcps=config.rcps,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )

        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f if (config.fused_add_norm or not config.rcps) else RCPSAddNormWrapper(norm_f)
        # ---- Optional memory sidecar ----
        if config.rcps and config.use_memory:
            raise ValueError(
                "The current memory sidecar is not reverse-complement equivariant. "
                "Set use_memory=false when rcps=true."
            )
        self.use_memory = config.use_memory
        self.memory_persist_across_batches = config.memory_persist_across_batches
        hidden_dim = config.d_model * (2 if config.rcps else 1)

        if self.use_memory:
            self.memory_writer = MemoryWriterWrapper(
                d_model=hidden_dim,
                d_sum=config.memory_d_sum,
                d_mem=config.memory_d_mem,
                pool="mean",
                gate="vector",
            )
            self.memory_pool = MemoryPool(max_size=config.memory_max_size)
            self.memory_attn = MemoryCrossAttention(
                d_model=hidden_dim,
                d_mem=config.memory_d_mem,
                n_heads=config.memory_n_heads,
            )
            self.memory_write_stride = config.memory_write_stride
            self.memory_read_stride = config.memory_read_stride

    def forward(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        if self.use_memory and not self.memory_persist_across_batches:
            self.memory_pool.reset()

        all_hidden_states = []

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        if attention_mask is None and input_ids is not None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
            if pad_token_id is not None:
                attention_mask = input_ids.ne(pad_token_id)
        reverse_indices = (
            _valid_prefix_reverse_indices(attention_mask)
            if attention_mask is not None
            else None
        )

        residual = None

        # Snapshot only entries from earlier batches. Entries written below are
        # read through local_entries so their writer graph remains trainable.
        memory_global = None
        if self.use_memory and self.memory_persist_across_batches:
            memory_global = self.memory_pool.get()
            if memory_global is not None and memory_global.shape[0] != hidden_states.shape[0]:
                self.memory_pool.reset()
                memory_global = None

        local_entries = []

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            bim = _get_bimamba_from_layer(layer)
            if bim is not None:
                bim.attention_mask = attention_mask
                bim.reverse_indices = reverse_indices
            hidden_states, residual = layer(hidden_states, residual, inference_params=None)

            if self.use_memory:
                # Read only memories that existed before this layer. This avoids
                # letting a layer attend to a summary of its own output.
                memory_local = (
                    torch.stack(local_entries, dim=1)
                    if local_entries else None
                )
                if memory_global is None:
                    memory = memory_local
                elif memory_local is None:
                    memory = memory_global
                else:
                    memory = torch.cat([memory_global, memory_local], dim=1)

                if (
                    memory is not None
                    and memory.shape[1] > 0
                    and (i % self.memory_read_stride) == 0
                ):
                    hidden_states = hidden_states + self.memory_attn(hidden_states, memory)

                if (i % self.memory_write_stride) == 0:
                    bim = _get_bimamba_from_layer(layer)
                    if bim is not None:
                        entry, _ = self.memory_writer.writer(
                            bim.last_fwd,
                            bim.last_bwd,
                            attn_mask=attention_mask,
                        )
                    else:
                        entry, _ = self.memory_writer(
                            hidden_states,
                            attn_mask=attention_mask,
                        )
                    local_entries.append(entry)
                    if self.memory_persist_across_batches:
                        self.memory_pool.push(entry)

        # ---- final norm  ----
        if not self.fused_add_norm:
            if self.rcps:
                hidden_states = self.norm_f(hidden_states, residual=residual, prenorm=False)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            if self.rcps:
                hidden_states_fwd = fused_add_norm_fn(
                    hidden_states[..., :hidden_states.shape[-1] // 2],
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., :hidden_states.shape[-1] // 2],
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states_rc = fused_add_norm_fn(
                    hidden_states[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            else:
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            if output_hidden_states and (
                    len(all_hidden_states) == 0 or all_hidden_states[-1] is not hidden_states
            ):
                all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states


def cross_entropy(logits, y, ignore_index=-100):
    """Cross entropy loss."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def weighted_cross_entropy(logits, y, loss_weights, ignore_index=-100):
    """Weighted cross entropy loss (discounts certain tokens, e.g., repeated base pairs in genome)."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="none")
    loss_weights = loss_weights.view(-1)
    loss_weights[y == ignore_index] = 0.0
    # TODO: Follows GPN implementation, but should we remove weight normalization?
    return (ce * (loss_weights / loss_weights.sum())).sum()


class CaduceusPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for Caduceus backbone."""
    config_class = CaduceusConfig
    base_model_prefix = "caduceus"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BiMambaWrapper"]

    def _init_weights(
            self,
            module,
            initializer_range=0.02,  # Now only used for embedding layer.
            **kwargs,
    ):
        """Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py"""

        n_layer = self.config.n_layer
        initialized_cfg = self.config.initializer_cfg if self.config.initializer_cfg is not None else {}
        rescale_prenorm_residual = initialized_cfg.get("rescale_prenorm_residual", True)
        initializer_range = initialized_cfg.get("initializer_range", initializer_range)
        n_residuals_per_layer = initialized_cfg.get("n_residuals_per_layer", 1)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth.
            #   > Scale the weights of residual layers at initialization by a factor of 1/√N where N is the # of
            #   residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class Caduceus(CaduceusPreTrainedModel):
    """Caduceus model that can be instantiated using HF patterns."""

    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config)

        if config.rcps:
            assert config.complement_map is not None, "Complement map must be provided for RCPS."

        # Adjust vocab size and complement maps if vocab padding is set.
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (config.vocab_size % config.pad_vocab_size_multiple)
        if config.complement_map is not None and config.vocab_size > len(config.complement_map):
            for i in range(len(config.complement_map), config.vocab_size):
                config.complement_map[i] = i

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}

        self.backbone = CaduceusMixerModel(
            config,
            **factory_kwargs,
            **kwargs
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple, BaseModelOutputWithNoAttention]:

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        hidden_states, all_hidden_states = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )

        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None
            )
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states


class CaduceusForMaskedLM(CaduceusPreTrainedModel):
    """HF-compatible Caduceus model for masked language modeling."""

    def __init__(self, config: CaduceusConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        if config.rcps:
            self.lm_head = RCPSLMHead(
                complement_map=self.config.complement_map,  # Use caduceus config as it might have been updated
                vocab_size=self.config.vocab_size,  # Use caduceus config as it might have been updated
                true_dim=config.d_model,
                dtype=dtype
            )
        else:
            self.lm_head = nn.Linear(
                config.d_model,
                self.config.vocab_size,  # Use caduceus config as it might have been updated
                bias=False,
                **factory_kwargs
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.caduceus.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError("Setting input embeddings for RCPS LM is not supported.")
        self.caduceus.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Overrides output embeddings."""
        if self.config.rcps:
            raise NotImplementedError("Setting output embeddings for RCPS LM is not supported.")
        self.lm_head = new_embeddings

    def tie_weights(self):
        """Tie weights, accounting for RCPS."""
        if self.config.rcps:
            self.lm_head.set_weight(self.get_input_embeddings().weight)
        else:
            super().tie_weights()

    def get_decoder(self):
        """Get decoder (backbone) for the model."""
        return self.caduceus

    def set_decoder(self, decoder):
        """Set decoder (backbone) for the model."""
        self.caduceus = decoder

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            loss_weights: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """HF-compatible forward method."""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.caduceus(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss = weighted_cross_entropy(logits, labels, loss_weights, ignore_index=self.config.pad_token_id)
            else:
                loss = cross_entropy(logits, labels, ignore_index=self.config.pad_token_id)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class CaduceusForSequenceClassification(CaduceusPreTrainedModel):
    def __init__(
            self,
            config: CaduceusConfig,
            pooling_strategy: str = "mean",
            conjoin_train: bool = False,
            conjoin_eval: bool = False,
            device=None,
            dtype=None,
            **kwargs):
        super().__init__(config, **kwargs)
        if pooling_strategy not in ["mean", "max", "first", "last"]:
            raise NotImplementedError(f"Pooling strategy `{pooling_strategy}` not implemented.")
        self.pooling_strategy = pooling_strategy
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_labels = kwargs.get("num_labels", config.num_labels)
        self.caduceus = Caduceus(config, **factory_kwargs, **kwargs)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

        self.conjoin_train = conjoin_train
        self.conjoin_eval = conjoin_eval

        # Initialize weights and apply final processing
        self.post_init()
        self.init_scorer()

    def init_scorer(self, initializer_range=0.02):
        initializer_range = self.config.initializer_cfg.get("initializer_range", initializer_range) \
            if self.config.initializer_cfg is not None else initializer_range
        self.score.weight.data.normal_(std=initializer_range)

    def get_input_embeddings(self):
        return self.caduceus.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        if self.config.rcps:
            raise NotImplementedError("Setting input embeddings for RCPS LM is not supported.")
        self.caduceus.embeddings.word_embeddings = value

    def pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if self.pooling_strategy == "mean":  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if self.pooling_strategy == "max":  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if self.pooling_strategy == "last":  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[-1, ...]
        if self.pooling_strategy == "first":  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get hidden representations from the backbone
        if self.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = torch.stack(
                [
                    transformer_outputs[0][..., :self.config.d_model],
                    torch.flip(transformer_outputs[0][..., self.config.d_model:], dims=[1, 2])
                ],
                dim=-1
            )
        elif self.conjoin_train or (self.conjoin_eval and not self.training):  # For conjoining / post-hoc conjoining
            assert input_ids is not None, "`input_ids` must be provided for conjoining."
            assert input_ids.ndim == 3, "`input_ids` must be 3D tensor: channels corresponds to forward and rc strands."
            transformer_outputs = self.caduceus(
                input_ids[..., 0],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            transformer_outputs_rc = self.caduceus(
                input_ids[..., 1],
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # Stack along channel dimension (dim=-1)
            hidden_states = torch.stack([transformer_outputs[0], transformer_outputs_rc[0]], dim=-1)
        else:
            transformer_outputs = self.caduceus(
                input_ids,
                inputs_embeds=None,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]

        # Pool and get logits
        pooled_hidden_states = self.pool_hidden_states(hidden_states)
        # Potentially run `score` twice (with parameters shared) for conjoining
        if hidden_states.ndim == 4:  # bsz, seq_len, hidden_dim, 2 where last channel has the stacked fwd and rc reps
            logits_fwd = self.score(pooled_hidden_states[..., 0])
            logits_rc = self.score(pooled_hidden_states[..., 1])
            logits = (logits_fwd + logits_rc) / 2
        else:
            logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )
