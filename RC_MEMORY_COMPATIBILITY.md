# RC and cross-layer memory compatibility audit

## Bottom line

The inherited production architecture is bidirectional but is not configured
as a strict reverse-complement-equivariant model. Maize DNA-Mamba sets
`bidirectional=true` and `rcps=false`. The repository contains a strict RCPS
implementation, but `CaduceusMixerModel` deliberately raises an error when
`rcps=true` and `use_memory=true`.

Consequently, Maize DNA-Mamba preserves the bidirectional Caduceus backbone
and memory sidecar, sets the correct DNA complement map, and does **not** claim
strict RC equivariance.

## Backbone mechanisms

### Ordinary bidirectional Mamba

`caduceus/modeling_caduceus.py:145-217`, class `BiMambaWrapper`, creates a
forward and reverse Mamba. It reverses the valid (non-PAD) sequence prefix,
runs the reverse mixer, aligns the output back to original positions, and adds
or multiplies the two streams. This is sequence-direction processing; it does
not complement nucleotide ids.

When `bidirectional_weight_tie=true`, lines 172-176 share only the input and
output projection parameters. Other Mamba parameters are not explicitly tied
there. This path alone is not sufficient evidence for strict reversal or
reverse-complement equivariance.

### Strict RCPS path

`caduceus/modeling_rcps.py` defines:

- `RCPSEmbedding`: embeds forward input and its reverse complement into paired
  feature channels;
- `RCPSWrapper` / `RCPSMambaBlock`: applies the same transformation to paired
  RC representations;
- `RCPSAddNormWrapper`: preserves the paired-channel action through residual
  normalization;
- `RCPSLMHead`: ties output logits through the token complement map.

`caduceus/modeling_caduceus.py:69` selects RCPS blocks, lines 220-236 select
the RCPS embedding, and lines 604-617 select the RCPS LM head.

## Memory module classification

| Component | Current classification | Evidence and reason |
|---|---|---|
| BCW pooling | A, conditional | `caduceus/memory/writer.py:48-62` uses masked mean pooling. Mean pooling preserves sequence-reversal invariance if its input feature action is handled correctly. |
| BCW directional fusion | C | `BidirectionalMemoryWriter.forward`, lines 64-94, concatenates `(z_fwd, z_bwd, abs difference, product)` into an unconstrained MLP gate. Swapping forward/backward branches does not force `g(b,a)=1-g(a,b)`, so the memory slot is not guaranteed symmetric. |
| Memory slot container | A, conditional | `caduceus/memory_pool.py`, class `MemoryPool`, only stores and concatenates slots. It preserves whatever RC property a slot already has but cannot repair a non-equivariant slot. |
| Layer-slot aggregation | A, conditional | `caduceus/memory_cross_attn.py:30` applies LayerNorm then a mean over memory entries. RC does not reorder network layers, so this operation is safe only if each slot has a valid invariant/equivariant representation. |
| Memory projection | C in current form; B repairable | `memory_cross_attn.py:19-22,30-32` uses an unconstrained `Linear(d_mem, d_model)` and does not enforce the paired-channel symmetry required by RCPS. A paired or explicitly symmetrized projection is a small localized repair. |
| Conservative scalar reader gate | A, conditional | A scalar `0.1 * sigmoid(-4)` commutes with RC, but it cannot repair a non-equivariant projected context. |
| Broadcast over positions | A, conditional | A `[B,1,D]` context is constant along sequence positions, so positional reversal is safe. RCPS channel reversal is safe only if the context vector itself is channel-symmetric. |
| Residual addition | A, conditional | Addition preserves equivariance when both terms transform identically; it propagates a violation when the injected memory term does not. |

Overall classification: **C in the current implementation, with a localized B
repair available**. This conclusion is also encoded directly at
`caduceus/modeling_caduceus.py:309-314`, which rejects strict RCPS plus memory.

## Minimal strict-RC repair proposal (not implemented)

A minimal repair should be isolated behind `rcps=true` so the supported
non-RCPS DNA model and its checkpoints remain unchanged:

1. Build an RC-invariant memory slot by applying the same writer to a state and
   its RC transform and averaging the two slot outputs, or constrain the BCW
   fusion so branch swapping provably leaves the slot unchanged.
2. For RCPS reads, project one half-width context and construct the paired
   output as `[v, flip_channels(v)]`, or explicitly average a proposed update
   with the RC transform of the update computed from the RC input.
3. Keep the existing scalar conservative gate.
4. Add numerical tests of `F(RC(x)) = RC(F(x))` for embedding, every write/read
   boundary, final hidden states, and complemented MLM logits, at both padded
   and unpadded lengths including 10,240.
5. Remove the explicit guard only after those tests pass.

Impact on the supported non-RCPS path: none if the repair is activated only for
`rcps=true`. Strict-RC DNA checkpoints would form a distinct compatibility
class and should not be mixed with the current memory-enabled checkpoints under
strict loading.

## Data augmentation decision

The original fixed-window DNA pilot did not apply random reverse-complement
augmentation. The later teacher-approved OneMaize plan explicitly requires a
training-only 0.5 reverse-complement probability, which the dynamic dataset now
implements without duplicating samples. Validation and test remain in forward
orientation. This does not conflict with a strict-RC architecture because the
supported memory-enabled model has `rcps=false`; augmentation improves strand
exposure but is not presented as proof of architectural equivariance.
