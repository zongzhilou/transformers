# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicIndexedLayer
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_recurrent_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4HyperHead,
    DeepseekV4MLP,
    DeepseekV4RMSNorm,
    DeepseekV4TopKRouter,
    DeepseekV4UnweightedRMSNorm,
    load_balancing_loss_func,
)
from ..glm5_next.modeling_glm5_next import (
    Glm5NextTextForgetGate,
    Glm5NextTextLinearAttention,
    Glm5NextTextModel,
    Glm5NextTextRMSNormGated,
    eager_attention_forward,
)


@auto_docstring(checkpoint="louzongzhi/Shensi-V1-N")
@strict
class ShensiConfig(PreTrainedConfig):
    r"""
    latent_size (`int`, *optional*):
        Width of the routed-expert latent space.
    kv_deepemb_rank (`int`, *optional*):
        Rank of the cached key/value latent.
    index_block_size (`int`, *optional*):
        Number of consecutive tokens averaged into one indexer block.
    index_block_topk (`int`, *optional*):
        Number of top-scoring indexer blocks retained before token refinement.
    index_head_dim (`int`, *optional*):
        Indexer query/key head dimension.
    index_k_size (`int`, *optional*):
        Rank of the cached indexer key latent.
    index_n_heads (`int`, *optional*):
        Number of indexer query heads.
    index_topk_tokens (`int`, *optional*):
        Maximum number of key positions selected per query.
    indexer_types (`list[str]`, *optional*):
        Per-layer indexer mode, either `"full"` or `"shared"`.
    linear_conv_kernel_dim (`int`, *optional*):
        Kernel size of the short convolution in linear attention.
    linear_head_dim (`int`, *optional*):
        Head dimension of linear attention.
    linear_lower_bound (`float`, *optional*):
        Lower bound of the linear-attention forget-gate decay.
    linear_num_heads (`int`, *optional*):
        Number of linear-attention heads.
    first_k_dense_replace (`int`, *optional*):
        Number of initial dense layers before sparse MoE layers.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer feed-forward type, either `"dense"` or `"sparse"`.
    swiglu_limit (`float`, *optional*):
        Clamp limit of the SwiGLU gate and up projections.
    deepemb_rank (`int`, *optional*):
        Rank of the token-conditioned deep-embedding transform in the dense MLP.
    scoring_func (`str`, *optional*):
        Router scoring function.
    hc_active_streams (`int`, *optional*):
        Number of active hyper-connection streams updated by each sublayer.
    hc_adapter_knob (`float`, *optional*):
        Fraction of `hc_latent_size` used as the bottleneck rank of each routed hyper-connection stream adapter.
    hc_latent_size (`int`, *optional*):
        Rank of the basis used to compress hyper-connection streams between layers.
    hc_conv_kernels (`list[int]`, *optional*):
        Kernel sizes of the hyper-connection temporal convolutions.
    hc_eps (`float`, *optional*):
        Numerical floor of hyper-connection normalization.
    hc_fixed_streams (`int`, *optional*):
        Number of hyper-connection streams that are always active.
    hc_mult (`int`, *optional*):
        Total number of hyper-connection streams.
    attn_res_block_layer_types (`list[str]`, *optional*):
        Per-layer attention-residual type, either `"block_write_layer"` or `"block_read_layer"`.
    attn_res_block_size (`int`, *optional*):
        Number of consecutive layers between attention-residual write layers.
    attn_res_latent_size (`int`, *optional*):
        Shared latent width of the block summary, depth-routing keys and gate bottleneck.
    num_attn_res_blocks (`int`, *optional*):
        Number of attention-residual blocks defined by `attn_res_block_layer_types`.
    rosa_embed_dim (`int`, *optional*):
        Dimension of the ROSA token embeddings.
    rosa_layer_ids (`list[int]`, *optional*):
        Decoder layer indices that inject ROSA predictions.
    rosa_reset_token_id (`int`, *optional*):
        Token id that resets the ROSA suffix automaton.
    erc_loss_alpha (`float`, *optional*):
        Specialization threshold of the expert-router coupling loss.
    erc_loss_coef (`float`, *optional*):
        Weight of the expert-router coupling loss.
    """

    model_type = "shensi"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
    }
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }

    # Core
    vocab_size: int = 129280
    hidden_size: int = 1280
    latent_size: int | None = None  # MoE routed-expert width
    num_hidden_layers: int = 19
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = True

    # Attention (MLA)
    num_attention_heads: int = 16
    num_key_value_heads: int = 1
    attention_bias: bool = False
    attention_dropout: float = 0.0
    head_dim: int = 512
    q_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int = 0
    v_head_dim: int | None = None
    layer_types: list[str] | None = None
    kv_deepemb_rank: int | None = None

    # Indexer
    index_block_size: int = 128
    index_block_topk: int = 64
    index_head_dim: int = 128
    index_k_size: int | None = None
    index_n_heads: int = 64
    index_topk_tokens: int = 2048
    indexer_types: list[str] | None = None

    # Linear attention
    linear_conv_kernel_dim: int = 4
    linear_head_dim: int = 128
    linear_lower_bound: float = -5.0
    linear_num_heads: int = 16

    # Feed-forward
    first_k_dense_replace: int = 3
    mlp_bias: bool = False
    mlp_layer_types: list[str] | None = None
    swiglu_limit: float = 10.0
    deepemb_rank: int = 32

    # Mixture-of-Experts (MoE)
    moe_intermediate_size: int | None = None
    n_routed_experts: int = 256
    norm_topk_prob: bool = True
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"

    # Hyper-connections
    hc_active_streams: int = 2
    hc_adapter_knob: float = 0.025
    hc_latent_size: int | None = None
    hc_conv_kernels: list[int] | None = None
    hc_eps: float = 1.0e-6
    hc_fixed_streams: int = 1
    hc_mult: int = 4

    # Attention residual
    attn_res_block_layer_types: list[str] | None = None
    attn_res_block_size: int = 2
    attn_res_latent_size: int | None = None
    num_attn_res_blocks: int | None = None

    # ROSA
    rosa_embed_dim: int | None = None
    rosa_layer_ids: list[int] | None = None
    rosa_reset_token_id: int | None = None

    # Auxiliary losses
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    def __post_init__(self, **kwargs):
        num_layers = self.num_hidden_layers

        # Attention
        if self.num_key_value_heads is None:
            self.num_key_value_heads = 1
        if self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = self.head_dim
        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim
        if self.q_lora_rank is None:
            self.q_lora_rank = self.hidden_size // 4
        if self.kv_deepemb_rank is None:
            self.kv_deepemb_rank = self.head_dim // 8

        # Hyper-connections
        if self.hc_conv_kernels is None:
            self.hc_conv_kernels = [4 * (i + 1) for i in range(self.hc_mult - 1)]
        else:
            self.hc_mult = 1 + len(self.hc_conv_kernels)
        if not 0 <= self.hc_fixed_streams <= self.hc_active_streams <= self.hc_mult:
            raise ValueError(
                "Expected 0 <= hc_fixed_streams <= hc_active_streams <= hc_mult, got "
                f"{self.hc_fixed_streams}, {self.hc_active_streams}, {self.hc_mult}."
            )
        if self.hc_latent_size is None:
            self.hc_latent_size = self.hidden_size // self.hc_mult

        # Widths derived from hidden_size
        if self.latent_size is None:
            self.latent_size = self.hidden_size // 4
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.hidden_size // 2
        if self.attn_res_latent_size is None:
            self.attn_res_latent_size = self.hidden_size // 10
        if self.rosa_embed_dim is None:
            self.rosa_embed_dim = self.hidden_size
        if self.index_k_size is None:
            self.index_k_size = self.index_head_dim // 4

        # Feed-forward schedule: dense prefix followed by sparse MoE layers.
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * self.first_k_dense_replace + ["sparse"] * (
                num_layers - self.first_k_dense_replace
            )

        # Attention-residual schedule: write layers are spaced by attn_res_block_size after the dense prefix.
        num_dense_layers = self.mlp_layer_types.count("dense")
        if self.attn_res_block_layer_types is None:
            self.attn_res_block_layer_types = [
                "block_write_layer"
                if layer_idx == 0
                or (layer_idx >= num_dense_layers and (layer_idx - num_dense_layers) % self.attn_res_block_size == 0)
                else "block_read_layer"
                for layer_idx in range(num_layers)
            ]
        num_write_layers = self.attn_res_block_layer_types.count("block_write_layer")
        if self.num_attn_res_blocks is None:
            self.num_attn_res_blocks = num_write_layers
        elif self.num_attn_res_blocks != num_write_layers:
            raise ValueError(
                f"`num_attn_res_blocks`={self.num_attn_res_blocks} does not match the "
                f"{num_write_layers} block-write layers in `attn_res_block_layer_types`."
            )

        # Per-layer attention types follow the attention-residual write schedule.
        if self.layer_types is None:
            self.layer_types = [
                "indexed_attention" if block_type == "block_write_layer" else "linear_attention"
                for block_type in self.attn_res_block_layer_types
            ]

        # Indexer schedule: either one mode per layer or one mode per indexed layer.
        num_indexed_layers = self.layer_types.count("indexed_attention")
        if self.indexer_types is None:
            self.indexer_types = ["full"] * num_layers
        elif len(self.indexer_types) == num_indexed_layers and len(self.indexer_types) != num_layers:
            indexed_layer_indices = [
                layer_idx for layer_idx, layer_type in enumerate(self.layer_types) if layer_type == "indexed_attention"
            ]
            indexer_types = ["full"] * num_layers
            for position, layer_idx in enumerate(indexed_layer_indices):
                indexer_types[layer_idx] = self.indexer_types[position]
            self.indexer_types = indexer_types

        # ROSA sits at the first and third quartiles of the sparse stack.
        if self.rosa_layer_ids is None:
            num_sparse_layers = num_layers - num_dense_layers
            first_rosa_layer = num_dense_layers + num_sparse_layers // 4
            middle_rosa_layer = num_dense_layers + 3 * num_sparse_layers // 4
            self.rosa_layer_ids = sorted({first_rosa_layer, middle_rosa_layer})

        super().__post_init__(**kwargs)


class ShensiRMSNorm(DeepseekV4RMSNorm):
    pass


class ShensiUnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_square = hidden_states.square().mean(dim=-1, keepdim=True, dtype=torch.float32)
        return hidden_states * torch.rsqrt(mean_square + self.eps).to(hidden_states.dtype)


class ShensiRMSNormGated(Glm5NextTextRMSNormGated):
    pass


class ShensiROSACache:
    """Auxiliary ROSA cache: one suffix automaton per sequence.

    Unlike key/value caches, `update` consumes token ids and returns, for each position, the token that followed the
    longest matching suffix seen before, or -1 when the suffix is new. The cache is croppable and supports the usual
    beam-search batch operations.
    """

    is_croppable = True

    def __init__(self) -> None:
        self.sequence_states: list[dict] = []

    def update(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> torch.LongTensor:
        is_1d = token_ids.dim() == 1
        if is_1d:
            token_ids = token_ids.unsqueeze(0)
        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.shape != token_ids.shape:
                raise ValueError(
                    f"`attention_mask` shape {tuple(attention_mask.shape)} does not match "
                    f"`token_ids` shape {tuple(token_ids.shape)}."
                )
            attention_mask = attention_mask.tolist()

        batch_size, _ = token_ids.shape
        if not self.sequence_states:
            self.sequence_states = [self._new_state() for _ in range(batch_size)]
        elif len(self.sequence_states) != batch_size:
            raise ValueError(
                f"ROSA cache batch size {len(self.sequence_states)} does not match input batch size {batch_size}."
            )

        predictions = []
        for batch_idx, sequence in enumerate(token_ids.tolist()):
            state = self.sequence_states[batch_idx]
            batch_predictions = []
            for position, token in enumerate(sequence):
                if attention_mask is not None and not attention_mask[batch_idx][position]:
                    batch_predictions.append(-1)
                    continue
                batch_predictions.append(self._update_state(state, token, eos_token_id))
            predictions.append(batch_predictions)

        if not predictions:
            return token_ids.new_full(token_ids.shape, -1)

        outputs = torch.tensor(predictions, dtype=torch.long, device=token_ids.device)
        return outputs.squeeze(0) if is_1d else outputs

    def reset(self) -> None:
        """Resets the cache values while preserving the object."""
        self.sequence_states = []

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder the cached sequences for beam search."""
        indices = beam_idx.tolist()
        self.sequence_states = [self.sequence_states[index] for index in indices]

    def crop(self, tokens_to_remove: int) -> None:
        """Remove the last `tokens_to_remove` tokens, given as a negative offset."""
        if tokens_to_remove > 0:
            raise ValueError(f"`crop` expects a negative number of tokens to remove, got {tokens_to_remove}.")
        if tokens_to_remove == 0 or not self.sequence_states:
            return
        for batch_idx, state in enumerate(self.sequence_states):
            tokens = state["tokens"][:tokens_to_remove]
            self.sequence_states[batch_idx] = self._new_state()
            for token in tokens:
                self._update_state(self.sequence_states[batch_idx], token)

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat each cached sequence `repeats` times in the batch dimension."""
        self.sequence_states = [self._clone_state(state) for state in self.sequence_states for _ in range(repeats)]

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Only keep the cached sequences at `indices` in the batch dimension."""
        self.sequence_states = [self._clone_state(self.sequence_states[index]) for index in indices.tolist()]

    @classmethod
    def from_token_ids(
        cls,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> "ShensiROSACache":
        cache = cls()
        cache.update(token_ids, attention_mask=attention_mask, eos_token_id=eos_token_id)
        return cache

    @staticmethod
    def _new_state() -> dict:
        return {
            "transitions": [{}],
            "suffix_links": [-1],
            "lengths": [0],
            "last_positions": [-1],
            "tokens": [],
            "last_state": 0,
            "state_count": 1,
            "position": 0,
        }

    @staticmethod
    def _grow_state(state: dict, index: int) -> None:
        transitions = state["transitions"]
        while len(transitions) <= index:
            transitions.append({})
            state["suffix_links"].append(-1)
            state["lengths"].append(0)
            state["last_positions"].append(-1)

    @staticmethod
    def _update_state(state: dict, token: int, eos_token_id: int | None = None) -> int:
        transitions = state["transitions"]
        suffix_links = state["suffix_links"]
        lengths = state["lengths"]
        last_positions = state["last_positions"]
        tokens = state["tokens"]

        tokens.append(token)
        current_state = state["state_count"]
        state["state_count"] = current_state + 1
        ShensiROSACache._grow_state(state, current_state)
        lengths[current_state] = lengths[state["last_state"]] + 1
        index = state["last_state"]

        while index != -1:
            next_state = transitions[index].get(token)
            if next_state is not None:
                break
            transitions[index][token] = current_state
            index = suffix_links[index]

        if index == -1:
            suffix_links[current_state] = 0
        else:
            candidate = next_state
            if lengths[index] + 1 == lengths[candidate]:
                suffix_links[current_state] = candidate
            else:
                clone_state = state["state_count"]
                state["state_count"] = clone_state + 1
                ShensiROSACache._grow_state(state, clone_state)
                transitions[clone_state] = transitions[candidate].copy()
                lengths[clone_state] = lengths[index] + 1
                suffix_links[clone_state] = suffix_links[candidate]
                last_positions[clone_state] = last_positions[candidate]

                while index != -1 and transitions[index].get(token) == candidate:
                    transitions[index][token] = clone_state
                    index = suffix_links[index]

                suffix_links[candidate] = clone_state
                suffix_links[current_state] = clone_state

        index = state["last_state"] = current_state
        prediction = -1
        while index != -1:
            if lengths[index] > 0 and last_positions[index] >= 0:
                prediction = tokens[last_positions[index] + 1]
                break
            index = suffix_links[index]

        index = state["last_state"]
        position = state["position"]
        while index != -1 and last_positions[index] < position:
            last_positions[index] = position
            index = suffix_links[index]

        state["position"] = position + 1
        if eos_token_id is not None and token == eos_token_id:
            state.clear()
            state.update(ShensiROSACache._new_state())
        return prediction

    @staticmethod
    def _clone_state(state: dict) -> dict:
        return {
            "transitions": [transitions.copy() for transitions in state["transitions"]],
            "suffix_links": list(state["suffix_links"]),
            "lengths": list(state["lengths"]),
            "last_positions": list(state["last_positions"]),
            "tokens": list(state["tokens"]),
            "last_state": state["last_state"],
            "state_count": state["state_count"],
            "position": state["position"],
        }


class ShensiAttentionCache(DynamicIndexedLayer):
    """Per-layer cache for indexed attention.

    Stores the low-rank K/V latent (FP8 outside training), the preallocated indexer key buffer, and the token ids
    the indexer needs. The first indexed layer also owns the shared `ShensiROSACache`; every other indexed layer
    reads the owner's token ids through `source`, so batch operations run once.
    """

    _layer_type = "indexed_attention"
    is_croppable = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Non-owner layers reuse the owner's token ids.
        self.source = None
        self.token_ids: torch.Tensor | None = None
        self.rosa_cache: ShensiROSACache | None = None
        self.indexer_buffer: torch.Tensor | None = None
        self.indexer_length = 0

    def update_indexer(self, indexer_key_states: torch.Tensor) -> torch.Tensor:
        # The indexer buffer is preallocated and grown geometrically.
        batch_size, seq_len, dim = indexer_key_states.shape
        required = self.indexer_length + seq_len
        self._ensure_indexer_capacity(batch_size, required, dim, indexer_key_states.dtype, indexer_key_states.device)
        self.indexer_buffer[:batch_size, self.indexer_length : required] = indexer_key_states
        return self._commit_indexer(batch_size, required)

    def update(self, key_states, value_states, *args, **kwargs):
        # Cache one latent; FP8 outside training, compute dtype in training.
        if torch.is_grad_enabled():
            stored = key_states
        else:
            stored = key_states.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        if not self.is_initialized:
            self.lazy_initialization(stored, stored)
        self.keys = torch.cat([self.keys, stored], dim=-2)
        self.values = self.keys
        keys = self.keys.to(key_states.dtype)
        return keys, keys

    def reset(self) -> None:
        super().reset()
        self.token_ids = None
        self.rosa_cache = None
        self.indexer_buffer = None
        self.indexer_length = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.rosa_cache is not None:
            self.rosa_cache.reorder_cache(beam_idx)
        if self.indexer_buffer is not None:
            self.indexer_buffer = self.indexer_buffer.index_select(0, beam_idx.to(self.indexer_buffer.device))
            self.indexer_keys = self.indexer_buffer[:, : self.indexer_length]
        if self._sync_token_ids():
            return
        if self.token_ids is not None:
            self.token_ids = self.token_ids.index_select(0, beam_idx.to(self.token_ids.device))

    def crop(self, tokens_to_remove: int) -> None:
        super().crop(tokens_to_remove)
        if tokens_to_remove == 0:
            return
        if self.rosa_cache is not None:
            self.rosa_cache.crop(tokens_to_remove)
        if self.indexer_buffer is not None:
            self.indexer_length = max(0, self.indexer_length - abs(tokens_to_remove))
            self.indexer_keys = self.indexer_buffer[:, : self.indexer_length]
        if self._sync_token_ids():
            return
        if self.token_ids is None:
            return
        self.token_ids = self.token_ids[:, : -abs(tokens_to_remove)]

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.rosa_cache is not None:
            self.rosa_cache.batch_repeat_interleave(repeats)
        if self.indexer_buffer is not None:
            self.indexer_buffer = self.indexer_buffer.repeat_interleave(repeats, dim=0)
            self.indexer_keys = self.indexer_buffer[:, : self.indexer_length]
        if self._sync_token_ids():
            return
        if self.token_ids is not None:
            self.token_ids = self.token_ids.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.rosa_cache is not None:
            self.rosa_cache.batch_select_indices(indices)
        if self.indexer_buffer is not None:
            self.indexer_buffer = self.indexer_buffer[indices, ...]
            self.indexer_keys = self.indexer_buffer[:, : self.indexer_length]
        if self._sync_token_ids():
            return
        if self.token_ids is not None:
            self.token_ids = self.token_ids[indices, ...]

    def offload(self) -> None:
        super().offload()
        if self.indexer_buffer is not None:
            self.indexer_buffer = self.indexer_buffer.to("cpu", non_blocking=True)
            self.indexer_keys = self.indexer_buffer[:, : self.indexer_length]

    def prefetch(self) -> None:
        super().prefetch()
        if self.indexer_buffer is not None and self.indexer_buffer.device != self.device:
            self.indexer_buffer = self.indexer_buffer.to(self.device, non_blocking=True)
            self.indexer_keys = self.indexer_buffer[:, : self.indexer_length]

    def update_indexer_split(self, indexer_latents: torch.Tensor, valid_channel: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, latent_dim = indexer_latents.shape
        required = self.indexer_length + seq_len
        self._ensure_indexer_capacity(
            batch_size, required, latent_dim + 1, indexer_latents.dtype, indexer_latents.device
        )
        self.indexer_buffer[:batch_size, self.indexer_length : required, :latent_dim] = indexer_latents
        self.indexer_buffer[:batch_size, self.indexer_length : required, latent_dim] = valid_channel.squeeze(-1)
        return self._commit_indexer(batch_size, required)

    def update_token_ids(self, token_ids: torch.Tensor | None) -> torch.Tensor | None:
        if self._sync_token_ids():
            return self.token_ids
        if token_ids is None:
            self.token_ids = None
        elif self.token_ids is None:
            self.token_ids = token_ids
        else:
            self.token_ids = torch.cat([self.token_ids, token_ids], dim=1)
        return self.token_ids

    def _ensure_indexer_capacity(
        self, batch_size: int, required: int, dim: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        if self.indexer_buffer is None:
            capacity = max(required, 2048)
            self.indexer_buffer = torch.empty((batch_size, capacity, dim), dtype=dtype, device=device)
        else:
            capacity = self.indexer_buffer.shape[1]
            if required > capacity:
                capacity = max(required, capacity * 2)
            batch_capacity = max(batch_size, self.indexer_buffer.shape[0])
            if capacity > self.indexer_buffer.shape[1] or batch_capacity > self.indexer_buffer.shape[0]:
                buffer = torch.empty((batch_capacity, capacity, dim), dtype=dtype, device=device)
                buffer[: self.indexer_buffer.shape[0], : self.indexer_length] = self.indexer_buffer[
                    :, : self.indexer_length
                ]
                self.indexer_buffer = buffer

    def _commit_indexer(self, batch_size: int, required: int) -> torch.Tensor:
        self.indexer_length = required
        self.indexer_keys = self.indexer_buffer[:batch_size, :required]
        self.is_indexer_initialized = True
        return self.indexer_keys

    def _sync_token_ids(self) -> bool:
        if self.source is not None:
            self.token_ids = self.source.token_ids
            return True
        return False


class ShensiIndexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) indexer with a low-rank key cache and hierarchical block
    selection for Shensi-V1.

    The indexer uses lightweight query/key projections separate from the main attention path. Keys
    are cached as low-rank latents, reconstructed through a DeepEmbed-style up-projection and token
    embedding, and scored hierarchically: blocks are ranked first, then the tokens inside the
    retained blocks are rescored at full resolution.

    **Cache strategy**: the indexer state cache lives on the per-layer `DynamicIndexedLayer`
    (or the `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `past_key_values.update_indexer()`.
    """

    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.q_lora_rank: int = config.q_lora_rank
        self.index_k_size: int = config.index_k_size
        self.index_block_size: int = config.index_block_size
        self.index_block_topk: int = config.index_block_topk
        self.index_topk_tokens: int = config.index_topk_tokens

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.k = nn.Linear(self.hidden_size, self.index_k_size, bias=False)
        self.k_up = nn.Linear(self.index_k_size, self.head_dim, bias=False)
        self.k_emb = (
            nn.Embedding(config.vocab_size, self.head_dim) if layer_idx == config.indexer_types.index("full") else None
        )
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.BoolTensor,
        past_key_values: Cache | None,
        token_ids: torch.LongTensor | None = None,
    ) -> torch.LongTensor:
        """
        Selects the top-k key positions per query for hierarchical sparse attention.

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            attention_mask: Local boolean padding mask of shape `[B, S]`.
            past_key_values: Cache object containing the indexer state cache for this layer.
            token_ids: Token ids aligned with the cached index keys, shape `[B, T]`. These may include
                past tokens during generation and are used by the shared token embedding.

        Returns:
            `torch.LongTensor`: the `int32` top-k key indices of shape `[B, S, topk_tokens]`. The
            eager / SDPA paths turn these into an additive sparse mask.
        """
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid).view(hidden_shape) * self.softmax_scale
        k = self.k(hidden_states)

        valid_channel = attention_mask.to(k.dtype)[..., None]

        packed_states = torch.cat([k, valid_channel], dim=-1)

        kv_len = seq_len
        current_length = seq_len
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            if cache_layer.source is not None and cache_layer.source.indexer_keys is not None:
                packed_states = cache_layer.source.indexer_keys
            else:
                latent = k.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                valid = valid_channel.to(torch.float8_e4m3fn)
                if hasattr(cache_layer, "update_indexer_split"):
                    packed_states = cache_layer.update_indexer_split(latent, valid)
                else:
                    packed_states = past_key_values.update_indexer(torch.cat([latent, valid], dim=-1), self.layer_idx)
            kv_len = packed_states.shape[-2]
            current_length = kv_len

        valid_keys = packed_states[..., -1].bool()

        keys = packed_states[..., : self.index_k_size].to(self.k_up.weight.dtype)
        keys = self.k_up(keys)
        if self.k_emb is not None and token_ids is not None and token_ids.shape[1] >= keys.shape[1]:
            keys = keys * self.k_emb(token_ids[:, -keys.shape[1] :])
        keys = self.k_norm(keys)

        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)

        any_valid = valid_keys.any(dim=-1)
        first_valid = torch.where(
            any_valid,
            valid_keys.to(torch.int64).argmax(dim=-1),
            torch.zeros(batch_size, dtype=torch.int64, device=hidden_states.device),
        )
        last_valid = torch.where(
            any_valid,
            kv_len - 1 - valid_keys.flip(-1).to(torch.int64).argmax(dim=-1),
            torch.full_like(first_valid, -1),
        )
        q_positions = torch.arange(seq_len, device=hidden_states.device) + (current_length - seq_len)
        query_end = torch.minimum(q_positions.unsqueeze(0) + 1, last_valid.unsqueeze(-1) + 1)
        query_start = first_valid.unsqueeze(-1).expand(-1, seq_len)
        valid_query = attention_mask.bool()
        query_start = torch.where(valid_query, query_start, torch.zeros_like(query_start))
        query_end = torch.where(valid_query, query_end, torch.zeros_like(query_end))

        blocked_seq_len, blocked_head_dim = keys.shape[-2:]
        num_blocks = (blocked_seq_len + self.index_block_size - 1) // self.index_block_size
        counts = torch.full(
            (*keys.shape[:-2], num_blocks), self.index_block_size, device=keys.device, dtype=torch.float32
        )
        padded_len = num_blocks * self.index_block_size
        blocked_k = keys
        if padded_len != blocked_seq_len:
            blocked_k = F.pad(blocked_k, (0, 0, 0, padded_len - blocked_seq_len))
            counts[..., -1] = blocked_seq_len - (num_blocks - 1) * self.index_block_size
        blocked_k = blocked_k.reshape(*blocked_k.shape[:-2], num_blocks, self.index_block_size, blocked_head_dim).sum(
            dim=-2, dtype=torch.float32
        )
        blocked_k = blocked_k / counts.unsqueeze(-1)

        q_blocks = q.transpose(-3, -2)
        blocked_k = blocked_k.to(q.dtype)
        block_scores = torch.matmul(q_blocks, blocked_k.transpose(-1, -2).unsqueeze(-3)).relu_()
        block_k_score = torch.einsum("bhsp,bhs->bsp", block_scores.float(), weights.transpose(-1, -2))

        cu_seqlen_blocked_ks = query_start // self.index_block_size
        cu_seqlen_blocked_ke = (query_end + self.index_block_size - 1) // self.index_block_size
        block_positions = torch.arange(num_blocks, device=block_k_score.device)
        first_block = cu_seqlen_blocked_ks.clamp(0, num_blocks - 1).long().unsqueeze(-1)
        last_block = (cu_seqlen_blocked_ke - 1).clamp(0, num_blocks - 1).long().unsqueeze(-1)
        block_k_score.scatter_(-1, first_block, float("inf"))
        block_k_score.scatter_(-1, last_block, float("inf"))
        block_k_score.masked_fill_(
            (block_positions < cu_seqlen_blocked_ks.unsqueeze(-1))
            | (block_positions >= cu_seqlen_blocked_ke.unsqueeze(-1)),
            float("-inf"),
        )
        block_topk_eff = min(self.index_block_topk, block_k_score.shape[-1])
        topk_block_indices = block_k_score.bfloat16().topk(block_topk_eff, dim=-1, sorted=False).indices

        num_kv = keys.shape[-2]
        pos_in_block = torch.arange(self.index_block_size, device=q.device)
        batch_offsets = torch.arange(batch_size, device=keys.device).view(-1, 1, 1, 1) * num_kv
        keys_flat = keys.reshape(batch_size * num_kv, self.head_dim)
        token_weights = weights.to(q.dtype)

        if seq_len == 1:
            k_abs = topk_block_indices.unsqueeze(-1) * self.index_block_size + pos_in_block
            flat_indices = (k_abs.clamp(0, num_kv - 1) + batch_offsets).reshape(-1)
            gathered_k = keys_flat[flat_indices].reshape(
                batch_size, seq_len, block_topk_eff * self.index_block_size, self.head_dim
            )
            token_scores = torch.matmul(q, gathered_k.transpose(-1, -2)).relu_()
            block_sparse_logits = torch.einsum("bsh,bshk->bsk", token_weights, token_scores)
            in_range = (
                (k_abs >= query_start.unsqueeze(-1).unsqueeze(-1))
                & (k_abs < query_end.unsqueeze(-1).unsqueeze(-1))
                & (k_abs < num_kv)
            )
            block_sparse_logits.masked_fill_(~in_range.flatten(-2), float("-inf"))
        else:
            block_chunk = 2 if block_topk_eff >= 32 and batch_size * seq_len <= 8192 else 1
            block_sparse_logits = q.new_empty(
                (batch_size, seq_len, block_topk_eff * self.index_block_size), dtype=q.dtype
            )
            for block_start in range(0, block_topk_eff, block_chunk):
                block_end = min(block_start + block_chunk, block_topk_eff)
                block_ids = topk_block_indices[..., block_start:block_end]
                k_abs = block_ids.unsqueeze(-1) * self.index_block_size + pos_in_block
                flat_indices = (k_abs.clamp(0, num_kv - 1) + batch_offsets).reshape(-1)
                gathered_k = keys_flat[flat_indices].reshape(
                    batch_size, seq_len, (block_end - block_start) * self.index_block_size, self.head_dim
                )
                token_scores = torch.matmul(q, gathered_k.transpose(-1, -2)).relu_()
                logits = torch.einsum("bsh,bshk->bsk", token_weights, token_scores)
                in_range = (
                    (k_abs >= query_start.unsqueeze(-1).unsqueeze(-1))
                    & (k_abs < query_end.unsqueeze(-1).unsqueeze(-1))
                    & (k_abs < num_kv)
                )
                logits.masked_fill_(~in_range.flatten(-2), float("-inf"))
                block_sparse_logits[..., block_start * self.index_block_size : block_end * self.index_block_size] = (
                    logits
                )

        topk_tokens_eff = min(self.index_topk_tokens, block_sparse_logits.shape[-1])
        relevant_topk_indices = block_sparse_logits.topk(topk_tokens_eff, dim=-1, sorted=False).indices

        absolute_topk_block_indices = topk_block_indices.gather(-1, relevant_topk_indices // self.index_block_size)
        topk_indices = absolute_topk_block_indices * self.index_block_size + (
            relevant_topk_indices % self.index_block_size
        )
        topk_indices = topk_indices.to(torch.int32)
        topk_indices -= query_start.unsqueeze(-1).to(torch.int32)
        mask_lo = topk_indices >= 0
        mask_hi = topk_indices - (query_end - query_start).unsqueeze(-1).to(torch.int32) < 0
        topk_indices.masked_fill_(~(mask_lo & mask_hi), -1)

        topk_indices = torch.where(
            topk_indices >= 0, topk_indices + query_start.unsqueeze(-1).to(torch.int32), topk_indices
        )
        selected_valid = (
            valid_keys.unsqueeze(1).expand(-1, seq_len, -1).gather(2, topk_indices.clamp(0, keys.shape[1] - 1))
        )
        topk_indices = topk_indices.masked_fill((topk_indices >= 0) & ~selected_valid, -1)

        self_positions = (current_length - seq_len + torch.arange(seq_len, device=hidden_states.device)).to(
            torch.int32
        )
        self_positions = self_positions.unsqueeze(0).expand(batch_size, -1)
        self_valid = (
            attention_mask.bool()
            & (self_positions < kv_len)
            & valid_keys.gather(1, self_positions.clamp(0, kv_len - 1).long())
        )
        self_positions = self_positions.unsqueeze(-1)
        has_self = topk_indices.eq(self_positions).any(dim=-1, keepdim=True)
        replace = self_valid.unsqueeze(-1) & ~has_self
        last_token = topk_indices[..., -1:]
        last_token = torch.where(replace, self_positions, last_token)
        topk_indices = torch.cat((topk_indices[..., :-1], last_token), dim=-1)

        output_width = self.index_topk_tokens
        topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
        topk_indices = topk_indices[..., :output_width]
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        return topk_indices.to(torch.int32)


class ShensiAttention(nn.Module):
    """
    Gated DeepSeek Sparse Attention extended with **self-position exclusion** for Shensi-V1.

    `config.indexer_types[layer_idx]` decides whether this layer runs its own indexer (`"full"`) or
    reuses the previous full layer's top-k selection (`"shared"`). `next_skip_topk` signals that the
    next layer will reuse this layer's top-k, so it is propagated upward via `prev_topk_indices`.
    """

    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.is_causal = True
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = ShensiRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * (self.qk_head_dim + self.v_head_dim), bias=False)

        self.kv = nn.Linear(self.hidden_size, config.kv_deepemb_rank, bias=False)
        self.kv_up = nn.Linear(config.kv_deepemb_rank, self.v_head_dim, bias=False)
        self.kv_emb = (
            nn.Embedding(config.vocab_size, config.kv_deepemb_rank)
            if layer_idx == config.layer_types.index("indexed_attention")
            else None
        )
        self.kv_norm = ShensiRMSNorm(self.v_head_dim, eps=config.rms_norm_eps)

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.scaling = self.qk_head_dim ** (-0.5)
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if self.skip_topk else ShensiIndexer(config, layer_idx)
        self.next_skip_topk = (
            not self.skip_topk and config.indexer_types[min(layer_idx + 1, len(config.indexer_types) - 1)] == "shared"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size, seq_len = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_len, -1, self.qk_head_dim)

        q_residual = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q, gate = torch.chunk(
            self.q_b_proj(q_residual).view(batch_size, seq_len, self.num_heads, self.qk_head_dim + self.v_head_dim),
            2,
            dim=-1,
        )
        gate = gate.reshape(batch_size, seq_len, -1)
        q = q.view(query_shape).transpose(1, 2)

        cache_layer = past_key_values.layers[self.layer_idx] if past_key_values is not None else None

        shared_kv = cache_layer is not None and cache_layer.source is not None and cache_layer.source.keys is not None
        if shared_kv:
            kv_latents = cache_layer.source.keys.squeeze(1).to(self.kv_up.weight.dtype)
        else:
            kv_latents = self.kv(hidden_states)
            if cache_layer is not None:
                cached_latents, _ = past_key_values.update(
                    kv_latents.unsqueeze(1), kv_latents.unsqueeze(1), self.layer_idx
                )
                kv_latents = cached_latents.squeeze(1)

        if cache_layer is not None:
            token_ids = cache_layer.update_token_ids(input_ids)
        else:
            token_ids = input_ids

        current_length = kv_latents.shape[1]

        if token_ids is not None and self.kv_emb is not None and token_ids.shape[1] >= kv_latents.shape[1]:
            kv_latents = kv_latents + self.kv_emb(token_ids[:, -kv_latents.shape[1] :])
        kv = self.kv_norm(self.kv_up(kv_latents)).unsqueeze(1)

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states=hidden_states,
                q_resid=q_residual,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                token_ids=token_ids,
            )
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        attention_mask = self.build_attention_mask_from_topk(
            topk_indices=topk_indices,
            query_states=q,
            kv_length=kv.shape[2],
        )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        positions = current_length - seq_len + torch.arange(seq_len, device=hidden_states.device)
        self_values = kv[:, 0].index_select(1, positions).unsqueeze(2)
        self_values = F.normalize(self_values, dim=-1)
        attn_output = attn_output - (attn_output * self_values).sum(dim=-1, keepdim=True) * self_values

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        # A shared layer must keep forwarding the indices it reused; otherwise a chain of two
        # shared layers loses them and the second one has nothing to reuse.
        return attn_output, attn_weights, topk_indices if (self.next_skip_topk or self.skip_topk) else None

    def build_attention_mask_from_topk(
        self,
        topk_indices: torch.Tensor,
        query_states: torch.Tensor,
        kv_length: int,
    ) -> torch.Tensor | None:
        """
        Convert topk_indices into the mask expected by the active backend.

        Only supporting SDPA and Eager as we have a 3D dependency which cannot be mapped to FA
        without a custom kernel that can select on a per indices bases per row (query -> topk keys).
        """
        # -1 is invalid as per convention in the indexer
        # NOTE: The indexer already took care of also excluding padding tokens and causality
        topk_valid = topk_indices.ge(0) & topk_indices.lt(kv_length)

        # Clamp only so scatter has a legal index
        safe_indices = topk_indices.clamp(0, kv_length - 1)
        selected_counts = torch.zeros(
            topk_indices.shape[0],  # batch size
            topk_indices.shape[1],  # q_length
            kv_length,  # kv_length
            dtype=torch.int32,
            device=topk_indices.device,
        )
        selected_counts.scatter_add_(-1, safe_indices, topk_valid.to(torch.int32))

        # Final mask 0 == False (not visible), 1 == True (visible)
        mask = selected_counts.ne(0).unsqueeze(1)

        # SDPA
        # trf-ignore: TRF051
        if self.config._attn_implementation == "sdpa":
            return mask

        # Eager
        min_dtype = torch.finfo(query_states.dtype).min
        # we need 0s where the tokens should be taken into account, and -inf otherwise (mask is already of boolean type)
        mask = torch.where(mask, torch.full((), 0.0, device=query_states.device, dtype=query_states.dtype), min_dtype)
        return mask


class ShensiForgetGate(Glm5NextTextForgetGate):
    def __init__(self, config: ShensiConfig):
        super().__init__()


class ShensiLinearAttention(Glm5NextTextLinearAttention):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.forget_gate = ShensiForgetGate(config)
        self.o_norm = ShensiRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)


class ShensiMLP(DeepseekV4MLP):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.act_fn = ACT2FN["relu2"]
        self.token_shift = nn.Parameter(torch.zeros(config.hidden_size))
        self.deepemb_rank = config.deepemb_rank
        self.deepemb = (
            nn.Embedding(config.vocab_size, self.deepemb_rank**2)
            if layer_idx == config.mlp_layer_types.index("dense")
            else None
        )
        self.deepemb_a_proj = nn.Linear(config.hidden_size, self.deepemb_rank, bias=False)
        self.deepemb_b_proj = nn.Linear(self.deepemb_rank, self.intermediate_size, bias=True)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shifted = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        x = x + (shifted - x) * self.token_shift.to(x.dtype)
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        deepemb_states = self.deepemb_a_proj(x)
        deepemb_matrices = self.deepemb(input_ids).view(*input_ids.shape, self.deepemb_rank, self.deepemb_rank)
        deepemb_states = torch.einsum("...r,...rs->...s", deepemb_states, deepemb_matrices)
        deepemb_scale = self.deepemb_b_proj(deepemb_states)
        return self.down_proj((self.act_fn(gate) * up) * deepemb_scale)


class ShensiTopKRouter(DeepseekV4TopKRouter):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        del self.e_score_correction_bias
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(scores, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class ShensiExperts(DeepseekV4Experts):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.hidden_dim = config.latent_size


class ShensiSparseMoeBlock(nn.Module):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.is_block_write_layer = config.attn_res_block_layer_types[layer_idx] == "block_write_layer"
        self.gate = ShensiTopKRouter(config) if self.is_block_write_layer else None
        self.experts = ShensiExperts(config) if self.is_block_write_layer else None
        self.routed_expert_down_proj = nn.Linear(config.hidden_size, config.latent_size, config.mlp_bias)
        self.routed_expert_norm = ShensiRMSNorm(config.latent_size, config.rms_norm_eps)
        self.routed_expert_up_proj = nn.Linear(config.latent_size, config.hidden_size, config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _, weights, indices = self.gate(hidden_states)
        routed = self.experts(self.routed_expert_down_proj(flat), indices, weights)
        return self.routed_expert_up_proj(self.routed_expert_norm(routed)).view(batch, seq_len, hidden_dim)


class ShensiHCState(NamedTuple):
    """State carried from ShensiHyperConnection.forward to write_back."""

    residual: torch.Tensor
    mixed: torch.Tensor
    post: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class ShensiHyperConnection(nn.Module):
    def __init__(self, config: ShensiConfig, is_mlp: bool):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_active_streams = config.hc_active_streams
        self.hc_fixed_streams = config.hc_fixed_streams
        self.hc_latent_size = config.hc_latent_size
        self.hc_adapter_knob = config.hc_adapter_knob
        self.num_active_routed_streams = config.hc_active_streams - config.hc_fixed_streams

        num_routed_streams = config.hc_mult - config.hc_fixed_streams
        kernels = config.hc_conv_kernels if is_mlp else ()
        self.num_features = 1 + len(kernels)
        self.num_route_coefficients = num_routed_streams if self.hc_active_streams < self.hc_mult else 0
        self.num_post_coefficients = self.hc_active_streams * self.num_features
        num_coefficients = self.hc_mult + self.num_route_coefficients + self.num_post_coefficients
        self.rms_norm_eps = config.rms_norm_eps

        self.fn = nn.Parameter(torch.empty(num_coefficients, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(num_coefficients))
        self.scale = nn.Parameter(torch.empty(3))

        self.down = (
            nn.Parameter(
                torch.empty(
                    num_routed_streams, math.ceil(self.hc_latent_size * self.hc_adapter_knob), config.hidden_size
                )
            )
            if self.num_route_coefficients > 0
            else None
        )
        self.up = (
            nn.Parameter(
                torch.empty(
                    num_routed_streams, config.hidden_size, math.ceil(self.hc_latent_size * self.hc_adapter_knob)
                )
            )
            if self.num_route_coefficients > 0
            else None
        )
        self.decay = nn.Parameter(torch.empty(num_routed_streams)) if self.num_route_coefficients > 0 else None
        self.convs = (
            nn.ModuleList(
                [
                    nn.Conv1d(
                        config.hidden_size, config.hidden_size, kernel, groups=config.hidden_size, padding=kernel - 1
                    )
                    for kernel in kernels
                ]
            )
            if kernels
            else None
        )

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, ShensiHCState]:
        batch, seq_len, _, hidden_size = streams.shape
        flat = F.rms_norm(streams.flatten(2).float(), (self.hc_mult * hidden_size,), None, self.rms_norm_eps)
        coefficients = F.linear(flat, self.fn.float())

        split_sizes = [self.hc_mult, self.num_route_coefficients, self.num_post_coefficients]
        pre_w, route_w, post_w = coefficients.split(split_sizes, dim=-1)
        pre_b, route_b, post_b = self.base.split(split_sizes, dim=-1)
        pre_scale, route_scale, post_scale = self.scale.unbind(0)

        pre = torch.softmax(torch.addcmul(pre_b, pre_w, pre_scale), dim=-1).to(streams.dtype)
        collapsed = torch.matmul(pre.unsqueeze(-2), streams).squeeze(-2)

        if self.num_route_coefficients > 0:
            route_w = torch.addcmul(route_b, route_w, route_scale)
            scores, top = route_w.topk(self.num_active_routed_streams, dim=-1)
            fixed = torch.arange(self.hc_fixed_streams, device=streams.device).expand(batch, seq_len, -1)
            indices = torch.cat([fixed, top + self.hc_fixed_streams], dim=-1)
            route_weights = torch.cat(
                [scores.new_ones(self.hc_fixed_streams).expand(batch, seq_len, -1), scores.sigmoid()], dim=-1
            )
        else:
            indices = torch.arange(self.hc_active_streams, device=streams.device).expand(batch, seq_len, -1)
            route_weights = streams.new_ones(self.hc_active_streams).expand(batch, seq_len, -1)

        active = streams.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, hidden_size))
        if self.decay is not None:
            decay = torch.sigmoid(-self.decay.to(active.dtype))
            if self.hc_fixed_streams > 0:
                decay = torch.cat([decay.new_ones(self.hc_fixed_streams), decay])
            active = active * decay[indices].unsqueeze(-1)

        post = (torch.sigmoid(torch.addcmul(post_b, post_w, post_scale)) * 2.0).view(
            batch, seq_len, self.hc_active_streams, self.num_features
        )
        return collapsed, ShensiHCState(
            residual=streams,
            mixed=active,
            post=post.to(streams.dtype),
            indices=indices,
            weights=route_weights.to(streams.dtype),
        )

    def write_back(self, output: torch.Tensor, state: ShensiHCState, residual: torch.Tensor) -> torch.Tensor:
        if self.convs is None:
            addition = state.post * output.unsqueeze(2)
        else:
            conv_input = output.transpose(1, 2).to(self.convs[0].weight.dtype, memory_format=torch.contiguous_format)
            branches = [output]
            for conv in self.convs:
                branches.append(conv(conv_input)[..., : output.shape[1]].transpose(1, 2).to(output.dtype))
            addition = torch.matmul(state.post, torch.stack(branches, dim=2))

        if self.down is not None:
            for slot in range(self.hc_fixed_streams, self.hc_active_streams):
                selected = state.indices[:, :, slot] - self.hc_fixed_streams
                slot_update = None
                for expert_idx in range(self.down.shape[0]):
                    candidate = F.linear(F.linear(output, self.down[expert_idx]), self.up[expert_idx])
                    slot_update = (
                        candidate
                        if slot_update is None
                        else torch.where((selected == expert_idx).unsqueeze(-1), candidate, slot_update)
                    )
                addition[:, :, slot, :] += slot_update * state.weights[..., slot].unsqueeze(-1)

        updated = state.mixed + addition
        residual_streams = state.residual
        if self.hc_fixed_streams > 0:
            updated[:, :, : self.hc_fixed_streams, :] += residual.unsqueeze(2)
        else:
            residual_streams = residual_streams + residual.unsqueeze(2)
        updated = updated.to(residual_streams.dtype)
        return residual_streams.scatter(2, state.indices.unsqueeze(-1).expand_as(updated), updated)


class ShensiHyperHead(DeepseekV4HyperHead):
    pass


class ShensiAttentionResidual(nn.Module):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm = ShensiUnweightedRMSNorm(config.rms_norm_eps)
        self.g_a_proj = (
            nn.Linear(config.hidden_size, config.attn_res_latent_size, bias=False)
            if layer_idx < config.num_hidden_layers
            else None
        )
        self.g_b_proj = (
            nn.Linear(config.attn_res_latent_size, 3 * config.hidden_size, bias=True)
            if layer_idx < config.num_hidden_layers
            else None
        )
        self.q_proj = nn.Linear(config.hidden_size, config.attn_res_latent_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.attn_res_latent_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.num_attn_res_blocks + 2))
        self.weight = nn.Parameter(torch.empty(2))

    def forward(
        self,
        prefix_sum: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        num_blocks: int,
    ) -> torch.Tensor:
        config = self.config
        out_dtype = prefix_sum.dtype
        is_output = self.layer_idx >= config.num_hidden_layers
        hidden_size = config.hidden_size
        max_blocks = config.num_attn_res_blocks + 1

        prefix_sum = prefix_sum.float()
        delta = delta.float() if delta is not None else None

        head = blocks[..., 0, :]
        bank = blocks[..., :max_blocks, :]
        deltas = bank[..., 1:, :]
        boundaries = (head.float().unsqueeze(-2) + deltas.float().cumsum(dim=-2)).to(bank.dtype)

        if is_output:
            current = delta
        else:
            # One fp32 reduction over the visible deltas: the bf16 boundary would lose precision as depth grows
            committed = deltas[..., : num_blocks - 1, :].float().sum(dim=-2) if num_blocks > 1 else 0.0
            global_state = head.float() + committed + prefix_sum
            current = global_state

        sources = torch.cat(
            [head.unsqueeze(-2).to(bank.dtype), boundaries, current.unsqueeze(-2).to(bank.dtype)], dim=-2
        )
        valid = torch.ones(max_blocks + 1, dtype=torch.bool, device=blocks.device)
        valid[1:max_blocks] = torch.arange(max_blocks - 1, device=blocks.device) < (num_blocks - 1)

        source_state = self.norm(sources)
        query = self.q_proj(self.norm(current).to(self.q_proj.weight.dtype)).float()
        keys = F.normalize(self.k_proj(source_state.to(self.k_proj.weight.dtype)).float(), dim=-1)
        logit_scale, residual_gate = self.weight.float().unbind(0)
        logits = torch.einsum("...r,...nr->...n", query, keys) * logit_scale
        logits = logits + self.bias[: sources.shape[-2]].float()
        logits = logits.masked_fill(~valid, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        mixed = torch.einsum("...n,...nd->...d", weights.to(sources.dtype), sources).float()

        if is_output:
            output = current + torch.tanh(residual_gate) * (mixed - current)
            current_rms = (current.square().mean(dim=-1, keepdim=True) + self.norm.eps).sqrt()
            output_rms = (output.square().mean(dim=-1, keepdim=True) + self.norm.eps).sqrt()
            output = output * (current_rms / output_rms).detach().clamp(max=10.0)
            return (output - current).to(out_dtype)

        anchor = global_state + delta if delta is not None else global_state
        gate_state = self.norm(delta) if delta is not None else self.norm(global_state)
        gates = self.g_b_proj(F.gelu(self.g_a_proj(gate_state.to(self.g_a_proj.weight.dtype))))
        decay, erase, write_gate = gates.float().split(hidden_size, dim=-1)

        carried = torch.sigmoid(decay) * global_state
        if delta is None:
            updated = carried
        else:
            # Erase along the absolute state this sublayer reads, then write its own update
            if num_blocks > 1:
                top = logits[..., 1:max_blocks].argmax(dim=-1) + 1
                index = top.unsqueeze(-1).unsqueeze(-1).expand(*top.shape, 1, hidden_size)
                key = F.normalize(sources.gather(-2, index).squeeze(-2).float(), dim=-1)
            else:
                key = F.normalize(head.float(), dim=-1)
            projection = torch.einsum("...d,...d->...", key, carried)[..., None]
            updated = torch.addcmul(carried, torch.sigmoid(erase) * key, projection, value=-1.0)
            updated = torch.addcmul(updated, torch.sigmoid(write_gate), delta)

        branch = updated + mixed - global_state
        output = anchor + torch.tanh(residual_gate) * (branch - anchor)
        anchor_rms = (anchor.square().mean(dim=-1, keepdim=True) + self.norm.eps).sqrt()
        output_rms = (output.square().mean(dim=-1, keepdim=True) + self.norm.eps).sqrt()
        output = output * (anchor_rms / output_rms).detach().clamp(max=10.0)
        return (output - anchor).to(out_dtype)


class ShensiAttnBank(nn.Module):
    """Persistent low-rank residuals.

    The bank stores the attention-residual block summaries and the latents that carry the hyper-connection
    streams between layers. Both bases stay in fp32 at rest and encode/decode cast them to the activation dtype.
    """

    def __init__(self, config: ShensiConfig):
        super().__init__()
        # Hyper-parameters
        self.rank = config.attn_res_latent_size
        self.stream_rank = config.hc_latent_size
        self.hidden_size = config.hidden_size
        # Learnable parameters
        self.basis = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        self.stream_basis = nn.Parameter(torch.empty(self.stream_rank, self.hidden_size))
        self.token_down = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.token_up = nn.Linear(self.rank, self.hidden_size, bias=False)
        # Transient per-forward state: never saved in the state dict
        self.head = nn.Buffer(None, persistent=False)
        self.token_bias = nn.Buffer(None, persistent=False)
        self.block_residual = nn.Buffer(None, persistent=False)
        self.num_valid_blocks = nn.Buffer(torch.zeros((), dtype=torch.long), persistent=False)

    def open(self, inputs_embeds: torch.Tensor, num_blocks: int) -> "ShensiAttnBank":
        # Clear stale state first: a failed forward must not leak activations into the next DDP step.
        self.head = self.token_bias = self.block_residual = None
        self.num_valid_blocks.zero_()
        # The exact head embedding and its token bias stay outside the compressed bank.
        self.head = inputs_embeds
        self.token_bias = self.token_up(self.token_down(inputs_embeds))
        self.block_residual = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], num_blocks, self.rank)
        self.num_valid_blocks.zero_()
        return self

    def write(self, prefix_sum: torch.Tensor, block_idx: int | None = None) -> None:
        # Summaries are stored in basis coordinates.
        if block_idx is None:
            block_idx = int(self.num_valid_blocks) + 1
        index = torch.tensor([block_idx - 1], device=prefix_sum.device, dtype=torch.long)
        summary = self.encode((prefix_sum - self.token_bias).unsqueeze(-2))
        self.block_residual = self.block_residual.index_copy(-2, index, summary)
        self.num_valid_blocks.fill_(block_idx)

    def read_empty(self) -> torch.Tensor:
        num_blocks = self.block_residual.shape[-2]
        bias = self.token_bias.unsqueeze(-2).expand(-1, -1, num_blocks, -1)
        return torch.cat([self.head.unsqueeze(-2), bias], dim=-2)

    def read(self, num_blocks: int | None = None) -> torch.Tensor:
        blocks = self.block_residual if num_blocks is None else self.block_residual[..., :num_blocks, :]
        decoded = self.decode(blocks) + self.token_bias.unsqueeze(-2)
        return torch.cat([self.head.unsqueeze(-2), decoded], dim=-2)

    def update(self, blocks: torch.Tensor) -> None:
        # Only the summary rows are re-encoded; row 0 is the exact head.
        self.head = blocks[..., 0, :]
        self.block_residual = self.encode(blocks[..., 1:, :] - self.token_bias.unsqueeze(-2))

    def encode(self, blocks: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...h,rh->...r", blocks, self.basis.to(blocks.dtype))

    def decode(self, blocks: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...r,rh->...h", blocks, self.basis.to(blocks.dtype))

    def encode_streams(self, streams: torch.Tensor) -> torch.Tensor:
        # Inter-layer stream compression uses its own shared basis.
        return torch.einsum("...h,rh->...r", streams, self.stream_basis.to(streams.dtype))

    def decode_streams(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...r,rh->...h", latents, self.stream_basis.to(latents.dtype))

    def close(self) -> None:
        self.head = self.token_bias = self.block_residual = None
        self.num_valid_blocks.zero_()


class ShensiROSAEmbedding(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.rosa_embed_dim)
        self.embed_proj = nn.Linear(config.rosa_embed_dim, config.hidden_size, bias=False)
        self.stream_gate = nn.Linear(config.hidden_size, config.hc_mult, bias=True)

    def forward(self, hidden_states: torch.Tensor, token_ids: torch.LongTensor) -> torch.Tensor:
        valid = token_ids.ge(0)
        embeddings = self.embed_tokens(token_ids.clamp_min(0)) * valid.unsqueeze(-1)
        features = self.embed_proj(embeddings)
        gate = torch.sigmoid(self.stream_gate(hidden_states.mean(dim=2))).unsqueeze(-1)
        return gate * features.unsqueeze(2)


class ShensiDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_indexed_attention = config.layer_types[layer_idx] == "indexed_attention"
        self.self_attn = (
            ShensiAttention(config, layer_idx)
            if self.is_indexed_attention
            else ShensiLinearAttention(config, layer_idx)
        )
        self.is_hash = config.mlp_layer_types[layer_idx] == "dense"
        self.mlp = ShensiMLP(config, layer_idx) if self.is_hash else ShensiSparseMoeBlock(config, layer_idx)
        self.input_layernorm = ShensiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ShensiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = ShensiHyperConnection(config, is_mlp=False)
        self.ffn_hc = ShensiHyperConnection(config, is_mlp=True)
        self.is_block_write_layer = config.attn_res_block_layer_types[layer_idx] == "block_write_layer"
        self.prev_valid_blocks = sum(
            1 for block_type in config.attn_res_block_layer_types[:layer_idx] if block_type == "block_write_layer"
        )
        self.self_attention_res = ShensiAttentionResidual(config, layer_idx)
        self.mlp_res = ShensiAttentionResidual(config, layer_idx)
        self.rosa = ShensiROSAEmbedding(config) if layer_idx in config.rosa_layer_ids else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        blocks: torch.Tensor,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        rosa_token_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # A block-write layer commits the running prefix and opens the next block
        if self.is_block_write_layer:
            if self.prev_valid_blocks > 0:
                blocks = blocks.index_copy(
                    -2,
                    torch.tensor([self.prev_valid_blocks], device=blocks.device, dtype=torch.long),
                    prefix_sum.unsqueeze(-2),
                )
            prefix_sum = torch.zeros_like(prefix_sum)

        block_idx = self.prev_valid_blocks if self.is_block_write_layer else self.prev_valid_blocks - 1
        num_blocks = 1 + block_idx

        if self.rosa is not None and rosa_token_ids is not None:
            hidden_states = hidden_states + self.rosa(hidden_states, rosa_token_ids)

        # Attention
        collapsed, attn_state = self.attn_hc(hidden_states)
        topk_indices = prev_topk_indices
        if self.is_indexed_attention:
            attn_output, _, topk_indices = self.self_attn(
                self.input_layernorm(collapsed),
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                prev_topk_indices=prev_topk_indices,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )
        else:
            attn_output = self.self_attn(
                self.input_layernorm(collapsed), cache_params=past_key_values, attention_mask=attention_mask
            )

        attn_correction = self.self_attention_res(
            prefix_sum=prefix_sum,
            delta=attn_output,
            blocks=blocks,
            num_blocks=num_blocks,
        )
        hidden_states = self.attn_hc.write_back(attn_output, attn_state, attn_correction)
        prefix_sum = prefix_sum + attn_output

        # FFN
        collapsed, mlp_state = self.ffn_hc(hidden_states)
        mlp_input = self.post_attention_layernorm(collapsed)
        mlp_output = self.mlp(mlp_input, input_ids=input_ids) if self.is_hash else self.mlp(mlp_input)

        mlp_correction = self.mlp_res(
            prefix_sum=prefix_sum,
            delta=mlp_output,
            blocks=blocks,
            num_blocks=num_blocks,
        )
        hidden_states = self.ffn_hc.write_back(mlp_output, mlp_state, mlp_correction)
        prefix_sum = prefix_sum + mlp_output
        return hidden_states, prefix_sum, blocks, topk_indices


class ShensiStreamEmbeddings(nn.Module):
    """Embed a token sequence into the hyper-connection streams.

    Stream 0 is the shared token embedding. Each auxiliary stream owns an independent token embedding table,
    initialized as a copy of the shared table, and adds a depthwise causal convolution for local temporal
    context. `input_ids` selects the per-stream tables; when only `inputs_embeds` are available, the auxiliary
    streams use the shared embedding plus the convolution.
    """

    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.num_streams = config.hc_mult
        self.hidden_size = config.hidden_size
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.hidden_size,
                    self.hidden_size,
                    kernel,
                    groups=self.hidden_size,
                    padding=kernel - 1,
                )
                for kernel in config.hc_conv_kernels
            ]
        )
        self.token_embeddings = (
            nn.Embedding(config.vocab_size, (self.num_streams - 1) * self.hidden_size)
            if self.num_streams > 1
            else None
        )

    def forward(self, inputs_embeds: torch.Tensor, input_ids: torch.LongTensor | None = None) -> torch.Tensor:
        streams = [inputs_embeds]
        if self.convs:
            conv_input = inputs_embeds.transpose(1, 2).to(
                self.convs[0].weight.dtype, memory_format=torch.contiguous_format
            )
            convs = [
                conv(conv_input)[..., : inputs_embeds.shape[1]].transpose(1, 2).to(inputs_embeds.dtype)
                for conv in self.convs
            ]
        else:
            convs = []

        if self.token_embeddings is None:
            streams.extend(convs)
        elif input_ids is None:
            streams.extend(inputs_embeds + conv for conv in convs)
        else:
            tokens = self.token_embeddings(input_ids).view(*input_ids.shape, self.num_streams - 1, self.hidden_size)
            streams.extend(tokens[..., index, :].to(inputs_embeds.dtype) + conv for index, conv in enumerate(convs))
        return torch.stack(streams, dim=2)


@auto_docstring
class ShensiPreTrainedModel(PreTrainedModel):
    config: ShensiConfig
    config_class = ShensiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keep_in_fp32_modules_strict = [
        "A_log",
        "basis",
        "conv1d",
        "convs",
        "dt_bias",
        "hc_base",
        "hc_fn",
        "input_layernorm",
        "kv_norm",
        "o_norm",
        "post_attention_layernorm",
        "q_a_layernorm",
        "routed_expert_norm",
        "scale",
    ]
    _no_split_modules = ["ShensiDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_static_cache = False
    _can_compile_fullgraph = False
    _is_stateful = True
    _can_record_outputs = {
        "attentions": ShensiAttention,
        "hidden_states": [ShensiAttention, ShensiLinearAttention],
        "router_logits": OutputRecorder(ShensiTopKRouter, index=0),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, ShensiMLP):
            init.zeros_(module.token_shift)
            if module.deepemb is not None:
                init.copy_(module.deepemb.weight, torch.eye(module.deepemb_rank).reshape(1, -1))
            init.zeros_(module.deepemb_b_proj.weight)
            init.ones_(module.deepemb_b_proj.bias)
        elif isinstance(module, ShensiTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, ShensiExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, ShensiAttention):
            if module.kv_emb is not None:
                init.zeros_(module.kv_emb.weight)
        elif isinstance(module, ShensiIndexer):
            if module.k_emb is not None:
                init.ones_(module.k_emb.weight)
        elif isinstance(module, ShensiForgetGate):
            init.copy_(module.A_log, init.uniform_(module.A_log, a=1.0, b=16.0).log())
            init.uniform_(module.dt_bias, a=math.log(1e-3), b=math.log(1e-1))
            dt = module.dt_bias.exp().clamp_min(1e-4)
            init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))
        elif isinstance(module, ShensiHyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, ShensiHyperConnection):
            # A zero pre/route base keeps the initial mixing uniform.
            init.zeros_(module.base)
            init.ones_(module.scale)
            if module.num_route_coefficients > 0:
                values = torch.zeros_like(module.fn)
                values[module.hc_mult : module.hc_mult + module.num_route_coefficients].normal_(mean=0.0, std=std)
                init.copy_(module.fn, values)
            else:
                init.zeros_(module.fn)
            if module.down is not None:
                init.normal_(module.down, mean=0.0, std=std)
                init.zeros_(module.up)
                init.normal_(module.decay, mean=-3.0, std=0.5)
            if module.convs is not None:
                for conv in module.convs:
                    init.zeros_(conv.weight)
        elif isinstance(module, ShensiAttentionResidual):
            # Zero gate and bias keep the module an identity at init
            init.normal_(module.q_proj.weight, mean=0.0, std=std)
            init.normal_(module.k_proj.weight, mean=0.0, std=std)
            if module.g_a_proj is not None:
                init.normal_(module.g_a_proj.weight, mean=0.0, std=std)
                init.normal_(module.g_b_proj.weight, mean=0.0, std=std)
                init.copy_(
                    module.g_b_proj.bias,
                    module.g_b_proj.bias.new_tensor([3.0, -3.0, 3.0]).repeat_interleave(module.config.hidden_size),
                )
            init.zeros_(module.bias)
            init.copy_(module.weight, module.weight.new_tensor([1.0, 1.0e-3]))
        elif isinstance(module, ShensiROSAEmbedding):
            init.zeros_(module.stream_gate.weight)
            init.constant_(module.stream_gate.bias, -4.0)
        elif isinstance(module, ShensiStreamEmbeddings):
            # Cyclic replication: auxiliary tables start as copies of E(x); convolutions are inert.
            for conv in module.convs:
                init.zeros_(conv.weight)
                init.zeros_(conv.bias)
            if module.token_embeddings is not None:
                embed = getattr(self, "embed_tokens", None)
                if embed is None:
                    embed = self.get_input_embeddings()
                source = embed.weight
                for k in range(module.num_streams - 1):
                    start = k * self.config.hidden_size
                    init.copy_(
                        module.token_embeddings.weight[:, start : start + self.config.hidden_size],
                        source,
                    )
        elif isinstance(module, ShensiAttnBank):
            # Identity bases start as truncation projections.
            eye = torch.eye(max(self.config.hidden_size, module.rank))[: module.rank, : self.config.hidden_size]
            init.copy_(module.basis, eye.to(module.basis.device))
            stream_eye = torch.eye(max(self.config.hidden_size, module.stream_rank))[
                : module.stream_rank, : self.config.hidden_size
            ].to(module.stream_basis.device)
            init.copy_(module.stream_basis, stream_eye)
            init.normal_(module.token_down.weight, mean=0.0, std=std)
            init.zeros_(module.token_up.weight)
            init.zeros_(module.num_valid_blocks)


@torch.no_grad()
def rosa(
    token_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    eos_token_id: int | None = None,
) -> torch.LongTensor:
    """ROSA next-token predictions for a full token sequence.

    For each position, return the token that followed the longest matching suffix seen earlier, or -1 when the suffix
    has not occurred before. Ties on the matched length go to the most recent occurrence, and `eos_token_id` restarts
    the automaton. A fresh cache is used, so the operator is stateless.
    """
    return ShensiROSACache().update(token_ids, attention_mask=attention_mask, eos_token_id=eos_token_id)


class ShensiModel(Glm5NextTextModel):
    config: ShensiConfig

    def __init__(self, config: ShensiConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ShensiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ShensiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = ShensiHyperHead(config)

        self.stream_embeddings = ShensiStreamEmbeddings(config)

        self.num_attn_res_blocks = config.num_attn_res_blocks
        self.attn_bank = ShensiAttnBank(config)
        self.output_attn_res = ShensiAttentionResidual(config, config.num_hidden_layers)

        self.tie_mlp()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = self._create_cache()
        if past_key_values is not None:
            self._tie_indexed_caches(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            attention_mask = create_recurrent_attention_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1],
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )
            attention_mask = attention_mask.bool()

            causal_mask_mapping = {
                "indexed_attention": attention_mask,
                "linear_attention": attention_mask,
            }

        # Layer 0 consumes the full-width streams; later layers read the compressed latent.
        streams = self.stream_embeddings(inputs_embeds, input_ids)

        # Seed the bank with the exact embedding head.
        self.attn_bank.open(inputs_embeds, self.num_attn_res_blocks)
        prefix_sum = torch.zeros_like(inputs_embeds)

        # ROSA predictions are computed once and reused by every ROSA layer.
        rosa_token_ids = None
        if input_ids is not None and self.config.rosa_layer_ids:
            rosa_attention_mask = attention_mask
            if isinstance(rosa_attention_mask, dict):
                rosa_attention_mask = rosa_attention_mask.get("indexed_attention")
            if not isinstance(rosa_attention_mask, torch.Tensor) or rosa_attention_mask.dim() != 2:
                rosa_attention_mask = None
            reset_token_id = self.config.rosa_reset_token_id

            write_layers = [
                layer_idx
                for layer_idx, layer_type in enumerate(self.config.layer_types)
                if layer_type == "indexed_attention"
            ]
            if past_key_values is not None and write_layers:
                cache_layer = past_key_values.layers[write_layers[0]]
                if hasattr(cache_layer, "rosa_cache"):
                    if cache_layer.rosa_cache is None:
                        previous_ids = getattr(cache_layer, "token_ids", None)
                        cache_layer.rosa_cache = (
                            ShensiROSACache.from_token_ids(previous_ids)
                            if previous_ids is not None
                            else ShensiROSACache()
                        )
                    rosa_token_ids = cache_layer.rosa_cache.update(
                        input_ids,
                        attention_mask=rosa_attention_mask,
                        eos_token_id=reset_token_id,
                    )
            if rosa_token_ids is None:
                rosa_token_ids = rosa(input_ids, attention_mask=rosa_attention_mask, eos_token_id=reset_token_id)

        # The bank is read once; `blocks` travels through the layers directly.
        layers = self.layers[: self.config.num_hidden_layers]
        topk_indices = None
        blocks = self.attn_bank.read_empty()
        for i, decoder_layer in enumerate(layers):
            # Decode at layer boundaries; layer 0 reads the expansion directly.
            layer_streams = streams if i == 0 else self.attn_bank.decode_streams(streams)
            hidden_states, prefix_sum, blocks, topk_indices = decoder_layer(
                layer_streams,
                prefix_sum,
                blocks,
                input_ids=input_ids,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=None,
                prev_topk_indices=topk_indices,
                rosa_token_ids=rosa_token_ids,
                **kwargs,
            )
            # Compress after every layer except the last; the head consumes full-width streams.
            streams = self.attn_bank.encode_streams(hidden_states) if i + 1 < len(layers) else hidden_states

        # Sync the bank once and flush the last partial block.
        self.attn_bank.update(blocks)
        self.attn_bank.write(prefix_sum, self.num_attn_res_blocks)
        residual = self.attn_bank.read()
        self.attn_bank.close()
        hidden_states = self.hc_head(streams)
        hidden_states = hidden_states + self.output_attn_res(
            prefix_sum=torch.zeros_like(prefix_sum),
            delta=hidden_states,
            blocks=residual,
            num_blocks=residual.shape[-2],
        )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _create_cache(self) -> Cache:
        cache = DynamicCache(config=self.config)
        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "indexed_attention":
                cache.layers[layer_idx] = ShensiAttentionCache()
        return cache

    def _tie_indexed_caches(self, cache: Cache) -> None:
        indexed_layers = [
            layer_idx
            for layer_idx, layer_type in enumerate(self.config.layer_types)
            if layer_type == "indexed_attention"
        ]
        if len(indexed_layers) < 2:
            return
        owner = cache.layers[indexed_layers[0]]
        if not hasattr(owner, "source"):
            return
        for layer_idx in indexed_layers[1:]:
            layer = cache.layers[layer_idx]
            if hasattr(layer, "source"):
                layer.source = owner

    def tie_mlp(self) -> None:
        write_layers = [
            i
            for i, block_type in enumerate(self.config.attn_res_block_layer_types)
            if block_type == "block_write_layer"
        ]
        shared = {}
        for layer_idx, layer in enumerate(self.layers):
            if layer.is_hash:
                continue
            block_id = max(w for w in write_layers if w <= layer_idx)
            group = shared.setdefault(block_id, [layer.mlp, [layer_idx]])
            if group[0] is not layer.mlp:
                group[1].append(layer_idx)
                layer.mlp.gate = ShensiTopKRouter(self.config)
                layer.mlp.gate.weight = group[0].gate.weight
                layer.mlp.experts = group[0].experts

        group_patterns = {}
        group_tied_weights = {}
        moe_position = {
            layer_idx: pos
            for pos, layer_idx in enumerate(idx for idx, layer in enumerate(self.layers) if not layer.is_hash)
        }
        for first_mlp, indices in shared.values():
            first_mlp.sharing_layers = [moe_position[idx] for idx in indices]
            for idx in indices[1:]:
                for attr in ("gate", "experts"):
                    source_key = f"layers.{indices[0]}.mlp.{attr}"
                    target_key = f"layers.{idx}.mlp.{attr}"
                    group_patterns[target_key] = source_key
                    for param_name, _ in getattr(first_mlp, attr).named_parameters():
                        group_tied_weights[f"{target_key}.{param_name}"] = f"{source_key}.{param_name}"

        hash_layers = [(i, layer) for i, layer in enumerate(self.layers) if getattr(layer, "is_hash", False)]
        if len(hash_layers) > 1:
            first_idx, first_layer = hash_layers[0]
            for idx, layer in hash_layers[1:]:
                layer.mlp.deepemb = first_layer.mlp.deepemb
                source_key = f"layers.{first_idx}.mlp.deepemb"
                target_key = f"layers.{idx}.mlp.deepemb"
                group_patterns[target_key] = source_key
                group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        write_attention_layers = [layer for layer in self.layers if layer.is_indexed_attention]
        if len(write_attention_layers) > 1:
            first_idx = write_attention_layers[0].layer_idx
            first_attn = write_attention_layers[0].self_attn
            if first_attn.kv_emb is not None:
                for layer in write_attention_layers[1:]:
                    layer.self_attn.kv_emb = first_attn.kv_emb
                    source_key = f"layers.{first_idx}.self_attn.kv_emb"
                    target_key = f"layers.{layer.layer_idx}.self_attn.kv_emb"
                    group_patterns[target_key] = source_key
                    group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        full_indexers = [layer for layer in write_attention_layers if layer.self_attn.indexer is not None]
        if len(full_indexers) > 1:
            first_idx = full_indexers[0].layer_idx
            first_emb = full_indexers[0].self_attn.indexer.k_emb
            if first_emb is not None:
                for layer in full_indexers[1:]:
                    layer.self_attn.indexer.k_emb = first_emb
                    source_key = f"layers.{first_idx}.self_attn.indexer.k_emb"
                    target_key = f"layers.{layer.layer_idx}.self_attn.indexer.k_emb"
                    group_patterns[target_key] = source_key
                    group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        rosa_layers = [(layer.layer_idx, layer.rosa) for layer in self.layers if layer.rosa is not None]
        if rosa_layers and self.config.rosa_embed_dim == self.config.hidden_size:
            for layer_idx, rosa in rosa_layers:
                rosa.embed_tokens.weight = self.embed_tokens.weight
                target_key = f"layers.{layer_idx}.rosa.embed_tokens"
                group_patterns[target_key] = "embed_tokens"
                group_tied_weights[f"{target_key}.weight"] = "embed_tokens.weight"
        elif len(rosa_layers) > 1:
            first_idx, first_rosa = rosa_layers[0]
            for layer_idx, rosa in rosa_layers[1:]:
                rosa.embed_tokens = first_rosa.embed_tokens
                source_key = f"layers.{first_idx}.rosa.embed_tokens"
                target_key = f"layers.{layer_idx}.rosa.embed_tokens"
                group_patterns[target_key] = source_key
                group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        self._tied_weights_keys = {**dict(self._tied_weights_keys or {}), **group_patterns}
        self._moe_group_tied_weights = group_tied_weights

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        resized = {}

        def resize(table):
            if id(table) not in resized:
                resized[id(table)] = self._get_resized_embeddings(
                    table, new_num_tokens, pad_to_multiple_of, mean_resizing
                )
            return resized[id(table)]

        if self.stream_embeddings.token_embeddings is not None:
            self.stream_embeddings.token_embeddings = resize(self.stream_embeddings.token_embeddings)
        for layer in self.layers:
            if layer.is_hash and layer.mlp.deepemb is not None:
                layer.mlp.deepemb = resize(layer.mlp.deepemb)
            if layer.is_indexed_attention:
                if layer.self_attn.kv_emb is not None:
                    layer.self_attn.kv_emb = resize(layer.self_attn.kv_emb)
                indexer = layer.self_attn.indexer
                if indexer is not None and indexer.k_emb is not None:
                    indexer.k_emb = resize(indexer.k_emb)
            if layer.rosa is not None:
                if self.config.rosa_embed_dim == self.config.hidden_size:
                    layer.rosa.embed_tokens.weight = self.embed_tokens.weight
                else:
                    layer.rosa.embed_tokens = resize(layer.rosa.embed_tokens)
        return embeddings

    def post_init(self):
        PreTrainedModel.post_init(self)
        self.all_tied_weights_keys.update(getattr(self, "_moe_group_tied_weights", None) or {})
        self.tie_weights(recompute_mapping=False)


def erc_loss_func(
    router_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    # Perturb each router inside a small ball and penalize experts whose directions remain too close.
    R = router_weight
    norm_R = torch.norm(R, dim=1)
    distances = torch.cdist(R, R, p=2)
    distances = distances.masked_fill(torch.eye(R.size(0), dtype=torch.bool, device=distances.device), float("inf"))
    min_dist, _ = torch.min(distances, dim=1)
    eps = min_dist / 2 / norm_R

    low = (1 - eps).unsqueeze(1)
    high = (1 + eps).unsqueeze(1)
    noise = torch.rand_like(R)
    R_tilde = (low + noise * (high - low)) * R

    proxy = F.linear(R_tilde, down_proj_weight)
    coupling = torch.norm(torch.einsum("jDd,id->ijD", gate_up_proj, proxy), dim=-1)

    row_diff = torch.clamp(coupling - alpha * torch.diag(coupling).unsqueeze(1), min=0.0)
    col_diff = torch.clamp(coupling - alpha * torch.diag(coupling).unsqueeze(0), min=0.0)
    mask = torch.ones_like(coupling) - torch.eye(coupling.size(0), device=coupling.device)
    return ((row_diff + col_diff) * mask).mean()


class ShensiForCausalLM(DeepseekV4ForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, ShensiForCausalLM

        >>> model = ShensiForCausalLM.from_pretrained("louzongzhi/Shensi-V1-N")
        >>> tokenizer = AutoTokenizer.from_pretrained("louzongzhi/Shensi-V1-N")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        erc_loss = None
        if labels is not None and self.training:
            num_erc_groups = 0
            for layer in self.model.layers:
                if layer.is_hash or getattr(layer.mlp, "sharing_layers", None) is None:
                    continue
                group_loss = erc_loss_func(
                    layer.mlp.gate.weight,
                    layer.mlp.routed_expert_down_proj.weight,
                    layer.mlp.experts.gate_up_proj,
                    alpha=self.config.erc_loss_alpha,
                )
                erc_loss = group_loss if erc_loss is None else erc_loss + group_loss
                num_erc_groups += 1
            if erc_loss is not None:
                loss = loss + self.config.erc_loss_coef * (erc_loss / num_erc_groups).to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        self.model._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        return embeddings


__all__ = [
    "ShensiConfig",
    "ShensiPreTrainedModel",
    "ShensiModel",
    "ShensiForCausalLM",
]
