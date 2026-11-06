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

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicIndexedLayer
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_recurrent_attention_mask
from ...modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4DecoderLayer,
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
    Glm5NextTextAttention,
    Glm5NextTextForgetGate,
    Glm5NextTextIndexer,
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
        Width of the latent space used by the routed experts and attention residuals.
    num_mtp_layers (`int`, *optional*):
        Number of multi-token-prediction layers carried by the checkpoint.
    deepemb_kv_rank (`int`, *optional*):
        Rank of the key/value latent cached by indexed attention.
    kv_emb_rank (`int`, *optional*):
        Rank of the token-embedding factor applied to reconstructed keys and values.
    deepemb_rank (`int`, *optional*):
        Rank of the token-conditioned deep-embedding transform in the dense MLP.
    index_block_size (`int`, *optional*):
        Number of consecutive tokens averaged into one HISA block.
    index_head_dim (`int`, *optional*):
        Dimension of each indexer query and key head.
    index_latent_rank (`int`, *optional*):
        Rank of the latent space used to cache indexer keys.
    index_n_heads (`int`, *optional*):
        Number of query heads in the DSA/HISA indexer.
    index_topk (`int`, *optional*):
        Maximum number of key positions selected per query.
    index_topk_blocks (`int`, *optional*):
        Number of top-scoring blocks retained before token-level refinement.
    index_topk_freq (`int`, *optional*):
        Interval between layers that run the full indexer.
    indexer_types (`list[str]`, *optional*):
        Per-layer indexer mode, either `"full"` or `"shared"`.
    linear_conv_kernel_dim (`int`, *optional*):
        Kernel size of the short convolution in linear-attention layers.
    linear_head_dim (`int`, *optional*):
        Dimension of each key and value head in linear-attention layers.
    linear_lower_bound (`float`, *optional*):
        Lower bound applied to the forget-gate decay in linear-attention layers.
    linear_num_heads (`int`, *optional*):
        Number of key and value heads in linear-attention layers.
    first_k_dense_replace (`int`, *optional*):
        Number of initial layers that use the dense MLP before sparse MoE.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer feed-forward schedule, either `"dense"` or `"sparse"`.
    swiglu_limit (`float`, *optional*):
        Clamp limit applied to the gate and up projections.
    scoring_func (`str`, *optional*):
        Router scoring function: `"sqrtsoftplus"`, `"softmax"` or `"sigmoid"`.
    hc_active_streams (`int`, *optional*):
        Number of hyper-connection streams updated by each sublayer.
    hc_adapter_rank (`int`, *optional*):
        Rank of the low-rank adapter attached to each routed hyper-connection stream.
    hc_conv_kernels (`list[int]`, *optional*):
        Kernel sizes of the depthwise convolutions that create the hyper-connection streams.
    hc_eps (`float`, *optional*):
        Numerical floor used by hyper-connection normalization.
    hc_fixed_streams (`int`, *optional*):
        Number of hyper-connection streams that are always updated.
    hc_mult (`int`, *optional*):
        Total number of parallel hyper-connection streams.
    hc_stream_rank (`int`, *optional*):
        Rank of the basis used to compress hyper-connection streams between layers.
    attn_res_block_layer_types (`list[str]`, *optional*):
        Per-layer attention-residual schedule, either `"block_write_layer"` or `"block_read_layer"`.
    attn_res_block_size (`int`, *optional*):
        Number of consecutive layers between attention-residual write layers.
    attn_res_block_summary_rank (`int`, *optional*):
        Rank of the basis used to store attention-residual block summaries.
    attn_res_rank (`int`, *optional*):
        Rank of the depth-routing keys and channel gates.
    rosa_embed_dim (`int`, *optional*):
        Dimension of the ROSA token embeddings.
    rosa_layer_ids (`list[int]`, *optional*):
        Decoder layer indices that inject ROSA predictions.
    rosa_reset_token_id (`int`, *optional*):
        Token id that resets the ROSA suffix automaton.
    erc_loss_alpha (`float`, *optional*):
        Specialization threshold of the expert-router coupling loss.
    erc_loss_coef (`float`, *optional*):
        Weight of the expert-router coupling loss in the total training loss.
    """

    model_type = "shensi"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Standard MoE aliases used by FP8, TP, and EP
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "intermediate_size": "moe_intermediate_size",
        "block_summary_rank": "attn_res_block_summary_rank",
    }
    # Experts are sharded; the shared projections and hyper-connection modules stay replicated
    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    # Expert-parallel plan routes the same expert tensors through grouped GEMM
    base_model_ep_plan = {
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "ep_dispatch_experts",
    }

    # Core model dimensions
    vocab_size: int = 129280
    hidden_size: int = 1280
    latent_size: int = 320
    num_hidden_layers: int = 19
    num_mtp_layers: int = 1
    num_attention_heads: int = 16
    num_key_value_heads: int = 1
    head_dim: int = 512
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Attention
    layer_types: list[str] | None = None
    q_lora_rank: int = 320
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int = 0
    v_head_dim: int | None = None

    # DeepEmbed
    deepemb_kv_rank: int | None = None
    kv_emb_rank: int | None = None
    deepemb_rank: int = 32

    # Indexer
    index_block_size: int = 128
    index_head_dim: int = 128
    index_latent_rank: int | None = None
    index_n_heads: int = 64
    index_topk: int = 2048
    index_topk_blocks: int = 64
    index_topk_freq: int = 4
    indexer_types: list[str] | None = None

    # Linear attention
    linear_conv_kernel_dim: int = 4
    linear_head_dim: int = 128
    linear_lower_bound: float = -5.0
    linear_num_heads: int = 16

    # MLP
    first_k_dense_replace: int = 3
    mlp_bias: bool = False
    mlp_layer_types: list[str] | None = None
    swiglu_limit: float = 10.0

    # MoE
    moe_intermediate_size: int = 640
    n_routed_experts: int = 256
    norm_topk_prob: bool = True
    num_experts_per_tok: int = 6
    routed_scaling_factor: float = 1.5
    scoring_func: str = "sqrtsoftplus"

    # Hyper-connections
    hc_active_streams: int = 4
    hc_adapter_rank: int = 8
    hc_conv_kernels: list[int] | None = None
    hc_eps: float = 1.0e-6
    hc_fixed_streams: int = 1
    hc_mult: int = 4
    hc_stream_rank: int | None = None

    # Attention residuals
    attn_res_block_layer_types: list[str] | None = None
    attn_res_block_size: int = 2
    attn_res_block_summary_rank: int | None = None
    attn_res_rank: int = 32

    # ROSA
    rosa_embed_dim: int | None = None
    rosa_layer_ids: list[int] | None = None
    rosa_reset_token_id: int | None = None

    # Losses
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001

    def __post_init__(self, **kwargs):
        n = self.num_hidden_layers

        # Resolve defaults that depend on other fields
        if self.num_key_value_heads is None:
            self.num_key_value_heads = 1
        if self.hc_conv_kernels is None:
            self.hc_conv_kernels = [4 * (i + 1) for i in range(self.hc_mult - 1)]
        else:
            self.hc_mult = 1 + len(self.hc_conv_kernels)
        if self.rosa_embed_dim is None:
            self.rosa_embed_dim = self.hidden_size
        if self.qk_nope_head_dim is None:
            self.qk_nope_head_dim = self.head_dim
        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim

        # Default compression ranks are fractions of the corresponding head or hidden size
        if self.deepemb_kv_rank is None:
            self.deepemb_kv_rank = self.head_dim // 8
        if self.kv_emb_rank is None:
            self.kv_emb_rank = self.head_dim // 4
        if self.index_latent_rank is None:
            self.index_latent_rank = self.index_head_dim // 4
        if self.hc_stream_rank is None:
            self.hc_stream_rank = self.hidden_size // 4
        if self.attn_res_block_summary_rank is None:
            self.attn_res_block_summary_rank = self.hidden_size // 4

        # Dense prefix followed by sparse MoE layers
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * self.first_k_dense_replace + ["sparse"] * (
                n - self.first_k_dense_replace
            )

        # Layer 0 writes first; later writes follow attn_res_block_size.
        if self.attn_res_block_layer_types is None:
            n_dense = self.mlp_layer_types.count("dense")
            self.attn_res_block_layer_types = [
                "block_write_layer"
                if i == 0 or (i >= n_dense and (i - n_dense) % self.attn_res_block_size == 0)
                else "block_read_layer"
                for i in range(n)
            ]

        # Indexed attention and attention-residual writes share the same layer positions
        if self.layer_types is None:
            self.layer_types = [
                "indexed_attention" if block_type == "block_write_layer" else "linear_attention"
                for block_type in self.attn_res_block_layer_types
            ]

        # A full indexer runs every `index_topk_freq` indexed layers; the others reuse its top-k
        write_layers = [i for i, layer_type in enumerate(self.layer_types) if layer_type == "indexed_attention"]
        if self.indexer_types is None:
            self.indexer_types = ["shared"] * n
            for pos, layer_idx in enumerate(write_layers):
                if pos % self.index_topk_freq == 0:
                    self.indexer_types[layer_idx] = "full"
        elif len(self.indexer_types) == len(write_layers) and len(write_layers) != n:
            # Expand indexer types listed only for the indexed layers
            indexer_types = ["shared"] * n
            for pos, layer_idx in enumerate(write_layers):
                indexer_types[layer_idx] = self.indexer_types[pos]
            self.indexer_types = indexer_types

        # Place ROSA at the first and third quartiles of the sparse stack
        if self.rosa_layer_ids is None:
            n_dense = self.mlp_layer_types.count("dense")
            n_sparse = n - n_dense
            first_rosa = n_dense + n_sparse // 4
            middle_rosa = n_dense + 3 * n_sparse // 4
            self.rosa_layer_ids = sorted({first_rosa, middle_rosa})

        super().__post_init__(**kwargs)


class ShensiRMSNorm(DeepseekV4RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


class ShensiUnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    pass


class ShensiRMSNormGated(Glm5NextTextRMSNormGated):
    pass


def block_mean_pooling(k: torch.Tensor, k_scale: torch.Tensor | None, k_block_size: int) -> torch.Tensor:
    # Average keys within non-overlapping blocks
    seq_len, head_dim = k.shape[-2:]
    num_blocks = (seq_len + k_block_size - 1) // k_block_size
    compute_dtype = k.dtype if k.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

    dequant = k.to(compute_dtype)
    if k_scale is not None:
        dequant = dequant * k_scale.to(compute_dtype).unsqueeze(-1)

    counts = torch.full((*k.shape[:-2], num_blocks), k_block_size, device=k.device, dtype=torch.float32)
    padded_len = num_blocks * k_block_size
    if padded_len != seq_len:
        # The final block is partial, so use the real number of tokens in its average
        dequant = F.pad(dequant, (0, 0, 0, padded_len - seq_len))
        counts[..., -1] = seq_len - (num_blocks - 1) * k_block_size

    blocked = dequant.reshape(*dequant.shape[:-2], num_blocks, k_block_size, head_dim).sum(dim=-2, dtype=torch.float32)
    return blocked / counts.unsqueeze(-1)


def pool_mqa(
    q: torch.Tensor,
    blocked_k: torch.Tensor,
    blocked_k_scale: torch.Tensor | None,
    weights: torch.Tensor,
) -> torch.Tensor:
    # Score query heads against pooled key blocks
    q = q.transpose(-3, -2)
    k = blocked_k.to(q.dtype)
    if blocked_k_scale is not None:
        k = k * blocked_k_scale.to(q.dtype).unsqueeze(-1)

    scores = F.relu(torch.matmul(q, k.transpose(-1, -2).unsqueeze(-3)))
    # Reduce the per-head scores in float32 so block ranking is not lossy in bf16
    weights = weights.transpose(-1, -2).float().unsqueeze(-1)
    return (scores.float() * weights).sum(dim=-3)


def clean_and_maintain_logits(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    # Keep the first and last visible blocks selectable and mask the rest
    num_blocks = logits.shape[-1]
    positions = torch.arange(num_blocks, device=logits.device)

    first_block = cu_seqlen_ks.clamp(0, num_blocks - 1).long().unsqueeze(-1)
    last_block = (cu_seqlen_ke - 1).clamp(0, num_blocks - 1).long().unsqueeze(-1)
    # Boundary blocks stay selectable so the first visible token and the current token survive
    logits.scatter_(-1, first_block, float("inf"))
    logits.scatter_(-1, last_block, float("inf"))

    logits.masked_fill_(
        (positions < cu_seqlen_ks.unsqueeze(-1)) | (positions >= cu_seqlen_ke.unsqueeze(-1)),
        float("-inf"),
    )
    return logits


def block_sparse_mqa(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor | None,
    topk_block_indices: torch.Tensor,
    k_block_size: int,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    # Score all tokens belonging to the selected key blocks
    num_kv = k.shape[-2]
    block_starts = topk_block_indices.long() * k_block_size
    pos_in_block = torch.arange(k_block_size, device=q.device)
    k_abs = block_starts.unsqueeze(-1) + pos_in_block
    k_safe = k_abs.clamp(0, num_kv - 1)

    k = k.to(q.dtype)
    if k_scale is not None:
        k = k * k_scale.to(q.dtype).unsqueeze(-1)
    if k.dim() == 2:
        gathered_k = k[k_safe]
    else:
        batch_index = torch.arange(k.shape[0], device=k.device).view(-1, 1, 1, 1)
        gathered_k = k[batch_index, k_safe]

    # Flatten the selected blocks so one matmul scores every candidate token
    scores = F.relu(torch.matmul(q, gathered_k.flatten(-3, -2).transpose(-1, -2)))
    weights = weights.float().unsqueeze(-1)
    logits = (scores.float() * weights).sum(dim=-2)

    in_range = (
        (k_abs >= cu_seqlen_ks.unsqueeze(-1).unsqueeze(-1))
        & (k_abs < cu_seqlen_ke.unsqueeze(-1).unsqueeze(-1))
        & (k_abs < num_kv)
    )
    return logits.masked_fill(~in_range.flatten(-2), float("-inf"))


def hisa_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor | None,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
) -> torch.Tensor:
    # Select token indices with hierarchical block-sparse attention
    blocked_k = block_mean_pooling(k, k_scale, k_block_size)
    # Convert token-visible ranges into block-visible ranges, rounding the end up
    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

    block_k_score = pool_mqa(q, blocked_k, None, weights)
    clean_and_maintain_logits(block_k_score, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)

    # Rank blocks first; token scoring only runs on the retained blocks
    block_topk = min(block_topk, block_k_score.shape[-1])
    # Ranking only decides which blocks are kept, so bf16 is enough here and faster
    topk_block_indices = block_k_score.bfloat16().topk(block_topk, dim=-1, sorted=False).indices

    block_sparse_logits = block_sparse_mqa(
        q,
        k,
        k_scale,
        topk_block_indices,
        k_block_size,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )

    topk_tokens = min(topk_tokens, block_sparse_logits.shape[-1])
    relevant_topk_indices = block_sparse_logits.topk(topk_tokens, dim=-1, sorted=False).indices
    # Convert flattened offsets inside retained blocks back to absolute token positions
    absolute_topk_block_indices = topk_block_indices.gather(-1, relevant_topk_indices // k_block_size)
    topk_indices = absolute_topk_block_indices * k_block_size + (relevant_topk_indices % k_block_size)
    topk_indices = topk_indices.to(torch.int32)

    # Return indices relative to each query's visible start
    topk_indices -= cu_seqlen_ks.unsqueeze(-1).to(torch.int32)
    mask_lo = topk_indices >= 0
    mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks).unsqueeze(-1).to(torch.int32) < 0
    return topk_indices.masked_fill(~(mask_lo & mask_hi), -1)


def include_self_token(
    topk_indices: torch.Tensor,
    self_positions: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    # Ensure every valid query keeps its own token in the selected top-k set
    self_positions = self_positions.unsqueeze(-1)
    has_self = topk_indices.eq(self_positions).any(dim=-1, keepdim=True)
    replace = valid.unsqueeze(-1) & ~has_self

    last_token = topk_indices[..., -1:]
    # Replace one slot when the self token is absent, keeping the output width fixed
    last_token = torch.where(replace, self_positions, last_token)

    return torch.cat((topk_indices[..., :-1], last_token), dim=-1)


class ShensiIndexer(Glm5NextTextIndexer):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        del self.qk_rope_head_dim
        del self.index_kpool
        del self.index_kpool_always_select_tail
        del self.index_kpool_compress_ape
        del self.index_kpool_compress_gate

        self.topk_blocks = config.index_topk_blocks
        self.block_size = config.index_block_size
        self.latent_rank = config.index_latent_rank
        self.down_proj = nn.Linear(self.head_dim, self.latent_rank, bias=False)
        self.up_proj = nn.Linear(self.latent_rank, self.head_dim, bias=False)
        self.token_emb = (
            nn.Embedding(config.vocab_size, self.head_dim) if layer_idx == config.indexer_types.index("full") else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: torch.BoolTensor,
        past_key_values: Cache | None,
        token_ids: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid.to(self.wq_b.weight.dtype)).view(hidden_shape) * self.softmax_scale
        k_in = self.wk(hidden_states.to(self.wk.weight.dtype))
        k = self.k_norm(k_in.to(self.k_norm.weight.dtype)).to(k_in.dtype)
        k = k.view(hidden_shape).squeeze(2)

        valid_channel = attention_mask.to(k.dtype)[..., None]

        # Cache only the low-rank key; validity is stored as one extra channel
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            if cache_layer.source is not None and cache_layer.source.indexer_keys is not None:
                packed_states = cache_layer.source.indexer_keys
            else:
                stored = self.down_proj(k.to(self.down_proj.weight.dtype)).clamp(-448.0, 448.0)
                packed_states = past_key_values.update_indexer(
                    torch.cat([stored.to(torch.float8_e4m3fn), valid_channel.to(torch.float8_e4m3fn)], dim=-1),
                    self.layer_idx,
                )
            kv_len = packed_states.shape[1]
            current_length = kv_len
        else:
            stored = self.down_proj(k.to(self.down_proj.weight.dtype))
            packed_states = torch.cat([stored, valid_channel], dim=-1)
            kv_len = seq_len
            current_length = seq_len

        # Reconstruct full index keys from the cached latent
        keys = packed_states[..., : self.latent_rank].to(self.up_proj.weight.dtype)
        keys = self.up_proj(keys)
        if self.token_emb is not None and token_ids is not None and token_ids.shape[1] >= keys.shape[1]:
            # The shared embedding restores token identity lost by low-rank compression
            keys = keys * self.token_emb(token_ids[:, -keys.shape[1] :]).to(keys.dtype)
        valid_keys = packed_states[..., -1].bool()

        # Indexer logits are reduced in float32 for stable top-k ranking
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)

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
        # Each query only sees the contiguous valid key range ending at its own position
        q_positions = torch.arange(seq_len, device=hidden_states.device) + (current_length - seq_len)
        query_end = torch.minimum(q_positions.unsqueeze(0) + 1, last_valid.unsqueeze(-1) + 1)
        query_start = first_valid.unsqueeze(-1).expand(-1, seq_len)
        valid_query = attention_mask.bool()
        query_start = torch.where(valid_query, query_start, torch.zeros_like(query_start))
        query_end = torch.where(valid_query, query_end, torch.zeros_like(query_end))

        # The indexer scores blocks first and then tokens, returning positions relative to query_start
        topk_indices = hisa_indexer(
            q,
            keys,
            None,
            weights,
            query_start,
            query_end,
            k_block_size=self.block_size,
            block_topk=self.topk_blocks,
            topk_tokens=self.index_topk,
        )

        # Convert relative positions back to absolute cache positions
        topk_indices = torch.where(
            topk_indices >= 0, topk_indices + query_start.unsqueeze(-1).to(torch.int32), topk_indices
        )
        # Drop selections that landed on padding keys
        selected_valid = (
            valid_keys.unsqueeze(1).expand(-1, seq_len, -1).gather(2, topk_indices.clamp(0, keys.shape[1] - 1))
        )
        topk_indices = topk_indices.masked_fill((topk_indices >= 0) & ~selected_valid, -1)

        # Force the query's own key into the selection, evicting one scored slot if needed
        self_positions = (current_length - seq_len + torch.arange(seq_len, device=hidden_states.device)).to(
            torch.int32
        )
        self_positions = self_positions.unsqueeze(0).expand(batch_size, -1)
        self_valid = (
            attention_mask.bool()
            & (self_positions < kv_len)
            & valid_keys.gather(1, self_positions.clamp(0, kv_len - 1).long())
        )
        topk_indices = include_self_token(topk_indices, self_positions, self_valid)

        # Normalize the width so every layer exposes the same top-k shape, padding with -1
        output_width = self.index_topk
        topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
        topk_indices = topk_indices[..., :output_width]
        # Padded query slots see nothing
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        return topk_indices.to(torch.int32)

    def get_visible_tokens(self):
        raise AttributeError()

    def get_pooled_states(self):
        raise AttributeError()

    def append_visible_tail(self):
        raise AttributeError()


class ShensiAttention(Glm5NextTextAttention):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        del self.kv_a_proj_with_mqa
        del self.kv_a_layernorm
        del self.kv_b_proj
        del self.kv_lora_rank
        self.q_a_layernorm = ShensiRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * (self.qk_head_dim + self.v_head_dim), bias=False)
        self.indexer = None if self.skip_topk else ShensiIndexer(config, layer_idx)
        self.next_skip_topk = (
            not self.skip_topk and config.indexer_types[min(layer_idx + 1, len(config.indexer_types) - 1)] == "shared"
        )

        self.kv_latent_proj = nn.Linear(config.hidden_size, config.deepemb_kv_rank, bias=False)
        self.kv_up_proj = nn.Linear(config.deepemb_kv_rank, config.head_dim, bias=False)
        self.kv_norm = ShensiRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        first_write = config.layer_types.index("indexed_attention")
        self.kv_emb_down = nn.Embedding(config.vocab_size, config.kv_emb_rank) if layer_idx == first_write else None
        self.kv_emb_up = (
            nn.Linear(config.kv_emb_rank, config.head_dim, bias=False) if layer_idx == first_write else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        input_ids = kwargs.pop("input_ids", None)
        batch_size, seq_len = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_len, -1, self.qk_head_dim)

        # One projection packs queries and the output gate along the last axis
        q_residual = self.q_a_layernorm(self.q_a_proj(hidden_states.to(self.q_a_proj.weight.dtype)))
        q_and_gate = self.q_b_proj(q_residual)
        query_states, gate = torch.split(
            q_and_gate,
            [self.num_heads * self.qk_head_dim, self.num_heads * self.v_head_dim],
            dim=-1,
        )
        query_states = query_states.view(query_shape).transpose(1, 2)
        gate = gate.view(batch_size, seq_len, self.num_heads, self.v_head_dim)

        # Only the low-rank latent is cached; K and V are rebuilt from it on every step
        current_length = seq_len
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            if cache_layer.source is not None and cache_layer.source.keys is not None:
                source = cache_layer.source
                token_ids = source.token_ids
                cache_layer.token_ids = token_ids
                kv_latents = source.keys.squeeze(1).to(self.kv_up_proj.weight.dtype)
                current_length = source.get_seq_length()
            else:
                kv_latents = self.kv_latent_proj(hidden_states)
                keys, _ = past_key_values.update(kv_latents.unsqueeze(1), kv_latents.unsqueeze(1), self.layer_idx)
                kv_latents = keys.squeeze(1)
                token_ids = (
                    cache_layer.update_token_ids(input_ids) if hasattr(cache_layer, "update_token_ids") else None
                )
                current_length = cache_layer.get_seq_length()
        else:
            kv_latents = self.kv_latent_proj(hidden_states)
            token_ids = input_ids

        # The shared token embedding restores identity information that the latent drops
        kv = self.kv_norm(self.kv_up_proj(kv_latents.to(self.kv_up_proj.weight.dtype)))
        if token_ids is not None and self.kv_emb_down is not None and token_ids.shape[1] >= kv.shape[1]:
            token_emb = self.kv_emb_up(self.kv_emb_down(token_ids[:, -kv.shape[1] :]).to(self.kv_emb_up.weight.dtype))
            kv = kv * token_emb.to(kv.dtype)
        kv = kv.unsqueeze(1)

        # Shared-KV layers consume the indices computed by the last full indexer layer
        if self.skip_topk:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices
        else:
            topk_indices = self.indexer(
                hidden_states=hidden_states,
                q_resid=q_residual,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                token_ids=token_ids,
            )

        # Top-k indices replace the causal mask for attention
        kv_length = kv.shape[2]
        attention_mask = self.build_attention_mask_from_topk(
            topk_indices=topk_indices,
            query_states=query_states,
            kv_length=kv_length,
        )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # Remove the component along the query's own value so a token cannot copy itself
        positions = current_length - seq_len + torch.arange(seq_len, device=hidden_states.device)
        self_values = kv[:, 0].index_select(1, positions).unsqueeze(2).to(attn_output.dtype)
        denom = self_values.square().sum(dim=-1, keepdim=True) + 1e-6
        attn_output = attn_output - (attn_output * self_values).sum(dim=-1, keepdim=True) / denom * self_values

        # The sigmoid gate is applied per head before the output projection
        attn_output = attn_output * torch.sigmoid(gate.to(attn_output.dtype))

        attn_output = attn_output.reshape(batch_size, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices if self.next_skip_topk else None

    def expand_kv(self):
        raise AttributeError()


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
        # Blend the previous token into the current one before the gated projections
        shifted = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        x = x + (shifted - x) * self.token_shift.to(x.dtype)
        # The clamped gated path is scaled by a token-conditioned deep-embedding factor
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        # The embedding table supplies a small matrix per token, applied in a low-rank space
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
        # Project down once, route in the latent space, then project back to hidden size
        routed = self.experts(self.routed_expert_down_proj(flat), indices, weights)
        return self.routed_expert_up_proj(self.routed_expert_norm(routed)).view(batch, seq_len, hidden_dim)


class ShensiRoutedAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden_states))


class ShensiHyperConnection(nn.Module):
    def __init__(self, config: ShensiConfig, with_conv: bool):
        super().__init__()
        self.num_streams = config.hc_mult
        self.active_streams = config.hc_active_streams
        self.fixed_streams = config.hc_fixed_streams
        self.routed_streams = self.num_streams - self.fixed_streams

        flat_size = self.num_streams * config.hidden_size
        # Use one projection for stream weights; a two-factor gate has a zero-product dead zone
        self.pre = nn.Linear(flat_size, self.num_streams, bias=False)
        self.scale = nn.Parameter(torch.ones(2))

        # Only routed streams execute a stream-specific adapter; the fixed stream is the residual anchor
        use_conditional = self.active_streams < self.num_streams and self.routed_streams > 0
        self.route = nn.Linear(flat_size, self.routed_streams, bias=False) if use_conditional else None
        self.adapters = (
            nn.ModuleList(
                [ShensiRoutedAdapter(config.hidden_size, config.hc_adapter_rank) for _ in range(self.routed_streams)]
            )
            if use_conditional
            else None
        )
        self.decay_logit = nn.Parameter(self._default_decay_logits(torch.empty(0).device)) if use_conditional else None

        kernels = config.hc_conv_kernels if with_conv else []
        self.post = nn.Linear(flat_size, self.active_streams * (1 + len(kernels)), bias=True)
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

    def _default_decay_logits(self, device=None) -> torch.Tensor:
        if self.routed_streams == 0:
            return torch.full((self.num_streams,), -10.0, device=device)
        decays = torch.linspace(0.999, 0.9, self.routed_streams, device=device)
        routed = torch.log((1.0 - decays) / decays)
        return torch.cat([torch.full((self.fixed_streams,), -10.0, device=device), routed])

    def read(self, streams: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Per-stream weights form a convex combination; identical streams collapse to their mean.
        batch, seq_len, num_streams, hidden_size = streams.shape
        flat = streams.flatten(2).float()
        pre = torch.softmax(self.pre(flat.to(self.pre.weight.dtype)).float() * self.scale[0], dim=-1).to(streams.dtype)
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2)

        if self.route is not None:
            scores = self.route(flat.to(self.route.weight.dtype))
            top = scores.topk(self.active_streams - self.fixed_streams, dim=-1).indices
            fixed = torch.arange(self.fixed_streams, device=streams.device).expand(batch, seq_len, -1)
            indices = torch.cat([fixed, top + self.fixed_streams], dim=-1)
            route_weights = torch.cat(
                [torch.ones_like(fixed, dtype=scores.dtype), scores.gather(-1, top).sigmoid()], dim=-1
            )
        else:
            indices = torch.arange(self.active_streams, device=streams.device).expand(batch, seq_len, -1)
            route_weights = torch.ones(indices.shape, device=streams.device, dtype=streams.dtype)

        active = streams.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, hidden_size))
        state = {
            "streams": streams,
            "indices": indices,
            "route_weights": route_weights.to(streams.dtype),
            "active": active,
            "flat": flat,
        }
        return collapsed, state

    def write(self, state: dict, output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        active = state["active"]
        batch, seq_len, active_streams, hidden_size = active.shape

        # Routed streams keep their own (learned) time constant; the fixed stream is untouched
        if self.decay_logit is not None:
            decay = 1.0 - torch.sigmoid(self.decay_logit.to(active.dtype))
            if self.fixed_streams > 0:
                decay = torch.cat(
                    [
                        torch.ones(self.fixed_streams, device=active.device, dtype=active.dtype),
                        decay[self.fixed_streams :],
                    ]
                )
            mixed = active * decay[state["indices"]].unsqueeze(-1)
        else:
            mixed = active

        features = [output]
        if self.convs is not None:
            for conv in self.convs:
                conv_input = output.transpose(1, 2).to(conv.weight.dtype)
                features.append(conv(conv_input)[..., :seq_len].transpose(1, 2).to(output.dtype))

        post = torch.sigmoid(self.post(state["flat"].to(self.post.weight.dtype)) * self.scale[1]) * 2.0
        post = post.view(batch, seq_len, self.active_streams, len(features))
        addition = sum(post[..., i].unsqueeze(-1) * feature.unsqueeze(2) for i, feature in enumerate(features))

        # Run only the selected routed stream adapter; decode cost is one adapter per token, not one per stream
        if self.adapters is not None:
            adapter_update = torch.zeros_like(addition)
            for slot in range(self.fixed_streams, self.active_streams):
                selected = state["indices"][:, :, slot] - self.fixed_streams
                for expert_idx, adapter in enumerate(self.adapters):
                    mask = selected == expert_idx
                    if mask.any():
                        update = adapter(output[mask])
                        scale = state["route_weights"][..., slot][mask].unsqueeze(-1)
                        adapter_update[..., slot, :][mask] = update * scale.to(update.dtype)
            addition = addition + adapter_update

        # The fixed streams are the residual anchors: only they carry the block residual
        if self.fixed_streams > 0:
            residual_mask = torch.zeros(self.active_streams, device=active.device, dtype=active.dtype)
            residual_mask[: min(self.fixed_streams, self.active_streams)] = 1.0
            updated = mixed + addition + residual_mask.view(1, 1, -1, 1) * residual.unsqueeze(2)
            residual_streams = state["streams"]
        else:
            updated = mixed + addition
            residual_streams = state["streams"] + residual.unsqueeze(2)
        updated = updated.to(residual_streams.dtype)
        return residual_streams.scatter(2, state["indices"].unsqueeze(-1).expand_as(updated), updated)


class ShensiHyperHead(DeepseekV4HyperHead):
    pass


class ShensiAttentionResidual(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rank = config.attn_res_rank
        self.norm = ShensiUnweightedRMSNorm(config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, self.rank, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.rank, bias=False)
        self.gate_down = nn.Linear(config.hidden_size, self.rank, bias=False)
        self.gate_up = nn.Linear(self.rank, 3 * config.hidden_size, bias=True)
        self.logit_scale = nn.Parameter(torch.empty(1))
        self.residual_gate = nn.Parameter(torch.empty(1))

    def forward(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_2d = prefix.dim() == 2
        if is_2d:
            prefix = prefix.unsqueeze(0)
            delta = delta.unsqueeze(0) if delta is not None else None
            blocks = blocks.unsqueeze(0)
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        plain = prefix + delta if delta is not None else prefix
        state = self.norm(plain)

        gates = self.gate_up(
            self.gate_down(state.to(self.gate_down.weight.dtype)).to(self.gate_up.weight.dtype)
        ).float()
        decay, erase, write = gates.split(self.hidden_size, dim=-1)
        decay = 1.0 + torch.tanh(decay)
        write = 1.0 + torch.tanh(write)
        erase = torch.tanh(erase)
        updated = decay * prefix + write * (delta if delta is not None else 0.0)
        if delta is not None:
            direction = F.normalize(delta, dim=-1)
            updated = updated - erase * direction * torch.einsum("...d,...d->...", direction, updated).unsqueeze(-1)

        routed = torch.zeros_like(updated)
        if num_blocks > 0:
            blk = blocks[..., :num_blocks, :].float()
            query = self.q_proj(state.to(self.q_proj.weight.dtype)).float()
            keys = F.normalize(self.k_proj(self.norm(blk).to(self.k_proj.weight.dtype)).float(), dim=-1)
            logits = torch.einsum("bsr,bsnr->bsn", query, keys) * self.logit_scale.float()
            weights = torch.softmax(logits, dim=-1)
            routed = torch.einsum("bsn,bsnd->bsd", weights, blk)

        branch = updated + routed
        plain_rms = plain.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        branch_rms = branch.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        output_scale = (plain_rms / branch_rms).detach().clamp(max=10.0)
        output = plain + torch.tanh(self.residual_gate.float()) * branch * output_scale
        updated = plain + torch.tanh(self.residual_gate.float()) * updated * output_scale
        if is_2d:
            output = output.squeeze(0)
            updated = updated.squeeze(0)
        return output.to(out_dtype), updated.to(out_dtype)


class ShensiAttnResBank(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.rank = config.attn_res_block_summary_rank
        self.basis = nn.Parameter(torch.empty(self.rank, config.hidden_size))

    def initialize(self, inputs_embeds: torch.Tensor, num_blocks: int) -> torch.Tensor:
        # Create an encoded bank with empty block-summary rows
        return inputs_embeds.new_zeros(*inputs_embeds.shape[:2], num_blocks, self.rank)

    def write(self, blocks: torch.Tensor, block_idx: int, summary: torch.Tensor) -> torch.Tensor:
        # Encode before writing so the bank never holds a full-rank delta.
        summary = self.encode(summary.unsqueeze(-2))
        return blocks.index_copy(
            -2,
            torch.tensor([block_idx - 1], device=summary.device, dtype=torch.long),
            summary,
        )

    def read(self, blocks: torch.Tensor, inputs_embeds: torch.Tensor, num_blocks: int | None = None) -> torch.Tensor:
        # Decode the encoded bank and prepend the exact token embeddings
        if num_blocks is not None:
            blocks = blocks[..., :num_blocks, :]
        return torch.cat([inputs_embeds.unsqueeze(-2), self.decode(blocks)], dim=-2)

    def encode(self, blocks: torch.Tensor) -> torch.Tensor:
        # Accumulate the projection in float32
        return torch.einsum("bsnh,rh->bsnr", blocks.float(), self.basis.float()).to(blocks.dtype)

    def decode(self, blocks: torch.Tensor) -> torch.Tensor:
        # Inverse of `encode`; float32 accumulation keeps the reconstructed summaries stable
        return torch.einsum("bsnr,rh->bsnh", blocks.float(), self.basis.float()).to(blocks.dtype)


class ShensiROSALayer(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.embedding = nn.Embedding(config.vocab_size, config.rosa_embed_dim)
        self.value_proj = nn.Linear(config.rosa_embed_dim, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.hc_mult, bias=True)

    def forward(self, hidden_states: torch.Tensor, rosa_token_ids: torch.LongTensor) -> torch.Tensor:
        # Gate ROSA features per stream using the stream-mean gate.
        rosa_features = self.value_proj(self.embed(rosa_token_ids))
        gate = torch.sigmoid(self.gate_proj(hidden_states.mean(dim=2))).unsqueeze(-1)
        return gate * rosa_features.unsqueeze(2)

    def embed(self, rosa_token_ids: torch.LongTensor) -> torch.Tensor:
        # Map ROSA ids to embeddings; -1 marks missing predictions.
        valid = rosa_token_ids.ge(0)
        embeddings = self.embedding(rosa_token_ids.clamp_min(0))
        return embeddings * valid.unsqueeze(-1)


class ShensiDecoderLayer(DeepseekV4DecoderLayer):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = (
            ShensiAttention(config, layer_idx)
            if config.layer_types[layer_idx] == "indexed_attention"
            else ShensiLinearAttention(config, layer_idx)
        )
        self.mlp = (
            ShensiMLP(config, layer_idx)
            if config.mlp_layer_types[layer_idx] == "dense"
            else ShensiSparseMoeBlock(config, layer_idx)
        )
        self.attn_hc = ShensiHyperConnection(config, with_conv=False)
        self.ffn_hc = ShensiHyperConnection(config, with_conv=True)

        self.is_write = config.layer_types[layer_idx] == "indexed_attention"
        self.is_hash = config.mlp_layer_types[layer_idx] == "dense"
        prev_valid_blocks = sum(
            1 for block_type in config.attn_res_block_layer_types[:layer_idx] if block_type == "block_write_layer"
        )
        self.block_idx = prev_valid_blocks if self.is_write else prev_valid_blocks - 1
        self.self_attention_attn_res = ShensiAttentionResidual(config)
        self.mlp_attn_res = ShensiAttentionResidual(config)
        self.rosa = ShensiROSALayer(config) if layer_idx in config.rosa_layer_ids else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        partial_block: torch.Tensor | None = None,
        blocks: torch.Tensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        rosa_token_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # A write layer first flushes the previous partial block into the bank, then starts a new one
        if self.is_write:
            if self.block_idx > 0:
                blocks = blocks.index_copy(
                    -2,
                    torch.tensor([self.block_idx], device=blocks.device, dtype=torch.long),
                    partial_block.unsqueeze(-2),
                )
            partial_block = torch.zeros_like(partial_block)
        # Visible bank plus the in-progress block; the latter is not written back to the bank yet
        partial_blocks = torch.cat([blocks[..., : 1 + self.block_idx, :], partial_block.unsqueeze(-2)], dim=-2)

        # ROSA injects suffix-automaton predictions before the attention sublayer
        if self.rosa is not None and rosa_token_ids is not None:
            hidden_states = hidden_states + self.rosa(hidden_states, rosa_token_ids)

        # Attention path: collapse streams, run attention, then write back to the routed streams
        collapsed, state = self.attn_hc.read(hidden_states)
        topk_indices = prev_topk_indices
        if self.is_write:
            attn_output, _, topk_indices = self.self_attn(
                self.input_layernorm(collapsed),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                prev_topk_indices=prev_topk_indices,
                input_ids=input_ids,
            )
        else:
            attn_output = self.self_attn(
                self.input_layernorm(collapsed), cache_params=past_key_values, attention_mask=attention_mask
            )
        attn_residual, _ = self.self_attention_attn_res(
            partial_block,
            attn_output,
            partial_blocks,
            num_blocks=1 + self.block_idx + 1,
        )
        # Write only the depth correction; the block prefix stays in the bank.
        attn_residual = attn_residual - (partial_block + attn_output)
        hidden_states = self.attn_hc.write(state, attn_output, attn_residual)
        partial_block = partial_block + attn_output

        # MLP path mirrors the attention path and advances the partial block
        collapsed, state = self.ffn_hc.read(hidden_states)
        mlp_output = (
            self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
            if self.is_hash
            else self.mlp(self.post_attention_layernorm(collapsed))
        )
        partial_blocks = torch.cat([blocks[..., : 1 + self.block_idx, :], partial_block.unsqueeze(-2)], dim=-2)
        mlp_residual, _ = self.mlp_attn_res(
            partial_block,
            mlp_output,
            partial_blocks,
            num_blocks=1 + self.block_idx + 1,
        )
        mlp_residual = mlp_residual - (partial_block + mlp_output)
        hidden_states = self.ffn_hc.write(state, mlp_output, mlp_residual)
        partial_block = partial_block + mlp_output
        return hidden_states, partial_block, blocks, topk_indices


class ShensiStreamExpansion(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.kernels = config.hc_conv_kernels
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    config.hidden_size,
                    config.hidden_size,
                    kernel,
                    groups=config.hidden_size,
                    padding=kernel - 1,
                )
                for kernel in self.kernels
            ]
        )

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        # Stream 0 is the embedding itself; every other stream is a depthwise causal convolution
        streams = [embeds]
        seq_len = embeds.shape[1]
        conv_input = embeds.transpose(1, 2)
        conv_input = conv_input.to(self.convs[0].weight.dtype) if self.convs else conv_input
        for conv in self.convs:
            streams.append(conv(conv_input)[..., :seq_len].transpose(1, 2).to(embeds.dtype))
        return torch.stack(streams, dim=2)


class ShensiStreamCompression(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.rank = config.hc_stream_rank
        self.basis = nn.Parameter(torch.empty(self.rank, config.hidden_size))

    def encode(self, streams: torch.Tensor) -> torch.Tensor:
        # Keep only the coordinates of each stream in the shared basis
        return torch.einsum("bskh,rh->bskr", streams.float(), self.basis.float()).to(streams.dtype)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # Reconstruct full-width streams before they enter a hyper-connection block
        return torch.einsum("bskr,rh->bskh", latents.float(), self.basis.float()).to(latents.dtype)


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
        "k_norm",
        "kv_norm",
        "norm",
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
            if module.kv_emb_down is not None:
                init.ones_(module.kv_emb_down.weight)
                init.copy_(
                    module.kv_emb_up.weight,
                    torch.full_like(module.kv_emb_up.weight, 1.0 / self.config.kv_emb_rank),
                )
        elif isinstance(module, ShensiIndexer):
            eye = torch.eye(max(module.head_dim, module.latent_rank))
            init.copy_(module.down_proj.weight, eye[: module.latent_rank, : module.head_dim])
            init.copy_(module.up_proj.weight, eye[: module.head_dim, : module.latent_rank])
            if module.token_emb is not None:
                init.ones_(module.token_emb.weight)
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
            # A single projection avoids the zero-product dead zone.
            init.zeros_(module.pre.weight)
            init.zeros_(module.post.weight)
            init.zeros_(module.post.bias)
            init.ones_(module.scale)
            if module.route is not None:
                # Non-zero routing breaks the initial top-k tie; its gradient is well-defined
                init.normal_(module.route.weight, mean=0.0, std=std)
            if module.adapters is not None:
                for adapter in module.adapters:
                    init.normal_(adapter.down.weight, mean=0.0, std=std)
                    init.zeros_(adapter.up.weight)
                init.normal_(module.decay_logit, mean=-3.0, std=0.5)
            if module.convs is not None:
                for conv in module.convs:
                    init.zeros_(conv.weight)
        elif isinstance(module, ShensiAttentionResidual):
            init.normal_(module.q_proj.weight, mean=0.0, std=std)
            init.normal_(module.k_proj.weight, mean=0.0, std=std)
            init.normal_(module.gate_down.weight, mean=0.0, std=std)
            init.normal_(module.gate_up.weight, mean=0.0, std=std)
            init.zeros_(module.gate_up.bias)
            init.ones_(module.logit_scale)
            init.constant_(module.residual_gate, 0.1)
        elif isinstance(module, ShensiROSALayer):
            init.zeros_(module.gate_proj.weight)
            init.constant_(module.gate_proj.bias, -4.0)
        elif isinstance(module, ShensiStreamExpansion):
            for conv in module.convs:
                init.zeros_(conv.weight)
        elif isinstance(module, ShensiStreamCompression):
            eye = torch.eye(max(self.config.hidden_size, module.rank))[: module.rank, : self.config.hidden_size]
            init.copy_(module.basis, eye.to(module.basis.device))
        elif isinstance(module, ShensiAttnResBank):
            eye = torch.eye(max(self.config.hidden_size, module.rank))[: module.rank, : self.config.hidden_size]
            init.copy_(module.basis, eye.to(module.basis.device))


class ShensiAttentionCache(DynamicIndexedLayer):
    _layer_type = "indexed_attention"

    def __init__(self, **kwargs):
        super().__init__()
        self.source = None
        self.token_ids: torch.Tensor | None = None
        self.rosa_state = None

    def update(self, key_states, value_states, *args, **kwargs):
        # Cache one latent; FP8 for inference, compute dtype for training.
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

    def reset(self) -> None:
        super().reset()
        self.token_ids = None
        self.rosa_state = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.rosa_state is not None:
            self.rosa_state.reorder_cache(beam_idx)
        if self._sync_token_ids():
            return
        if self.token_ids is not None:
            self.token_ids = self.token_ids.index_select(0, beam_idx.to(self.token_ids.device))

    def crop(self, tokens_to_remove: int) -> None:
        super().crop(tokens_to_remove)
        if tokens_to_remove == 0:
            return
        if self.rosa_state is not None:
            self.rosa_state.crop(tokens_to_remove)
        if self._sync_token_ids():
            return
        if self.token_ids is None:
            return
        self.token_ids = self.token_ids[:, : -abs(tokens_to_remove)]

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.rosa_state is not None:
            self.rosa_state.batch_repeat_interleave(repeats)
        if self._sync_token_ids():
            return
        if self.token_ids is not None:
            self.token_ids = self.token_ids.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.rosa_state is not None:
            self.rosa_state.batch_select_indices(indices)
        if self._sync_token_ids():
            return
        if self.token_ids is not None:
            self.token_ids = self.token_ids[indices, ...]

    def _sync_token_ids(self) -> bool:
        if self.source is not None:
            self.token_ids = self.source.token_ids
            return True
        return False


class ShensiROSAState:
    def __init__(self) -> None:
        self.sequence_states: list[dict] = []

    @classmethod
    def from_token_ids(
        cls,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> "ShensiROSAState":
        # Build a state by replaying a full token sequence
        state = cls()
        state.update(token_ids, attention_mask=attention_mask, eos_token_id=eos_token_id)
        return state

    def update(
        self,
        token_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | None = None,
    ) -> torch.LongTensor:
        # Update the cached automata and return predictions for the new tokens
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
                f"`rosa` state batch size {len(self.sequence_states)} does not match input batch size {batch_size}."
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
        # Clear all cached sequence states
        self.sequence_states = []

    def crop(self, tokens_to_remove: int) -> None:
        # Drop trailing tokens from each sequence state
        if tokens_to_remove == 0:
            return
        for batch_idx, state in enumerate(self.sequence_states):
            # Suffix automata are append-only here, so replay the kept prefix
            tokens = state["tokens"][: -abs(tokens_to_remove)]
            self.sequence_states[batch_idx] = self._new_state()
            for token in tokens:
                self._update_state(self.sequence_states[batch_idx], token)

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        # Reorder sequence states by beam index
        indices = beam_idx.tolist()
        self.sequence_states = [self.sequence_states[index] for index in indices]

    def batch_repeat_interleave(self, repeats: int) -> None:
        # Repeat each sequence state for beam expansion
        self.sequence_states = [self._clone_state(state) for state in self.sequence_states for _ in range(repeats)]

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        # Select sequence states by batch index
        self.sequence_states = [self._clone_state(self.sequence_states[index]) for index in indices.tolist()]

    @staticmethod
    def _new_state() -> dict:
        # Per-state: transitions, suffix link, longest-match length, and last occurrence position
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
        # A new token always creates one new automaton state extending the previous last state
        current_state = state["state_count"]
        state["state_count"] = current_state + 1
        ShensiROSAState._grow_state(state, current_state)
        lengths[current_state] = lengths[state["last_state"]] + 1
        index = state["last_state"]

        # Walk suffix links until a state already has this token as an outgoing transition
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
                # Standard suffix-automaton cloning: split `candidate` at the shorter length
                clone_state = state["state_count"]
                state["state_count"] = clone_state + 1
                ShensiROSAState._grow_state(state, clone_state)
                transitions[clone_state] = transitions[candidate].copy()
                lengths[clone_state] = lengths[index] + 1
                suffix_links[clone_state] = suffix_links[candidate]
                last_positions[clone_state] = last_positions[candidate]

                while index != -1 and transitions[index].get(token) == candidate:
                    transitions[index][token] = clone_state
                    index = suffix_links[index]

                suffix_links[candidate] = clone_state
                suffix_links[current_state] = clone_state

        # The deepest state whose last occurrence has a successor yields the ROSA prediction
        index = state["last_state"] = current_state
        prediction = -1
        while index != -1:
            if lengths[index] > 0 and last_positions[index] >= 0:
                prediction = tokens[last_positions[index] + 1]
                break
            index = suffix_links[index]

        # Record this position as the latest occurrence for the state and all its suffixes
        index = state["last_state"]
        position = state["position"]
        while index != -1 and last_positions[index] < position:
            last_positions[index] = position
            index = suffix_links[index]

        state["position"] = position + 1
        # The boundary token is consumed, then the automaton restarts from the root
        if eos_token_id is not None and token == eos_token_id:
            state.clear()
            state.update(ShensiROSAState._new_state())
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


@torch.no_grad()
def rosa(
    token_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    eos_token_id: int | None = None,
) -> torch.LongTensor:
    # Compute ROSA next-token predictions for every position
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

    predictions = []
    for batch_idx, sequence in enumerate(token_ids.tolist()):
        # The stateless operator builds a fresh automaton per sequence
        state = ShensiROSAState._new_state()
        batch_predictions = []
        for position, token in enumerate(sequence):
            # Padding is skipped entirely so it neither predicts nor changes the automaton
            if attention_mask is not None and not attention_mask[batch_idx][position]:
                batch_predictions.append(-1)
                continue
            batch_predictions.append(ShensiROSAState._update_state(state, token, eos_token_id))
        predictions.append(batch_predictions)

    if not predictions:
        return token_ids.new_full(token_ids.shape, -1)

    outputs = torch.tensor(predictions, dtype=torch.long, device=token_ids.device)
    return outputs.squeeze(0) if is_1d else outputs


class ShensiModel(Glm5NextTextModel):
    config: ShensiConfig

    def __init__(self, config: ShensiConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ShensiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ShensiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = ShensiHyperHead(config)

        self.stream_expansion = ShensiStreamExpansion(config)
        self.stream_compression = ShensiStreamCompression(config)
        self.bank_attn_res = ShensiAttnResBank(config)
        self.output_attn_res = ShensiAttentionResidual(config)

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

        # Both hybrid layer types share the same 2D recurrent mask; the indexer consumes it directly
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            attention_mask = create_recurrent_attention_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            # Guarantee the mask to exist for the indexer
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

        # Expand streams, compress between layers, and seed bank slot 0 with the embedding.
        streams = self.stream_expansion(inputs_embeds)

        num_attn_res_blocks = self.config.attn_res_block_layer_types.count("block_write_layer")
        residual_encoded = self.bank_attn_res.initialize(inputs_embeds, num_attn_res_blocks)
        streams = self.stream_compression.encode(streams)
        prefix_sum = torch.zeros_like(inputs_embeds)

        # ROSA predictions are computed once and reused by every layer that has a ROSA module
        rosa_ids = None
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
                # The first indexed cache owns the incremental ROSA state across decoding steps
                cache_layer = past_key_values.layers[write_layers[0]]
                if hasattr(cache_layer, "rosa_state"):
                    if cache_layer.rosa_state is None:
                        previous_ids = getattr(cache_layer, "token_ids", None)
                        cache_layer.rosa_state = (
                            ShensiROSAState.from_token_ids(previous_ids)
                            if previous_ids is not None
                            else ShensiROSAState()
                        )
                    rosa_ids = cache_layer.rosa_state.update(
                        input_ids,
                        attention_mask=rosa_attention_mask,
                        eos_token_id=reset_token_id,
                    )
            if rosa_ids is None:
                rosa_ids = rosa(input_ids, attention_mask=rosa_attention_mask, eos_token_id=reset_token_id)

        topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            # Decode only at layer boundaries; inside the layer everything stays full-width
            layer_streams = self.stream_compression.decode(streams)
            layer_blocks = self.bank_attn_res.read(residual_encoded, inputs_embeds)

            hidden_states, prefix_sum, residual, topk_indices = decoder_layer(
                layer_streams,
                partial_block=prefix_sum,
                blocks=layer_blocks,
                input_ids=input_ids,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=None,
                prev_topk_indices=topk_indices,
                rosa_token_ids=rosa_ids,
                **kwargs,
            )

            # Re-encode immediately so only compressed state crosses the layer boundary
            streams = self.stream_compression.encode(hidden_states)
            # Row 0 is the exact token embedding and stays outside the low-rank bank
            residual_encoded = self.bank_attn_res.encode(residual[..., 1:, :])

        # Flush the final partial block, then run the depth-wise output residual
        residual_encoded = self.bank_attn_res.write(residual_encoded, num_attn_res_blocks, prefix_sum)
        residual = self.bank_attn_res.read(residual_encoded, inputs_embeds)
        streams = self.stream_compression.decode(streams)
        hidden_states = self.hc_head(streams)
        hidden_states, _ = self.output_attn_res(
            torch.zeros_like(prefix_sum),
            hidden_states,
            residual,
            num_blocks=1 + num_attn_res_blocks,
        )
        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _create_cache(self) -> Cache:
        # Indexed layers carry a custom cache that also stores token ids and, on the owner layer, the ROSA automaton.
        # All indexed layers share the owner's token ids
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
        # MoE layers between two write layers share one router and one expert pool
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

        # Register both module- and parameter-level ties so save/load and TP see the sharing
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

        # The dense deep-embedding table is shared by every hash layer
        hash_layers = [(i, layer) for i, layer in enumerate(self.layers) if getattr(layer, "is_hash", False)]
        if len(hash_layers) > 1:
            first_idx, first_layer = hash_layers[0]
            for idx, layer in hash_layers[1:]:
                layer.mlp.deepemb = first_layer.mlp.deepemb
                source_key = f"layers.{first_idx}.mlp.deepemb"
                target_key = f"layers.{idx}.mlp.deepemb"
                group_patterns[target_key] = source_key
                group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        # The KV factor table, the indexer table and the ROSA table are each shared once
        write_attention_layers = [layer for layer in self.layers if layer.is_write]
        if len(write_attention_layers) > 1:
            first_idx = write_attention_layers[0].layer_idx
            first_attn = write_attention_layers[0].self_attn
            if first_attn.kv_emb_down is not None:
                for layer in write_attention_layers[1:]:
                    layer.self_attn.kv_emb_down = first_attn.kv_emb_down
                    layer.self_attn.kv_emb_up = first_attn.kv_emb_up
                    for attr in ("kv_emb_down", "kv_emb_up"):
                        source_key = f"layers.{first_idx}.self_attn.{attr}"
                        target_key = f"layers.{layer.layer_idx}.self_attn.{attr}"
                        group_patterns[target_key] = source_key
                        group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        full_indexers = [layer for layer in write_attention_layers if layer.self_attn.indexer is not None]
        if len(full_indexers) > 1:
            first_idx = full_indexers[0].layer_idx
            first_emb = full_indexers[0].self_attn.indexer.token_emb
            if first_emb is not None:
                for layer in full_indexers[1:]:
                    layer.self_attn.indexer.token_emb = first_emb
                    source_key = f"layers.{first_idx}.self_attn.indexer.token_emb"
                    target_key = f"layers.{layer.layer_idx}.self_attn.indexer.token_emb"
                    group_patterns[target_key] = source_key
                    group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        rosa_layers = [(layer.layer_idx, layer.rosa) for layer in self.layers if layer.rosa is not None]
        if rosa_layers and self.config.rosa_embed_dim == self.config.hidden_size:
            # ROSA predicts token ids, so it can reuse the input token embedding table
            for layer_idx, rosa in rosa_layers:
                rosa.embedding.weight = self.embed_tokens.weight
                target_key = f"layers.{layer_idx}.rosa.embedding"
                group_patterns[target_key] = "embed_tokens"
                group_tied_weights[f"{target_key}.weight"] = "embed_tokens.weight"
        elif len(rosa_layers) > 1:
            first_idx, first_rosa = rosa_layers[0]
            for layer_idx, rosa in rosa_layers[1:]:
                rosa.embedding = first_rosa.embedding
                source_key = f"layers.{first_idx}.rosa.embedding"
                target_key = f"layers.{layer_idx}.rosa.embedding"
                group_patterns[target_key] = source_key
                group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        self._tied_weights_keys = {**dict(self._tied_weights_keys or {}), **group_patterns}
        self._moe_group_tied_weights = group_tied_weights

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        resized = {}

        # Every tied vocab table must be resized exactly once and rebound to its layers
        def resize(table):
            if id(table) not in resized:
                resized[id(table)] = self._get_resized_embeddings(
                    table, new_num_tokens, pad_to_multiple_of, mean_resizing
                )
            return resized[id(table)]

        for layer in self.layers:
            if layer.is_hash and layer.mlp.deepemb is not None:
                layer.mlp.deepemb = resize(layer.mlp.deepemb)
            if layer.is_write:
                if layer.self_attn.kv_emb_down is not None:
                    layer.self_attn.kv_emb_down = resize(layer.self_attn.kv_emb_down)
                indexer = layer.self_attn.indexer
                if indexer is not None and indexer.token_emb is not None:
                    indexer.token_emb = resize(indexer.token_emb)
            if layer.rosa is not None:
                if self.config.rosa_embed_dim == self.config.hidden_size:
                    # Rebind to the already resized input embedding table
                    layer.rosa.embedding.weight = self.embed_tokens.weight
                else:
                    layer.rosa.embedding = resize(layer.rosa.embedding)
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
    # Penalize routing coupling between experts
    R = router_weight
    norm_R = torch.norm(R, dim=1)
    # Perturbation radius is half the distance to the nearest expert
    distances = torch.cdist(R, R, p=2)
    distances = distances.masked_fill(torch.eye(R.size(0), dtype=torch.bool, device=distances.device), float("inf"))
    min_dist, _ = torch.min(distances, dim=1)
    eps = min_dist / 2 / norm_R

    # Sample a relative perturbation per expert rather than a fixed direction
    low = (1 - eps).unsqueeze(1)
    high = (1 + eps).unsqueeze(1)
    noise = torch.rand_like(R)
    R_tilde = (low + noise * (high - low)) * R

    # Coupling[i, j] measures how much expert i's router row activates expert j's subspace
    proxy = F.linear(R_tilde, down_proj_weight)
    coupling = torch.norm(torch.einsum("jDd,id->ijD", gate_up_proj, proxy), dim=-1)

    # Only off-diagonal couplings above the self-coupling threshold contribute
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

        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
            # ERC is computed once per tied expert group; layers sharing a pool must not double count
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
