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
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    MoeModelOutputWithPast,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear, repeat_kv
from ..glm5_next.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig
from ..glm5_next.image_processing_glm5_next import Glm5NextImageProcessor, Glm5NextImageProcessorKwargs
from ..glm5_next.modeling_glm5_next import (
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    Glm5NextPreTrainedModel,
    Glm5NextRMSNorm,
    Glm5NextTextAttention,
    Glm5NextTextExperts,
    Glm5NextTextForgetGate,
    Glm5NextTextLinearAttention,
    Glm5NextTextMLP,
    Glm5NextTextMoE,
    Glm5NextTextRMSNorm,
    Glm5NextTextRMSNormGated,
    Glm5NextTextTopkRouter,
    Glm5NextTextUnweightedRMSNorm,
    Glm5NextVisionAttention,
    Glm5NextVisionBlock,
    Glm5NextVisionMLP,
    Glm5NextVisionModel,
    Glm5NextVisionPatchEmbed,
    Glm5NextVisionPatchMerger,
    Glm5NextVisionRotaryEmbedding,
)
from ..glm5_next.processing_glm5_next import Glm5NextProcessor
from ..glm5_next.video_processing_glm5_next import Glm5NextVideoProcessor, Glm5NextVideoProcessorInitKwargs
from ..glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaForCausalLM,
    GlmMoeDsaIndexer,
    GlmMoeDsaModel,
)


class ShensiTextConfig(Glm5NextTextConfig):
    # indexer
    index_kpool = AttributeError()
    index_kpool_always_select_tail = AttributeError()
    index_k_deepemb_rank: int = 32
    # attention
    kv_lora_rank = AttributeError()
    num_key_value_heads: int = 1
    kv_deepemb_rank: int = 32
    o_groups: int = 8
    o_lora_rank: int = 1024

    def validate_architecture(self):
        raise AttributeError()


class ShensiVisionConfig(Glm5NextVisionConfig):
    pass


class ShensiConfig(Glm5NextConfig):
    pass


class ShensiTextCache(DynamicCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_input_ids: torch.LongTensor | None = None

    def update_input_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """
        Update the token-id history by concatenation, and return the full token-id history.

        Args:
            input_ids (`torch.Tensor`): New token ids, shape `[batch_size, seq_len]`.

        Returns:
            `torch.Tensor`: The full cached token ids, shape `[batch_size, kv_length]`.
        """
        self.past_input_ids = torch.cat([self.past_input_ids, input_ids], dim=1)
        return self.past_input_ids

    def reset(self) -> None:
        super().reset()
        self.past_input_ids = None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        if self.past_input_ids is not None:
            self.past_input_ids = self.past_input_ids.index_select(0, beam_idx.to(self.past_input_ids.device))

    def crop(self, tokens_to_remove: int) -> None:
        super().crop(tokens_to_remove)
        if tokens_to_remove == 0:
            return
        if self.past_input_ids is not None:
            self.past_input_ids = self.past_input_ids[..., : -abs(tokens_to_remove)]

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if self.past_input_ids is not None:
            self.past_input_ids = self.past_input_ids.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        if self.past_input_ids is not None:
            self.past_input_ids = self.past_input_ids[indices, ...]


class ShensiTextRMSNorm(Glm5NextTextRMSNorm):
    pass


class ShensiTextUnweightedRMSNorm(Glm5NextTextUnweightedRMSNorm):
    pass


class ShensiTextRMSNormGated(Glm5NextTextRMSNormGated):
    pass


def force_self_selection(index_scores: torch.Tensor) -> torch.Tensor:
    r"""
    Forces each query's own position into its top-k selection, keeping the self term that
    `exclusive_self_attention` subtracts.

    See DeepSeek Sparse Attention (https://arxiv.org/abs/2512.02556) for more details on the indexer, and
    Exclusive Self Attention (https://arxiv.org/abs/2603.09078) for why the self term is required: sparse
    selection does not guarantee a query selects itself, so every self-position score is overwritten with
    torch.finfo(dtype).max, the strict row maximum that top-k always keeps. The guarantee is unconditional
    and covers prefill, chunked prefill, and decoding in one code path.

    Args:
        index_scores (`torch.Tensor` of shape [batch_size, q_length, kv_length]):
            Selection scores after causality and padding masking; `kv_length` must be at least `q_length`
            (always true under causal attention). The query at local position i corresponds to the global
            KV position (kv_length - q_length) + i, i.e. the diagonal at offset (kv_length - q_length).
            Rows belonging to padding queries are also overwritten; their outputs are discarded downstream.

    Returns:
        The same scores with every self position overwritten, ready for top-k.
    """
    q_length = index_scores.shape[-2]
    kv_length = index_scores.shape[-1]

    # diagonal() returns a view into index_scores, so fill_ writes the self
    # positions in place instead of cloning a [B, q_length, kv_length]
    # tensor. Callers must pass a writable tensor (the fresh output of
    # masked_fill or addition) and run under torch.no_grad(), since the
    # in-place fill would otherwise be rejected by autograd.
    index_scores.diagonal(offset=kv_length - q_length, dim1=-2, dim2=-1).fill_(torch.finfo(index_scores.dtype).max)
    return index_scores


class ShensiTextIndexer(GlmMoeDsaIndexer):
    """
    DeepSeek Sparse Attention (DSA) indexer with extremely small keys cache for Shensi-V1.

    **Cache strategy**: the indexer key cache lives on the per-layer `DynamicIndexedLayer` (or the
    `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `cache.update_indexer()`.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        del self.config
        del self.hidden_size
        del self.qk_rope_head_dim
        del self.wq_b
        del self.wk
        del self.k_norm
        del self.weights_proj

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        # Refer: https://rwkv.cn/news/read?id=20250630 for more details.
        self.wk_a = nn.Linear(config.hidden_size, config.index_k_deepemb_rank, bias=False)
        self.wk_b = nn.Linear(config.index_k_deepemb_rank, self.head_dim, bias=False)
        self.k_emb = nn.Embedding(config.vocab_size, self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        cache: Cache | None,
    ) -> torch.LongTensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            input_ids: Indices of input sequence tokens in the vocabulary, shape `(batch_size, num_choices, sequence_length)`.
            attention_mask: Local boolean padding mask of shape `[B, S]`.
            cache: Cache object containing the indexer state cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]` (or `[B, S, 2*topk - 1]` with tail).
            The eager / SDPA paths turn these into an additive sparse mask.
        """
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        q = self.wq_b(q_resid).view(hidden_shape)
        k_latent = self.wk_a(hidden_states)

        if cache is not None:
            k_latent = cache.update_indexer(k_latent, self.layer_idx)

        k = self.k_norm(self.wk_b(k_latent) * self.k_emb(input_ids))

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
        if attention_mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
        else:
            index_scores = index_scores + attention_mask

        index_scores = force_self_selection(index_scores)
        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]


class ShensiTextGroupedLinear(DeepseekV4GroupedLinear):
    """Block-diagonal grouped linear used by the grouped output projection"""

    pass


def eager_text_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = F.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def exclusive_self_attention(
    attn_output: torch.Tensor,
    v_full: torch.Tensor,
) -> torch.Tensor:
    r"""
    Removes the component of each attention output along its own value vector, as described in Exclusive
    Self Attention.

    See Exclusive Self Attention (https://arxiv.org/abs/2603.09078) for more details. This function implements
    z_i = y_i - (y_i . v_i / ||v_i||^2) * v_i, re-projecting each attention output into the subspace
    orthogonal to the current token, so attention only aggregates information the residual stream cannot
    already supply.

    Args:
        attn_output (`torch.Tensor` of shape [batch_size, num_heads, q_length, v_head_dim]):
            Attention outputs for the current step; `q_length` is 1 during decoding.
        v_full (`torch.Tensor` of shape [batch_size, num_kv_heads, kv_length, v_head_dim]):
            Full value sequence including any cached prefix; `kv_length` must be at least `q_length`
            (always true under causal attention), and the last `q_length` positions are aligned with the
            current queries.

    Returns:
        The re-projected attention outputs, same shape and dtype as `attn_output`.
    """
    num_heads, q_length, head_dim = attn_output.shape[1:]
    num_kv_heads, kv_length = v_full.shape[1], v_full.shape[2]

    # The current queries are always the last q_length positions of the KV
    # sequence, so this slice is each query's own value vector.
    v_self = v_full[:, :, kv_length - q_length :, :]

    # Broadcast shared KV heads to the number of query heads; no-op when
    # they already match.
    if num_kv_heads != num_heads:
        v_self = v_self.repeat_interleave(num_heads // num_kv_heads, dim=1)

    # Cast to the output dtype to support mixed-precision caches. The floor
    # is dtype-derived instead of a fixed constant: exactly-zero value rows
    # normalize to zero, so nothing is subtracted for them.
    v_hat = F.normalize(
        v_self.to(attn_output.dtype),
        dim=-1,
        eps=torch.finfo(attn_output.dtype).smallest_normal,
    )

    # normalize already divides by ||v||, so the projection formula needs
    # no separate normalization step.
    return attn_output - ((attn_output * v_hat).sum(dim=-1, keepdim=True) * v_hat)


class ShensiTextAttention(Glm5NextTextAttention):
    """
    DeepSeek Sparse Attention (DSA) extended with **extremely small KV cache** for Shensi-V1.

    Unique architectures:
    - Shared-KV Multi-Query Attention.
    - Per-head learnable attention sink like gpt_oss.
    - Value-orthogonal attention constraint (excluding self position).
    - Head-specific sigmoid gate after the `attention_interface`.
    - Grouped low-rank output projection for perfs.

    `config.indexer_types[layer_idx]` decides whether this layer runs its own indexer (`"full"`) or reuses the previous
    full layer's top-k selection (`"shared"`). `next_skip_topk` signals that the *next* layer will reuse this layer's
    top-k, so it is propagated upward via `prev_topk_indices`.
    """

    def __init__(self, config: ShensiTextConfig, layer_idx: int):
        super().__init__()
        del self.kv_lora_rank
        del self.q_a_proj
        del self.q_a_layernorm
        del self.q_b_proj
        del self.kv_a_proj_with_mqa
        del self.kv_a_layernorm
        del self.kv_b_proj
        del self.o_proj

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_norm = ShensiTextRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * (self.qk_head_dim + self.v_head_dim), bias=False)
        self.q_b_norm = ShensiTextUnweightedRMSNorm(eps=config.rms_norm_eps)

        # Refer: https://rwkv.cn/news/read?id=20250630 for more details.
        self.kv_deepemb_rank = config.kv_deepemb_rank
        self.kv_a_proj = nn.Linear(self.hidden_size, self.kv_deepemb_rank, bias=False)
        self.kv_b_proj = nn.Linear(self.kv_deepemb_rank, self.num_heads * self.v_head_dim, bias=False)
        self.kv_emb = nn.Embedding(config.vocab_size, self.v_head_dim)
        self.kv_norm = ShensiTextRMSNorm(self.v_head_dim, eps=config.rms_norm_eps)

        self.o_a_proj = ShensiTextGroupedLinear(
            self.num_heads * self.v_head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, self.hidden_size, bias=False)

        self.sinks = nn.Parameter(torch.empty(self.num_heads))

    def expand_kv(self):
        raise AttributeError()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        cache: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        q_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        kv_shape = (batch_size, -1, self.num_heads, self.v_head_dim)

        # LoRA based path is guaranteed based on the config validation
        q_resid = self.q_a_norm(self.q_a_proj(hidden_states))
        q, gate = torch.split(
            self.q_b_proj(q_resid),
            [self.num_heads * self.qk_head_dim, self.num_heads * self.v_head_dim],
            dim=-1,
        )
        q = self.q_b_norm(q.view(q_shape).transpose(1, 2))
        gate = gate.view(batch_size, seq_length, self.num_heads, self.v_head_dim)

        kv_latent = self.kv_a_proj(hidden_states).view(batch_size, 1, seq_length, self.kv_deepemb_rank)

        # Cache read / write is performed while latent KV is still compressed
        if cache is not None:
            kv_latent = cache.update(kv_latent, kv_latent, self.layer_idx, kwargs=kwargs)[0]

        kv = self.kv_norm(self.kv_b_proj(kv_latent).view(kv_shape) * self.kv_emb(input_ids).unsqueeze(-2))
        kv = kv.transpose(1, 2)

        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states=hidden_states,
                q_resid=q_resid,
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache=cache,
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

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_text_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = exclusive_self_attention(attn_output, kv)
        attn_output = attn_output.reshape(batch_size, seq_length, self.num_heads, self.v_head_dim).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        grouped = attn_output.reshape(batch_size, seq_length, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights, topk_indices if self.next_skip_topk else None


class ShensiTextForgetGate(Glm5NextTextForgetGate):
    pass


class ShensiTextLinearAttention(Glm5NextTextLinearAttention):
    pass


class ShensiTextMLP(Glm5NextTextMLP):
    pass


class ShensiTextTopkRouter(Glm5NextTextTopkRouter):
    pass


class ShensiTextExperts(Glm5NextTextExperts):
    pass


class ShensiTextMoE(Glm5NextTextMoE):
    pass


class ShensiTextDecoderLayer(GlmMoeDsaDecoderLayer):
    def __init__(self, config: ShensiTextConfig, layer_idx: int):
        self.block_type = config.layer_types[layer_idx]

        super().__init__(config, layer_idx)
        self.self_attn = (
            ShensiTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else ShensiTextAttention(config, layer_idx)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        topk_indices = None
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache=cache,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _, topk_indices = self.self_attn(
                hidden_states=hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids[:, -hidden_states.shape[1] :])
        hidden_states = residual + hidden_states
        return hidden_states, topk_indices


class ShensiPreTrainedModel(Glm5NextPreTrainedModel):
    _keep_in_fp32_modules_strict = [
        "e_score_correction_bias",
        "gate.weight",
        "conv1d",
        "dt_bias",
        "A_log",
    ]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, ShensiTextForgetGate):
            # Following FLA initialization
            # NOTE: This is incredibly important so keep it this way at all costs
            if module.safe_gate_lower_bound is not None:
                init.zeros_(module.A_log)
            else:
                init.copy_(
                    module.A_log,
                    init.uniform_(module.A_log, a=1.0, b=16.0).log(),
                )
            init.uniform_(
                module.dt_bias,
                a=math.log(1e-3),
                b=math.log(1e-1),
            )
            dt = module.dt_bias.exp().clamp_min(1e-4)
            # (stable) inverse softplus
            init.copy_(
                module.dt_bias,
                dt + torch.log(-torch.expm1(-dt)),
            )
        elif isinstance(module, ShensiTextRMSNormGated):
            init.ones_(module.weight)
        elif isinstance(module, ShensiTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, ShensiTextTopkRouter):
            init.zeros_(module.e_score_correction_bias)
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, ShensiTextIndexer):
            init.zeros_(module.index_kpool_compress_ape)
            init.ones_(module.index_kpool_compress_gate)


class ShensiTextModel(GlmMoeDsaModel, ShensiPreTrainedModel):
    def __init__(self, config: ShensiTextConfig):
        super().__init__(config)
        del self.rotary_emb

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if input_ids is None:
            raise ValueError("You must specify exactly input_ids")

        if use_cache and cache is None:
            cache = ShensiTextCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache is not None:
            if cache.past_input_ids is None:
                cache.past_input_ids = input_ids.clone()
            else:
                input_ids = cache.update_input_ids(input_ids)

        past_seen = cache.get_seq_length() if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            linear_mask = create_recurrent_attention_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache=cache,
            )

            kv_length = past_seen + inputs_embeds.shape[1]
            pos = torch.arange(kv_length, device=inputs_embeds.device)
            causal = pos <= pos[past_seen:, None]
            if self.padding_idx is not None:
                indexed_mask = causal[None] & input_ids.ne(self.padding_idx)[:, None]
            else:
                indexed_mask = causal[None]

            causal_mask_mapping = {
                "indexed_attention": indexed_mask,
                "linear_attention": linear_mask,
            }

        hidden_states = inputs_embeds

        topk_indices = None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                # Key change using NoPE
                position_embeddings=None,
                input_ids=input_ids,
                cache=cache,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache,
        )


class ShensiForCausalLM(GlmMoeDsaForCausalLM, ShensiPreTrainedModel):
    pass


class ShensiRMSNorm(Glm5NextRMSNorm):
    pass


class ShensiVisionPatchEmbed(Glm5NextVisionPatchEmbed):
    pass


class ShensiVisionRotaryEmbedding(Glm5NextVisionRotaryEmbedding):
    pass


class ShensiVisionAttention(Glm5NextVisionAttention):
    pass


class ShensiVisionMLP(Glm5NextVisionMLP):
    pass


class ShensiVisionBlock(Glm5NextVisionBlock):
    pass


class ShensiVisionPatchMerger(Glm5NextVisionPatchMerger):
    pass


class ShensiVisionModel(Glm5NextVisionModel):
    pass


class ShensiModel(Glm5NextModel):
    pass


class ShensiForConditionalGeneration(Glm5NextForConditionalGeneration):
    pass


class ShensiImageProcessorKwargs(Glm5NextImageProcessorKwargs):
    pass


class ShensiImageProcessor(Glm5NextImageProcessor):
    pass


class ShensiVideoProcessorInitKwargs(Glm5NextVideoProcessorInitKwargs):
    pass


class ShensiVideoProcessor(Glm5NextVideoProcessor):
    pass


class ShensiProcessor(Glm5NextProcessor):
    pass


__all__ = [
    "ShensiConfig",
    "ShensiTextConfig",
    "ShensiVisionConfig",
    "ShensiPreTrainedModel",
    "ShensiTextModel",
    "ShensiVisionModel",
    "ShensiModel",
    "ShensiForConditionalGeneration",
    "ShensiImageProcessor",
    "ShensiVideoProcessor",
    "ShensiProcessor",
]
