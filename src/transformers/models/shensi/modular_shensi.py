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
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4CSACache,
    DeepseekV4CSACompressor,
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4GroupedLinear,
    DeepseekV4HCACache,
    DeepseekV4HCACompressor,
    DeepseekV4Indexer,
    DeepseekV4IndexerScorer,
    DeepseekV4MLP,
    DeepseekV4Model,
    DeepseekV4PreTrainedModel,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4TopKRouter,
    DeepseekV4UnweightedRMSNorm,
    load_balancing_loss_func,
)


@auto_docstring(checkpoint="louzongzhi/Shensi-V1-N")
@strict
class ShensiConfig(DeepseekV4Config):
    r"""
    scoring_func (`str`, *optional*):
        Router activation; one of `sqrtsoftplus`, `softmax`, `sigmoid`.
    rope_theta (`float`, *optional*):
        RoPE base of the main attention.
    layer_types (`list[str]`, *optional*):
        Per-layer attention schedule with values from `sliding_window`,
        `compressed_sparse_attention` and `heavily_compressed_attention`; defaults to two
        sliding-window layers followed by interleaved CSA / HCA pairs.
    compress_rates (`dict[str, int]`, *optional*):
        Compression ratio of each attention layer type: `4` for CSA, `128` for HCA.
    compress_rope_theta (`float`, *optional*):
        RoPE base of the compressed branches (paired with `rope_scaling` for YaRN).
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer feed-forward schedule with values from `hash_moe` (token-id routed) and `moe`
        (learned router); defaults to three hash-MoE layers followed by MoE layers.
    swiglu_limit (`float`, *optional*):
        Clamp limit of the routed experts' gate and up pre-activations.
    sliding_window (`int`, *optional*):
        Window of the sliding-window attention branch.
    o_groups (`int`, *optional*):
        Head groups of the grouped output projection.
    o_lora_rank (`int`, *optional*):
        Per-group intermediate dimension of the grouped output projection.
    index_n_heads (`int`, *optional*):
        Query heads of the DSA lightning indexer.
    index_head_dim (`int`, *optional*):
        Indexer head dimension.
    index_topk (`int`, *optional*):
        Compressed entries the indexer keeps per query.
    num_nextn_predict_layers (`int`, *optional*):
        Multi-token-prediction depths carried by the checkpoint.
    partial_rotary_factor (`float`, *optional*):
        Fraction of `head_dim` that gets RoPE; defaults to `qk_rope_head_dim / head_dim`.
    routed_expert_hidden_size (`int`, *optional*):
        Rank of the routed experts' low-rank bottleneck.
    routed_scaling_factor (`float`, *optional*):
        Scale applied to the routed-expert output.
    norm_topk_prob (`bool`, *optional*):
        Whether the top-k router weights are normalized.
    erc_loss_alpha (`float`, *optional*):
        Anchor coefficient of the expert-router coupling loss.
    erc_loss_coef (`float`, *optional*):
        Weight of the expert-router coupling loss.
    attn_res_block_size (`int`, *optional*):
        every `attn_res_block_size` layers form a block whose history the later layers read.
    """

    hidden_size: int = 1280
    moe_intermediate_size: int = 800
    num_hidden_layers: int = 19
    num_attention_heads: int = 16
    q_lora_rank: int = 320
    n_shared_experts = AttributeError()
    routed_expert_hidden_size: int = 320

    hc_mult = AttributeError()
    hc_sinkhorn_iters = AttributeError()
    hc_eps = AttributeError()
    o_groups: int = 2
    o_lora_rank: int = 320

    # ERC
    erc_loss_coef: float = 1.0
    erc_loss_alpha: float = 0.5

    # AttnRes
    attn_res_block_size: int = 2

    @property
    def attn_res_block_layer_types(self) -> list[str]:
        n_hash = self.mlp_layer_types.count("hash_moe")
        return [
            "block_write_layer"
            if i == 0 or (i >= n_hash and (i - n_hash) % self.attn_res_block_size == 0)
            else "block_read_layer"
            for i in range(self.num_hidden_layers)
        ]


class ShensiRMSNorm(DeepseekV4RMSNorm):
    pass


class ShensiUnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    pass


class ShensiRotaryEmbedding(DeepseekV4RotaryEmbedding):
    pass


class ShensiHCACache(DeepseekV4HCACache):
    pass


class ShensiCSACache(DeepseekV4CSACache):
    pass


class ShensiGroupedLinear(DeepseekV4GroupedLinear):
    pass


class ShensiHCACompressor(DeepseekV4HCACompressor):
    pass


class ShensiIndexerScorer(DeepseekV4IndexerScorer):
    pass


class ShensiIndexer(DeepseekV4Indexer):
    pass


class ShensiCSACompressor(DeepseekV4CSACompressor):
    pass


class ShensiAttention(DeepseekV4Attention):
    pass


class ShensiHashMLP(DeepseekV4MLP):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.intermediate_size = config.routed_expert_hidden_size
        self.deepemb = (
            nn.Embedding(config.vocab_size, self.intermediate_size)
            if layer_idx == config.mlp_layer_types.index("hash_moe")
            else None
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.limit)
        up = self.up_proj(hidden_states).clamp(min=-self.limit, max=self.limit)
        return self.down_proj((self.act_fn(gate) * up) * self.deepemb(input_ids))


class ShensiTopKRouter(DeepseekV4TopKRouter):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        del self.e_score_correction_bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(scores, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class ShensiExperts(DeepseekV4Experts):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.hidden_dim = config.routed_expert_hidden_size


class ShensiSparseMoeBlock(nn.Module):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.is_block_write_layer = config.attn_res_block_layer_types[layer_idx] == "block_write_layer"
        self.gate = ShensiTopKRouter(config) if self.is_block_write_layer else None
        self.experts = ShensiExperts(config) if self.is_block_write_layer else None
        self.routed_expert_down_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, config.mlp_bias)
        self.routed_expert_norm = ShensiRMSNorm(config.routed_expert_hidden_size, config.rms_norm_eps)
        self.routed_expert_up_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _, weights, indices = self.gate(hidden_states)
        routed = self.experts(self.routed_expert_down_proj(flat), indices, weights)
        return self.routed_expert_up_proj(self.routed_expert_norm(routed)).view(batch, seq_len, hidden_dim)


class ShensiAttentionResidual(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.norm = ShensiUnweightedRMSNorm(config.rms_norm_eps)
        self.q_a_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=False)
        self.q_b_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, bias=False)
        self.k_a_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=False)
        self.k_b_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, bias=False)
        self.g_a_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=True)
        self.g_b_proj = nn.Linear(config.routed_expert_hidden_size, 3 * config.hidden_size, bias=True)
        self.g_scale = nn.Parameter(torch.zeros(4))
        self.t = nn.Parameter(torch.linspace(0.0, 1.0, config.hidden_size) * math.log(2.0 * config.num_hidden_layers))

    def forward(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        state = self.norm(prefix + (delta if delta is not None else 0.0))

        # Write path: decay, erase and write gates predicted from the normalized stream state.
        decay_scale, erase_scale, write_scale, read_scale = self.g_scale.unbind()
        # Project the decay scale onto the non-negative axis while passing the gradient through, so
        # the gate is not frozen when it reaches the boundary.
        decay_scale = decay_scale + (decay_scale.clamp(min=0.0) - decay_scale).detach()
        r_decay, r_erase, r_write = (
            F.linear(
                F.linear(state, self.g_a_proj.weight.float(), self.g_a_proj.bias.float()),
                self.g_b_proj.weight.float(),
                self.g_b_proj.bias.float(),
            )
            .reshape(*state.shape[:-1], 3, -1)
            .unbind(-2)
        )
        decay = torch.exp(F.softplus(r_decay) * (-decay_scale * self.t.exp()))
        erase = F.softplus(r_erase) * erase_scale
        write = 1.0 + torch.tanh(r_write) * write_scale

        # Normalized write direction; `lam` clamps the erase step factor to (-1, 1), a contraction.
        khat = F.normalize(
            F.linear(
                F.linear(delta if delta is not None else state, self.k_a_proj.weight.float()),
                self.k_b_proj.weight.float(),
            ),
            dim=-1,
        )
        m = torch.addcmul(decay * prefix, write, delta) if delta is not None else decay * prefix
        lam = erase.mean(dim=-1, keepdim=True).clamp(min=-0.5)
        updated = torch.addcmul(
            m, khat, torch.einsum("...d,...d->...", khat, m).unsqueeze(-1) * lam / (1.0 + lam), value=-1.0
        )

        # Read path: score the bank rows and the stream with a diagonally whitened query and mix them
        # with a softmax. With no valid block there is nothing to read.
        if num_blocks > 0:
            blk = blocks[..., :num_blocks, :]
            H = updated.shape[-1]
            query = F.linear(F.linear(state, self.q_a_proj.weight.float()), self.q_b_proj.weight.float())

            # Per-coordinate second moments of the bank rows, shrunk toward one by the evidence weight
            # H / (num_blocks + H); the clamp keeps dead coordinates finite.
            m2 = blk.float().square().mean(-2)
            s = torch.lerp(m2, m2.new_ones(()), H / (H + num_blocks)).clamp_min(1e-20).rsqrt()
            q = s * query

            # One score per bank row, plus the current stream as the last entry.
            logits = torch.cat(
                (
                    (blk @ q.to(out_dtype).unsqueeze(-1)).squeeze(-1).float(),
                    torch.einsum("...d,...d->...", updated, q).unsqueeze(-1),
                ),
                dim=-1,
            )
            # The weights sum to at most one, so the read-out stays below the residual: the stream's
            # own weight is additionally scaled by sigmoid(logsumexp) * read_scale.
            w = torch.softmax(logits, dim=-1)
            g = torch.sigmoid(torch.logsumexp(logits, dim=-1, keepdim=True)) * read_scale
            routed = torch.addcmul(
                (w[..., :num_blocks].to(out_dtype).unsqueeze(-2) @ blk).squeeze(-2).float(),
                w[..., num_blocks:] * g,
                updated,
            )
        else:
            routed = torch.zeros_like(updated)

        output = updated + routed
        if output_norm_weight is not None:
            output = (
                output
                * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + self.norm.eps)
                * output_norm_weight.float()
            )

        return output.to(out_dtype), updated.to(out_dtype)


class ShensiDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.self_attn = ShensiAttention(config, layer_idx)
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        self.mlp = ShensiHashMLP(config, layer_idx) if self.is_hash else ShensiSparseMoeBlock(config, layer_idx)
        self.input_layernorm = ShensiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ShensiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_block_write_layer = config.attn_res_block_layer_types[layer_idx] == "block_write_layer"
        self.prev_valid_blocks = sum(
            1 for r in config.attn_res_block_layer_types[:layer_idx] if r == "block_write_layer"
        )
        self.self_attention_attn_res = ShensiAttentionResidual(config)
        self.mlp_attn_res = ShensiAttentionResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
        input_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # `prefix_sum` accumulates the block's residual stream; `delta` is this layer's contribution.
        delta = hidden_states - prefix_sum if hidden_states is not None else None

        if self.is_block_write_layer:
            written = prefix_sum.to(residual.dtype).unsqueeze(-2)
            residual = torch.cat(
                [
                    residual[..., : self.prev_valid_blocks, :],
                    written,
                    residual[..., self.prev_valid_blocks + 1 :, :],
                ],
                dim=-2,
            )

        hidden_states, prefix_sum = self.self_attention_attn_res(
            prefix_sum,
            delta,
            residual,
            output_norm_weight=self.input_layernorm.weight,
            num_blocks=self.prev_valid_blocks,
        )
        if self.is_block_write_layer:
            prefix_sum = None

        attn_output, _ = self.self_attn(hidden_states, **kwargs)
        hidden_states = hidden_states + attn_output

        if prefix_sum is None:
            prefix_sum = hidden_states
        else:
            prefix_sum = prefix_sum + hidden_states

        hidden_states, prefix_sum = self.mlp_attn_res(
            prefix_sum,
            prefix_sum,
            residual,
            output_norm_weight=self.post_attention_layernorm.weight,
            num_blocks=self.prev_valid_blocks + self.is_block_write_layer,
        )

        mlp_output = self.mlp(hidden_states, input_ids=input_ids) if self.is_hash else self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output

        prefix_sum = prefix_sum + hidden_states
        return hidden_states, prefix_sum, residual


class ShensiPreTrainedModel(DeepseekV4PreTrainedModel):
    _keep_in_fp32_modules_strict = [
        "sinks",
        "position_bias",
        "q_a_norm",
        "kv_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, ShensiHashMLP):
            # Deep embedding: the per-token scale of the routed-expert bottleneck, starting at one.
            # Non-owner hash layers carry the owner's table after `tie_mlp`, so this writes it once
            # per distinct table.
            if module.deepemb is not None:
                init.ones_(module.deepemb.weight)
        elif isinstance(module, ShensiTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, ShensiExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, ShensiAttention):
            init.zeros_(module.sinks)
        elif isinstance(module, (ShensiHCACompressor, ShensiCSACompressor, ShensiIndexer)):
            init.zeros_(module.position_bias)
        elif isinstance(module, ShensiRotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, ShensiAttentionResidual):
            d = self.config.hidden_size
            init.zeros_(module.g_a_proj.weight)
            init.zeros_(module.g_b_proj.weight)
            init.zeros_(module.g_scale)
            with torch.no_grad():
                bias = module.g_b_proj.bias
                bias[:d] = 2.0
                bias[d : 2 * d] = -2.0
                bias[2 * d :] = -2.0
            init.zeros_(module.q_a_proj.weight)
            init.zeros_(module.q_b_proj.weight)
            init.kaiming_normal_(module.k_a_proj.weight, mode="fan_in", nonlinearity="linear")
            init.kaiming_normal_(module.k_b_proj.weight, mode="fan_in", nonlinearity="linear")


class ShensiModel(DeepseekV4Model):
    def __init__(self, config: ShensiConfig):
        super().__init__(config)
        del self.hc_head

        self.num_attn_res_blocks = config.attn_res_block_layer_types.count("block_write_layer")
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
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
            # `generate()` may pass a per-layer-type mask dict already built by
            # `create_masks_for_generate`; all V4 layer types use the same sliding-window
            # mask, so use the prebuilt one directly. Otherwise build it here.
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = inputs_embeds
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        # Block history: one slot per attention-residual block, filled by that block's write layer.
        block_residual = hidden_states.new_empty(
            hidden_states.size(0),
            hidden_states.size(1),
            self.num_attn_res_blocks,
            hidden_states.size(2),
        )
        prefix_sum = hidden_states
        hidden_states = None
        residual = block_residual
        for layer in self.layers:
            hidden_states, prefix_sum, residual = layer(
                hidden_states,
                residual,
                prefix_sum,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
        hidden_states, _ = self.output_attn_res(
            prefix_sum,
            hidden_states,
            residual,
            output_norm_weight=None,
            num_blocks=self.num_attn_res_blocks,
        )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        # One table may be shared by several hash layers: resize each distinct table once, then rebind.
        resized = {}
        for layer in self.layers:
            if layer.is_hash:
                table = layer.mlp.deepemb
                if id(table) not in resized:
                    resized[id(table)] = self._get_resized_embeddings(
                        table, new_num_tokens, pad_to_multiple_of, mean_resizing
                    )
                layer.mlp.deepemb = resized[id(table)]
        return embeddings

    def tie_mlp(self) -> None:
        """Tie the MLP side: one router/expert set per attention-residual block, and one
        deep-embedding table across the hash-routed (leading) layers."""
        write_layers = [i for i, t in enumerate(self.config.attn_res_block_layer_types) if t == "block_write_layer"]
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
            elif layer.mlp.gate is None:
                layer.mlp.gate = ShensiTopKRouter(self.config)
                layer.mlp.experts = ShensiExperts(self.config)
        moe_position = {
            layer_idx: pos
            for pos, layer_idx in enumerate(idx for idx, layer in enumerate(self.layers) if not layer.is_hash)
        }
        group_patterns = {}
        group_tied_weights = {}
        for first_mlp, indices in shared.values():
            first_mlp.sharing_layers = [moe_position[idx] for idx in indices]
            for idx in indices[1:]:
                for attr in ("gate", "experts"):
                    source_key = f"layers.{indices[0]}.mlp.{attr}"
                    target_key = f"layers.{idx}.mlp.{attr}"
                    group_patterns[target_key] = source_key
                    for param_name, _ in getattr(first_mlp, attr).named_parameters():
                        group_tied_weights[f"{target_key}.{param_name}"] = f"{source_key}.{param_name}"
        # Deep embedding: one shared table across the hash layers (the per-token bottleneck scale).
        # Only the first hash layer creates one; the rest are bound to the same module here, and the
        # copies are registered as tied so the checkpoint keeps a single `deepemb.weight`.
        hash_layers = [(i, layer) for i, layer in enumerate(self.layers) if getattr(layer, "is_hash", False)]
        if len(hash_layers) > 1:
            first_idx, first_layer = hash_layers[0]
            for idx, layer in hash_layers[1:]:
                layer.mlp.deepemb = first_layer.mlp.deepemb
                source_key = f"layers.{first_idx}.mlp.deepemb"
                target_key = f"layers.{idx}.mlp.deepemb"
                group_patterns[target_key] = source_key
                group_tied_weights[f"{target_key}.weight"] = f"{source_key}.weight"

        self._tied_weights_keys = {**dict(self._tied_weights_keys or {}), **group_patterns}
        self._moe_group_tied_weights = group_tied_weights

    def post_init(self):
        # Tied MLP weights are registered after the base init so `tie_weights` can map them.
        PreTrainedModel.post_init(self)
        self.all_tied_weights_keys.update(getattr(self, "_moe_group_tied_weights", None) or {})
        self.tie_weights(recompute_mapping=False)


def erc_loss_func(
    router_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    r"""
    Auxiliary expert-router coupling (ERC) loss, as in arXiv 2512.23447.

    Each router row `R[i]` acts as a proxy token for the tokens routed to expert `i`, perturbed within
    `[1 - eps, 1 + eps]` where `eps` comes from the pairwise distances between router rows. Projecting
    the perturbed proxies through the experts yields the coupling matrix `M`, with `M[i, j]` the
    activation norm of expert `j` under the proxy of expert `i`. Every off-diagonal entry above
    `alpha * M[i, i]` is penalized:

    Args:
        router_weight:
            The router weight matrix `R` of shape [n_experts, hidden_dim] whose rows serve as proxy tokens.
        down_proj_weight:
            The routed expert down projection weight used to project the perturbed proxy tokens.
        gate_up_proj:
            The merged expert gate/up weight of shape [n_experts, 2 * intermediate_dim, routed_expert_hidden_size].
        alpha (`float`, *optional*):
            The scalar hyperparameter controlling the specialization level: an off-diagonal
            activation `M[i, j]` is penalized once it exceeds `alpha * M[i, i]`.

    Returns:
        The expert-router coupling loss.
    """
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
    M = torch.norm(torch.einsum("jDd,id->ijD", gate_up_proj, proxy), dim=-1)

    # Penalize off-diagonal entries above `alpha * M[i, i]`, from the rows and then the columns.
    row_diff = M - alpha * torch.diag(M).unsqueeze(1)
    row_diff_clamped = torch.clamp(row_diff, min=0.0)

    col_diff = M - alpha * torch.diag(M).unsqueeze(0)
    col_diff_clamped = torch.clamp(col_diff, min=0.0)

    mask = torch.ones_like(M) - torch.eye(M.size(0), device=M.device)
    total_diff = (row_diff_clamped + col_diff_clamped) * mask

    return total_diff.mean()


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
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
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
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        # ERC loss: one term per MoE group, averaged over the groups.
        erc_loss = None
        if labels is not None:
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
        # One table may be shared by several hash layers: resize each distinct table once, then rebind.
        resized = {}
        for layer in self.model.layers:
            if layer.is_hash:
                table = layer.mlp.deepemb
                if id(table) not in resized:
                    resized[id(table)] = self._get_resized_embeddings(
                        table, new_num_tokens, pad_to_multiple_of, mean_resizing
                    )
                layer.mlp.deepemb = resized[id(table)]
        return embeddings


__all__ = [
    "ShensiConfig",
    "ShensiPreTrainedModel",
    "ShensiModel",
    "ShensiForCausalLM",
]
