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

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.output_capturing import OutputRecorder
from ..deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from ..deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4CSACache,
    DeepseekV4CSACompressor,
    DeepseekV4DecoderLayer,
    DeepseekV4Experts,
    DeepseekV4ForCausalLM,
    DeepseekV4GroupedLinear,
    DeepseekV4HCACache,
    DeepseekV4HCACompressor,
    DeepseekV4HyperHead,
    DeepseekV4Indexer,
    DeepseekV4IndexerScorer,
    DeepseekV4Model,
    DeepseekV4PreTrainedModel,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseMoeBlock,
    DeepseekV4TopKRouter,
    DeepseekV4UnweightedRMSNorm,
    load_balancing_loss_func,
)


@auto_docstring(checkpoint="louzongzhi/Shensi-30B")
@strict
class ShensiConfig(DeepseekV4Config):
    r"""
    scoring_func (`str`):
        Router activation — `sqrtsoftplus`, `softmax`, or `sigmoid`.
    rope_theta (`float`):
        RoPE base for the main self-attention rotary.
    layer_types (`list[str]`):
        Per-layer attention schedule with values from
        `{"compressed_sparse_attention", "heavily_compressed_attention"}`.
        V4-Pro default: 2× HCA bootstrap + interleaved CSA / HCA.
    compress_rates (`dict[str, int]`):
        Per-layer-type compression rate. Default
        `{"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}`
        (m=4 for CSA, m'=128 for HCA, paper §2.3.1 / §2.3.2). BC: configs that ship
        `compress_rate_csa` / `compress_rate_hca` as top-level kwargs are folded
        in at `__post_init__` time.
    compress_rope_theta (`float`):
        RoPE base for the compressed branches (paired with
        `rope_scaling` for YaRN).
    hc_mult (`int`):
        Manifold-Constrained Hyper-Connection (mHC) expansion factor n_hc
        (always active; Section 2.2).
    mlp_layer_types (`list[str]`):
        Per-layer MoE schedule with values from
        `{"hash_moe", "moe"}`. `hash_moe` routes via a frozen
        `tid2eid[input_ids]` lookup (paper §2.1, "Hash-MoE bootstrap"); `moe`
        is the standard top-k routed MoE. Default: 3× `hash_moe` then `moe`
        for the rest. BC: legacy configs that ship `num_hash_layers` as a
        top-level kwarg are folded in at `__post_init__` time.
    swiglu_limit (`float`):
        Clip routed experts' gate/up pre-activations.
    sliding_window (`int`):
        Local window size n_win used in every attention block's
        sliding-window branch.
    o_groups (`int`):
        Number of head-groups g in the grouped output projection
        (paper §2.3.1, "Grouped Output Projection").
    o_lora_rank (`int`):
        Per-group intermediate dim d_g in the grouped output projection.
    index_n_heads (`int`):
        Number of indexer query heads n_h^I (paper §2.3.1, eq. 14).
    index_head_dim (`int`):
        Indexer head dim c^I (paper §2.3.1).
    index_topk (`int`):
        Number of compressed entries per query the Lightning Indexer
        keeps via top-k (paper §2.3.1, eq. 17).
    num_nextn_predict_layers (`int`):
        MTP layer count in the upstream checkpoint
        (not instantiated here).
    partial_rotary_factor (`float`, *optional*):
        Fraction of head_dim that gets RoPE.
        Defaults to `qk_rope_head_dim / head_dim` so cos/sin sizes to `qk_rope_head_dim`.
    routed_expert_hidden_size (`int`, *optional*):
        Intermediate size of the routed experts in MoE layers.
    hc_active_streams (`int`, *optional*):
        Active streams k refreshed per token; the other N−k streams stay unchanged (paper §3.3.2).
    hc_fixed_streams (`int`, *optional*):
        Fixed streams m always refreshed, on top of which routing selects k−m top-scoring streams (paper §3.3.2).
    hc_conv_kernels (`tuple[int, ...]`, *optional*):
        Kernel sizes of the causal depthwise 1D convolutions in the temporal augmentation (paper §3.3.1).
    attn_res_block_size (`int`, *optional*):
        AttnRes block size: layers are grouped into blocks of B, and each block's first layer writes its delta.
    output_sharing_layers (`bool`, *optional*):
        Whether to output the sharing layers (pool members) per pool.
    output_coupling_matrix (`bool`, *optional*):
        Whether to output the per-pool expert–router coupling matrix M.
    erc_loss_alpha (`float`, *optional*):
        Anchor coefficient α of the expert–router coupling (ERC) loss (arXiv 2512.23447).
    erc_loss_coef (`float`, *optional*):
        Coefficient scaling the expert–router coupling (ERC) loss.
    """

    hidden_size: int = 2560
    moe_intermediate_size: int = 1280
    num_hidden_layers: int = 51
    num_attention_heads: int = 32
    q_lora_rank: int = 640
    n_routed_experts: int = 1152

    hc_mult: int = 16
    sliding_window: int = 640
    o_groups: int = 4
    index_head_dim: int = 320

    routed_expert_hidden_size: int | None = 640

    hc_active_streams: int | None = 4
    hc_fixed_streams: int | None = 2
    hc_conv_kernels: tuple[int, ...] | list[int] | None = (4, 8, 12)

    attn_res_block_size: int | None = 6

    output_sharing_layers: bool = False
    output_coupling_matrix: bool = False
    erc_loss_alpha: float = 0.5
    erc_loss_coef: float = 1.0

    n_shared_experts = AttributeError()
    hc_sinkhorn_iters = AttributeError()
    hc_eps = AttributeError()

    @property
    def attn_res_block_layer_types(self) -> list[str]:
        return [
            "block_write_layer"
            if i == 0 or (i - self.mlp_layer_types.count("hash_moe")) % self.attn_res_block_size == 0
            else "block_read_layer"
            for i in range(self.num_hidden_layers)
        ]


class ShensiRMSNorm(DeepseekV4RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


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


class ShensiMLP(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.limit = config.swiglu_limit
        self.is_hash = True
        self.gate_up_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size * 2, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.deepemb = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        down = self.down_proj(self.act_fn(gate) * up)
        return down * self.deepemb(input_ids.to(self.deepemb.weight.device)).to(down.dtype)


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


class ShensiSparseMoeBlock(DeepseekV4SparseMoeBlock):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        del self.shared_experts
        self.gate = (
            ShensiTopKRouter(config) if config.attn_res_block_layer_types[layer_idx] == "block_write_layer" else None
        )
        self.experts = (
            ShensiExperts(config) if config.attn_res_block_layer_types[layer_idx] == "block_write_layer" else None
        )
        self.routed_expert_down_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=False)
        self.routed_expert_norm = ShensiRMSNorm(config.routed_expert_hidden_size, config.rms_norm_eps)
        self.routed_expert_up_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, list[int] | None]:
        batch, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _, weights, indices = self.gate(hidden_states)
        routed = self.experts(self.routed_expert_down_proj(flat), indices, weights)
        output = self.routed_expert_up_proj(self.routed_expert_norm(routed)).view(batch, seq_len, hidden_dim)

        coupling_matrix = None
        sharing_layers = getattr(self, "sharing_layers", None)
        if self.training and sharing_layers is not None:
            R = self.gate.weight
            norm_R = torch.norm(R, dim=1)
            distances = torch.cdist(R, R, p=2)
            distances = distances.masked_fill(
                torch.eye(R.size(0), dtype=torch.bool, device=distances.device), float("inf")
            )
            min_dist, _ = torch.min(distances, dim=1)
            eps = min_dist / 2 / norm_R

            low = (1 - eps).unsqueeze(1)
            high = (1 + eps).unsqueeze(1)
            noise = torch.rand_like(R)
            R_tilde = (low + noise * (high - low)) * R

            proxy = F.linear(R_tilde, self.routed_expert_down_proj.weight)
            coupling_matrix = torch.norm(torch.einsum("jDd,id->ijD", self.experts.gate_up_proj, proxy), dim=-1)

        return output, coupling_matrix, sharing_layers


class ShensiHyperConnection(nn.Module):
    def __init__(self, config: ShensiConfig, is_mlp: bool = False):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.active_streams = config.hc_active_streams
        self.fixed_streams = config.hc_fixed_streams
        self.routed_streams = self.active_streams - self.fixed_streams

        self.input_norm = ShensiUnweightedRMSNorm(eps=config.rms_norm_eps)
        self.route_norm = nn.LayerNorm(self.hc_mult * config.hidden_size)

        self.pre_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.pre_base = nn.Parameter(torch.empty(self.hc_mult))
        self.pre_scale = nn.Parameter(torch.empty(1))

        self.route_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.route_base = nn.Parameter(torch.empty(self.hc_mult))
        self.route_scale = nn.Parameter(torch.empty(1))

        self.is_mlp = is_mlp
        self.kr = (len(config.hc_conv_kernels) + 1) if self.is_mlp else 1
        if self.is_mlp:
            self.conv_kernels = config.hc_conv_kernels
            self.temporal_convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        config.hidden_size,
                        config.hidden_size,
                        ks,
                        padding=ks - 1,
                        groups=config.hidden_size,
                        bias=False,
                    )
                    for ks in self.conv_kernels
                ]
            )

        self.post_fn = nn.Parameter(
            torch.empty(self.active_streams * self.kr, self.active_streams * config.hidden_size)
        )
        self.post_base = nn.Parameter(torch.empty(self.active_streams * self.kr))
        self.post_scale = nn.Parameter(torch.empty(1))

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre = torch.sigmoid(F.linear(flat, self.pre_fn.float()) * self.pre_scale.float() + self.pre_base.float())
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return collapsed

    def write_back(self, hidden_streams: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        B, S, hc, H = hidden_streams.shape

        # `route_norm` is a weighted LayerNorm: run it in its own weight dtype (fp16 when
        # the model is fp16, fp32 under the strict keep-in-fp32 upcast), then cast to fp32
        # for the gate math — a plain `.float()` input would mix dtypes on CPU.
        flat = self.route_norm(hidden_streams.flatten(start_dim=2).to(self.route_norm.weight.dtype)).float()
        route_scores = torch.sigmoid(
            F.linear(flat, self.route_fn.float()) * self.route_scale.float() + self.route_base.float()
        )
        fixed_mask = torch.arange(hc, device=route_scores.device) < self.fixed_streams
        route_scores = route_scores.masked_fill(fixed_mask.view(1, 1, -1), -float("inf"))
        fixed_idx = torch.arange(self.fixed_streams, device=hidden_streams.device)
        fixed_idx = fixed_idx.view(1, 1, -1).expand(B, S, -1)
        routed_idx = route_scores.topk(self.routed_streams, dim=-1).indices
        active_idx = torch.cat([fixed_idx, routed_idx], dim=-1)
        p = torch.cat(
            [torch.ones_like(fixed_idx, dtype=route_scores.dtype), route_scores.gather(-1, routed_idx)], dim=-1
        )

        if self.is_mlp:
            x = sublayer_output.transpose(1, 2).to(self.temporal_convs[0].weight.dtype)
            conv_outs = [conv(x)[..., :S] for conv in self.temporal_convs]
            ortho = []
            prevs = [x]
            for g in conv_outs:
                v = g
                for prev in prevs:
                    denom = (prev * prev).sum(dim=1, keepdim=True).clamp_min(self.input_norm.eps)
                    v = v - ((prev * v).sum(dim=1, keepdim=True) / denom) * prev
                ortho.append(v)
                prevs.append(v)
            out_aug = torch.cat([x] + ortho, dim=1).transpose(1, 2).reshape(B, S, self.kr, H).float()
        else:
            out_aug = sublayer_output.float().unsqueeze(-2)

        active_streams = hidden_streams.gather(2, active_idx.unsqueeze(-1).expand(-1, -1, -1, H))
        post = 2 * torch.sigmoid(
            F.linear(self.input_norm(active_streams.flatten(start_dim=2).float()), self.post_fn.float()).view(
                B, S, self.active_streams, self.kr
            )
            * self.post_scale.float()
            + self.post_base.float().view(self.active_streams, self.kr)
        )

        delta = torch.einsum("bskr,bsrh->bskh", post, out_aug) * p.unsqueeze(-1)
        updated_active = delta.to(hidden_streams.dtype)
        return hidden_streams.scatter(2, active_idx.unsqueeze(-1).expand(-1, -1, -1, H), updated_active)


class ShensiAttentionResidual(nn.Module):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.norm = ShensiUnweightedRMSNorm(config.rms_norm_eps)
        self.gate_proj = nn.Parameter(torch.empty(3 * config.hidden_size, config.hidden_size))
        self.q_proj = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.k_proj = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
        prev_snapshot: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
        block_write_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = prefix_sum
        delta = hidden_states
        blocks = residual
        state = self.norm(prefix.float() + (delta.float() if delta is not None else 0.0))
        if delta is None:
            forgotten = prefix.float()
            updated = prefix.float()
        else:
            decay, erase, write = torch.sigmoid(
                F.linear(state, self.gate_proj.float()).reshape(*state.shape[:-1], 3, -1)
            ).unbind(-2)
            forgotten = decay * prefix.float()
            khat = F.normalize(F.linear(state, self.k_proj.float()), dim=-1)
            r = (khat * erase * forgotten).sum(dim=-1, keepdim=True)
            updated = forgotten - khat * r + write * delta.float()
        if num_blocks == 0:
            routed = torch.zeros_like(forgotten)
        else:
            values = torch.cat(
                [
                    blocks[..., :num_blocks, :].float(),
                    (updated - prev_snapshot).unsqueeze(-2),
                ],
                dim=-2,
            )
            reciprocal_std = torch.rsqrt(values.square().mean(dim=-1) + self.norm.eps)
            query = F.linear(state, self.q_proj.float())
            logits = (values * query.unsqueeze(-2)).sum(dim=-1) * reciprocal_std
            scores = F.softmax(logits, dim=-1)
            routed = scores.unsqueeze(-1).mul(values).sum(dim=-2)
        output = updated + routed
        if output_norm_weight is not None:
            output = (
                output
                * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + self.norm.eps)
                * output_norm_weight.float()
            )

        if block_write_idx >= 0:
            blocks = torch.cat(
                [
                    blocks[..., :block_write_idx, :],
                    (updated - prev_snapshot).to(blocks.dtype).unsqueeze(-2),
                    blocks[..., block_write_idx + 1 :, :],
                ],
                dim=-2,
            )
            # Advance the snapshot via the return value instead of an in-place `copy_`
            # (in-place mutation of an input breaks gradient checkpointing).
            prev_snapshot = updated.to(prev_snapshot.dtype)
        prefix = updated.to(prefix.dtype)
        return output.to(prefix.dtype), prefix, blocks, prev_snapshot


class ShensiDecoderLayer(DeepseekV4DecoderLayer):
    def __init__(self, config: ShensiConfig, layer_idx: int):
        super().__init__()
        self.mlp = (
            ShensiMLP(config)
            if config.mlp_layer_types[layer_idx] == "hash_moe"
            else ShensiSparseMoeBlock(config, layer_idx)
        )
        self.attn_hc = ShensiHyperConnection(config, is_mlp=False)
        self.ffn_hc = ShensiHyperConnection(config, is_mlp=True)
        self.is_block_write_layer = config.attn_res_block_layer_types[layer_idx] == "block_write_layer"
        self.prev_valid_blocks = sum(
            1 for r in config.attn_res_block_layer_types[:layer_idx] if r == "block_write_layer"
        )
        self.block_write_idx = self.prev_valid_blocks
        self.self_attention_attn_res = ShensiAttentionResidual(config)
        self.mlp_attn_res = ShensiAttentionResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None = None,
        prev_snapshot: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        hidden_states, prefix_sum, residual, prev_snapshot = self.self_attention_attn_res(
            hidden_states,
            residual,
            prefix_sum,
            prev_snapshot,
            output_norm_weight=self.input_layernorm.weight,
            num_blocks=self.prev_valid_blocks,
            block_write_idx=self.block_write_idx if self.is_block_write_layer else -1,
        )

        collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(collapsed, **kwargs)
        hidden_states = self.attn_hc.write_back(hidden_states, attn_output)

        if self.is_block_write_layer:
            prefix_sum = hidden_states
            prefix_delta = None
        else:
            prefix_delta = hidden_states
        mlp_valid_blocks = self.prev_valid_blocks + self.is_block_write_layer

        hidden_states, prefix_sum, residual, prev_snapshot = self.mlp_attn_res(
            prefix_delta,
            residual,
            prefix_sum,
            prev_snapshot,
            output_norm_weight=self.post_attention_layernorm.weight,
            num_blocks=mlp_valid_blocks,
            block_write_idx=-1,
        )

        collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(collapsed, input_ids=input_ids) if self.mlp.is_hash else self.mlp(collapsed)[0]
        hidden_states = self.ffn_hc.write_back(hidden_states, mlp_output)

        return hidden_states, prefix_sum, residual, prev_snapshot


class ShensiHyperHead(DeepseekV4HyperHead):
    def __init__(self, config: ShensiConfig):
        super().__init__()
        del self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float())
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class ShensiPreTrainedModel(DeepseekV4PreTrainedModel):
    _can_record_outputs = {
        "router_logits": OutputRecorder(ShensiTopKRouter, index=0),
        "coupling_matrix": OutputRecorder(ShensiSparseMoeBlock, index=1),
        "sharing_layers": OutputRecorder(ShensiSparseMoeBlock, index=2),
        "attentions": ShensiAttention,
    }
    _keep_in_fp32_modules_strict = {
        "attn_hc",
        "ffn_hc",
        "hc_head",
        "sinks",
        "position_bias",
        "q_a_norm",
        "kv_norm",
        "input_layernorm",
        "post_attention_layernorm",
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, ShensiTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, ShensiExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, ShensiAttention):
            init.zeros_(module.sinks)
        elif isinstance(module, ShensiHyperConnection):
            init.normal_(module.pre_fn, mean=0.0, std=std)
            init.zeros_(module.pre_base)
            init.constant_(module.pre_scale, 0.01)
            init.normal_(module.route_fn, mean=0.0, std=std)
            init.zeros_(module.route_base)
            init.ones_(module.route_scale)
            init.normal_(module.post_fn, mean=0.0, std=std)
            init.zeros_(module.post_base)
            init.constant_(module.post_scale, 0.01)
        elif isinstance(module, ShensiHyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
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
            init.zeros_(module.gate_proj)
            init.zeros_(module.q_proj)
            init.normal_(module.k_proj, mean=0.0, std=std)

    def post_init(self):
        PreTrainedModel.post_init(self)
        layer_types = self.config.attn_res_block_layer_types
        if layer_types and hasattr(self, "layers"):
            write_layers = [i for i, t in enumerate(layer_types) if t == "block_write_layer"]
            shared = {}
            for layer_idx, layer in enumerate(self.layers):
                if layer.mlp.is_hash:
                    continue
                block_id = max(w for w in write_layers if w <= layer_idx)
                pool = shared.setdefault(block_id, [layer.mlp, [layer_idx]])
                if pool[0] is not layer.mlp:
                    pool[1].append(layer_idx)
                    layer.mlp.gate = pool[0].gate
                    layer.mlp.experts = pool[0].experts
            moe_position = {
                layer_idx: pos
                for pos, layer_idx in enumerate(idx for idx, layer in enumerate(self.layers) if not layer.mlp.is_hash)
            }
            for first_mlp, indices in shared.values():
                first_mlp.sharing_layers = [moe_position[idx] for idx in indices]
            self._tied_weights_keys = {
                f"layers.{idx}.mlp.{attr}.{param_name}": f"layers.{indices[0]}.mlp.{attr}.{param_name}"
                for first_mlp, indices in shared.values()
                for idx in indices[1:]
                for attr in ("gate", "experts")
                for param_name, _ in getattr(first_mlp, attr).named_parameters()
            }
            self.all_tied_weights_keys = self._tied_weights_keys
        self.tie_weights(recompute_mapping=False)


class ShensiModel(DeepseekV4Model):
    def __init__(self, config: ShensiConfig):
        super().__init__(config)
        self.num_attn_res_blocks = config.attn_res_block_layer_types.count("block_write_layer")
        self.output_attn_res = ShensiAttentionResidual(config)

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
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        block_residual = hidden_states.new_empty(
            hidden_states.size(0),
            hidden_states.size(1),
            hidden_states.size(2),
            self.num_attn_res_blocks,
            hidden_states.size(3),
        )
        prefix_sum = hidden_states
        hidden_states = None
        residual = block_residual
        prev_snapshot = prefix_sum.new_zeros(prefix_sum.shape)

        output_hidden_states = kwargs.get("output_hidden_states", self.config.output_hidden_states)
        # The first layer consumes `prefix_sum` and receives `hidden_states=None`, so the
        # initial hidden state is the raw (4D, hc_mult-stream) input itself — collect it
        # explicitly instead of relying on the capture mechanism, which would prepend the
        # `None` first-layer input.
        all_hidden_states = (prefix_sum,) if output_hidden_states else ()

        for layer in self.layers:
            hidden_states, prefix_sum, residual, prev_snapshot = layer(
                hidden_states,
                residual,
                prefix_sum,
                prev_snapshot,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        hidden_states, _, _, _ = self.output_attn_res(
            hidden_states,
            residual,
            prefix_sum,
            prev_snapshot,
            output_norm_weight=None,
            num_blocks=self.num_attn_res_blocks,
            block_write_idx=-1,
        )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states or None,
        )


def erc_loss_func(
    coupling_matrix: torch.Tensor | None,
    alpha: float = 1.0,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary expert-router coupling (ERC) loss as in arXiv 2512.23447 - implemented in Pytorch.

    See arXiv 2512.23447 (https://arxiv.org/abs/2512.23447) for more details. This function implements the loss
    function presented in Figure 8 (pseudocode, `erc_loss` method) of the paper, given the coupling matrix `M`
    with `M[i, j]` the activation norm of expert `j` given the proxy token of expert `i`. For all `i != j`, a
    penalty is imposed wherever the off-diagonal elements `M[i, j]` or `M[j, i]` exceed `alpha * M[i, i]`, where
    `alpha` is a scalar hyperparameter:

    Args:
        coupling_matrix:
            The coupling matrix `M` of shape [n_experts, n_experts], with `M[i, j]` the activation
            norm of expert `j` given the proxy token of expert `i`.
        alpha (`float`, *optional*):
            The scalar hyperparameter controlling the specialization level: an off-diagonal
            activation `M[i, j]` is penalized once it exceeds `alpha * M[i, i]`.

    Returns:
        The expert-router coupling loss.
    """
    if coupling_matrix is None:
        return 0

    M = coupling_matrix

    # Penalize the off-diagonal rows that exceed alpha times the diagonal
    row_diff = M - alpha * torch.diag(M).unsqueeze(1)
    row_diff_clamped = torch.clamp(row_diff, min=0.0)

    # Penalize the off-diagonal columns that exceed alpha times the diagonal
    col_diff = M - alpha * torch.diag(M).unsqueeze(0)
    col_diff_clamped = torch.clamp(col_diff, min=0.0)

    mask = torch.ones_like(M) - torch.eye(M.size(0), device=M.device)
    total_diff = (row_diff_clamped + col_diff_clamped) * mask

    return total_diff.mean()


def compute_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
    sharing_layers: tuple[list[int] | None, ...] | None = None,
    coupling_matrices: tuple[torch.Tensor | None, ...] | None = None,
    alpha: float = 1.0,
) -> tuple[torch.Tensor | int, torch.Tensor | int]:
    r"""
    Computes the pool-level auxiliary losses for the shared MoE pools.

    The first term is the pool-level load-balancing loss (UniPool): the load-balancing
    statistics are averaged *within* each shared expert pool (an attention-residual block)
    instead of across all layers, as `load_balancing_loss_func` does. `sharing_layers` are the layers sharing each pool (UniPool
    paper, §4.2 "the L sharing layers"), mapped to their positions in the captured
    `router_logits` tuple
    each pool to the positions of its member layers in the captured `router_logits` tuple
    (see `ShensiPreTrainedModel.post_init`).

    The second term is the pool-level expert-router coupling loss (arXiv 2512.23447): each
    captured coupling matrix (computed by the pool-owning `ShensiSparseMoeBlock` and
    collected by the OutputRecorder on `coupling_matrix`) is run through the paper's
    `erc_loss_func`.

    Args:
        gate_logits:
            Logits from the `gate`, a tuple of per-moe-layer tensors of shape
            [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts.
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k`
            routing parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function, shape [batch_size X sequence_length]
            if not None.
        sharing_layers:
            The captured positions of each pool's sharing layers in `gate_logits` (one list
            per pool-owning layer, `None` for the read layers).
        coupling_matrices:
            The captured coupling matrices `M` of the shared pools (one per pool-owning
            layer, `None` for the read layers).
        alpha (`float`, *optional*):
            The coupling margin passed to `erc_loss_func`.

    Returns:
        A tuple `(aux_loss, erc_loss)` with the pool-level load-balancing loss and the
        expert-router coupling loss.
    """
    aux_loss = 0
    if gate_logits is not None and isinstance(gate_logits, tuple) and sharing_layers:
        for layer_ids in sharing_layers:
            if layer_ids is None:
                continue
            subset = tuple(gate_logits[pos] for pos in layer_ids)
            aux_loss += load_balancing_loss_func(subset, num_experts, top_k, attention_mask)

    erc_loss = 0
    if coupling_matrices:
        for coupling_matrix in coupling_matrices:
            if coupling_matrix is None:
                continue
            erc_loss += erc_loss_func(coupling_matrix, alpha=alpha)

    return aux_loss, erc_loss


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
        >>> from transformers import AutoTokenizer, DeepseekV4ForCausalLM

        >>> model = DeepseekV4ForCausalLM.from_pretrained("mistralai/DeepseekV4-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/DeepseekV4-8x7B-v0.1")

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
            output_coupling_matrix=output_router_logits,
            output_sharing_layers=output_router_logits,
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
            aux_loss, erc_loss = compute_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
                getattr(outputs, "sharing_layers", None),
                getattr(outputs, "coupling_matrix", None),
                self.config.erc_loss_alpha,
            )
            if labels is not None and aux_loss != 0:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device
            if labels is not None and erc_loss != 0:
                loss += self.config.erc_loss_coef * erc_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = [
    "ShensiConfig",
    "ShensiPreTrainedModel",
    "ShensiModel",
    "ShensiForCausalLM",
]
