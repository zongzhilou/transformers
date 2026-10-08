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
    DeepseekV4HyperHead,
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
    scoring_func (`str`):
        Router activation — `sqrtsoftplus`, `softmax`, or `sigmoid`.
    rope_theta (`float`):
        RoPE base for the main self-attention rotary.
    layer_types (`list[str]`):
        Per-layer attention schedule with values from
        `{"compressed_sparse_attention", "heavily_compressed_attention"}`.
        Default: 2× HCA bootstrap + interleaved CSA / HCA.
    compress_rates (`dict[str, int]`):
        Per-layer-type compression rate. Default
        `{"compressed_sparse_attention": 4, "heavily_compressed_attention": 128}`.
        BC: configs that ship `compress_rate_csa` / `compress_rate_hca` as
        top-level kwargs are folded in at `__post_init__` time.
    compress_rope_theta (`float`):
        RoPE base for the compressed branches (paired with
        `rope_scaling` for YaRN).
    hc_mult (`int`):
        Manifold-Constrained Hyper-Connection (mHC) expansion factor
        (always active).
    mlp_layer_types (`list[str]`):
        Per-layer MoE schedule with values from
        `{"hash_moe", "moe"}`. `moe` is the standard top-k routed MoE.
        Default: 3× `hash_moe` then `moe` for the rest. BC: legacy configs
        that ship `num_hash_layers` as a top-level kwarg are folded in at
        `__post_init__` time.
    swiglu_limit (`float`):
        Clip routed experts' gate/up pre-activations.
    sliding_window (`int`):
        Local window size used in every attention block's sliding-window
        branch.
    o_groups (`int`):
        Number of head-groups in the grouped output projection.
    o_lora_rank (`int`):
        Per-group intermediate dim in the grouped output projection.
    index_n_heads (`int`):
        Number of indexer query heads.
    index_head_dim (`int`):
        Indexer head dim.
    index_topk (`int`):
        Number of compressed entries per query the indexer keeps via top-k.
    num_nextn_predict_layers (`int`):
        MTP layer count in the upstream checkpoint
        (not instantiated here).
    partial_rotary_factor (`float`, *optional*):
        Fraction of head_dim that gets RoPE.
        Defaults to `qk_rope_head_dim / head_dim` so cos/sin sizes to `qk_rope_head_dim`.
    routed_expert_hidden_size (`int`, *optional*):
        Intermediate size of the routed experts in MoE layers.
    erc_loss_alpha (`float`, *optional*):
        Anchor coefficient α of the expert–router coupling (ERC) loss (arXiv 2512.23447).
    erc_loss_coef (`float`, *optional*):
        Coefficient scaling the expert–router coupling (ERC) loss.
    hc_active_streams (`int`, *optional*):
        Active streams k refreshed per token; the other N−k streams stay unchanged.
    hc_fixed_streams (`int`, *optional*):
        Fixed streams m always refreshed, on top of which routing selects k−m top-scoring streams.
    hc_conv_kernels (`tuple[int, ...]`, *optional*):
        Kernel sizes of the causal depthwise 1D convolutions in the temporal augmentation.
    attn_res_block_size (`int`, *optional*):
        AttnRes block size: layers are grouped into blocks of B, and each block's first layer writes its delta.
    """

    hidden_size: int = 1280
    moe_intermediate_size: int = 800
    num_hidden_layers: int = 19
    num_attention_heads: int = 16
    q_lora_rank: int = 320
    n_shared_experts = AttributeError()
    routed_expert_hidden_size: int = 320

    hc_mult: int = 16
    hc_sinkhorn_iters = AttributeError()
    hc_eps = AttributeError()
    o_groups: int = 2
    o_lora_rank: int = 320

    # ERC
    erc_loss_coef: float = 1.0
    erc_loss_alpha: float = 0.5

    # xHC
    hc_active_streams: int = 4
    hc_fixed_streams: int = 2
    hc_conv_kernels: tuple[int, ...] | list[int] = (4, 8, 12)

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
    def __init__(self, config: ShensiConfig):
        super().__init__()
        self.intermediate_size = config.routed_expert_hidden_size
        self.deepemb = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.limit)
        up = self.up_proj(hidden_states).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(self.act_fn(gate) * up) * self.deepemb(input_ids)


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


class ShensiHyperConnection(nn.Module):
    def __init__(self, config: ShensiConfig, is_mlp: bool = False):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.active_streams = config.hc_active_streams
        self.fixed_streams = config.hc_fixed_streams
        self.routed_streams = self.active_streams - self.fixed_streams

        self.input_norm = ShensiUnweightedRMSNorm(eps=config.rms_norm_eps)
        self.pre_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.pre_base = nn.Parameter(torch.empty(self.hc_mult))
        self.pre_scale = nn.Parameter(torch.empty(1))

        self.route_norm = nn.LayerNorm(self.hc_mult * config.hidden_size)
        self.route_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.route_base = nn.Parameter(torch.empty(self.hc_mult))
        self.route_scale = nn.Parameter(torch.empty(1))

        self.is_mlp = is_mlp
        self.kr = (len(config.hc_conv_kernels) + 1) if self.is_mlp else 1
        if self.is_mlp:
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
                    for ks in config.hc_conv_kernels
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
        self.q_a_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=False)
        self.q_b_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, bias=False)
        self.g_a_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=True)
        self.g_b_proj = nn.Linear(config.routed_expert_hidden_size, 3 * config.hidden_size, bias=True)
        self.g_scale = nn.Parameter(torch.zeros(4))
        self.t = nn.Parameter(torch.linspace(0.0, 1.0, config.hidden_size) * math.log(2.0 * config.num_hidden_layers))
        self.k_a_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=False)
        self.k_b_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, bias=False)

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

        # ---- write path ----
        decay_scale, erase_scale, write_scale, read_scale = self.g_scale.unbind()
        # keep decay <= 1 without freezing the gate: clamp the forward value, pass the
        # gradient through (a plain clamp() has zero grad at the boundary)
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

        # ---- read path: diag + above-MP-edge spike metric, rank decided by the data ----
        if num_blocks > 0:
            blk = blocks[..., :num_blocks, :]
            H = updated.shape[-1]
            M = updated.numel() // H
            query = F.linear(F.linear(state, self.q_a_proj.weight.float()), self.q_b_proj.weight.float())

            # deterministic-shape budget: K = 2H sampled value rows
            K = min(2 * H, M * num_blocks)
            ti = torch.randint(M, (K,), device=blocks.device)
            ji = torch.randint(num_blocks, (K,), device=blocks.device)
            rows = blk.detach()[(*torch.unravel_index(ti, blocks.shape[:-2]), ji)].float()

            # per-coordinate second moments, shrunk toward 1 by evidence: H/(K+H);
            # floor keeps dead coordinates finite, no epsilon tuning
            m2 = torch.einsum("kd,kd->d", rows, rows) / K
            s = torch.lerp(m2, m2.new_ones(()), H / (K + H)).clamp_min(1e-20).rsqrt()
            z = rows * s

            # one SVD + Marchenko-Pastur edge: keep only statistically estimable
            # directions; BBP debiasing recovers the population spike strength
            _, S, V = torch.svd_lowrank(z, q=min(64, K, H), niter=4)
            ell = S.square() / K
            keep = ell > (1.0 + math.sqrt(H / K)) ** 2
            b = ell[keep] - 1.0 - H / K
            theta = 0.5 * (b + (b.square() - 4.0 * H / K).clamp_min(0.0).sqrt())
            U = V[:, keep] * (theta / (1.0 + theta)).unsqueeze(0) * s.unsqueeze(-1)

            # metric folded entirely into the query: C^-1 ~ s^2 (I - U U^T),
            # values are only scored and retrieved, never touched by the metric
            q = s * (query - (query @ U) @ U.T)

            # scoring and retrieval: the only two passes over all values
            logits = torch.cat(
                (
                    (blk @ q.to(out_dtype).unsqueeze(-1)).squeeze(-1).float(),
                    torch.einsum("...d,...d->...", updated, q).unsqueeze(-1),
                ),
                dim=-1,
            )
            # exp(logit - softplus(lse)) == sigmoid(lse) * softmax(logit);
            # score sum stays below 1, the residual always dominates the readout
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
        self.mlp = ShensiHashMLP(config) if self.is_hash else ShensiSparseMoeBlock(config, layer_idx)
        self.input_layernorm = ShensiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ShensiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = ShensiHyperConnection(config, is_mlp=False)
        self.ffn_hc = ShensiHyperConnection(config, is_mlp=True)
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

        collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(collapsed, **kwargs)
        hidden_states = self.attn_hc.write_back(hidden_states, attn_output)

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

        collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(collapsed, input_ids=input_ids) if self.is_hash else self.mlp(collapsed)
        hidden_states = self.ffn_hc.write_back(hidden_states, mlp_output)

        prefix_sum = prefix_sum + hidden_states
        return hidden_states, prefix_sum, residual


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
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "hc_head",
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
        self.num_attn_res_blocks = config.attn_res_block_layer_types.count("block_write_layer")
        self.output_attn_res = ShensiAttentionResidual(config)
        self.tie_moe_groups()

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

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = super()._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        for layer in self.layers:
            if layer.is_hash:
                layer.mlp.deepemb = self._get_resized_embeddings(
                    layer.mlp.deepemb, new_num_tokens, pad_to_multiple_of, mean_resizing
                )
        return embeddings

    def tie_moe_groups(self) -> None:
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
        self._tied_weights_keys = {**dict(self._tied_weights_keys or {}), **group_patterns}
        self._moe_group_tied_weights = group_tied_weights

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
    r"""
    Computes auxiliary expert-router coupling (ERC) loss as in arXiv 2512.23447 - implemented in Pytorch.

    See arXiv 2512.23447 (https://arxiv.org/abs/2512.23447) for more details. This function implements the loss
    function presented in Figure 8 (pseudocode, `erc_loss` method) of the paper. Each router row `R[i]` is used as
    a proxy token for the input tokens routed to expert `i`, and the proxy tokens are perturbed within the interval
    `[1 - eps, 1 + eps]` where `eps` is derived from the pairwise distances between router rows. The perturbed proxy
    tokens are projected by `down_proj_weight` and fed through the expert weights `gate_up_proj`, which yields the
    coupling matrix `M` with `M[i, j]` the activation norm of expert `j` given the proxy token of expert `i`. For
    all `i != j`, a penalty is imposed wherever the off-diagonal elements `M[i, j]` or `M[j, i]` exceed
    `alpha * M[i, i]`, where `alpha` is a scalar hyperparameter:

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

    # Penalize the off-diagonal rows that exceed alpha times the diagonal
    row_diff = M - alpha * torch.diag(M).unsqueeze(1)
    row_diff_clamped = torch.clamp(row_diff, min=0.0)

    # Penalize the off-diagonal columns that exceed alpha times the diagonal
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

        >>> model = ShensiForCausalLM.from_pretrained("louzongzhi/Shensi")
        >>> tokenizer = AutoTokenizer.from_pretrained("louzongzhi/Shensi")

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
        for layer in self.model.layers:
            if layer.is_hash:
                layer.mlp.deepemb = self._get_resized_embeddings(
                    layer.mlp.deepemb, new_num_tokens, pad_to_multiple_of, mean_resizing
                )
        return embeddings


__all__ = [
    "ShensiConfig",
    "ShensiPreTrainedModel",
    "ShensiModel",
    "ShensiForCausalLM",
]
