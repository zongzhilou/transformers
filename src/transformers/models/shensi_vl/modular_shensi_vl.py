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
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...masking_utils import create_recurrent_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPooling,
)
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    torch_compilable_check,
)
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..auto import AutoConfig
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4RMSNorm, DeepseekV4UnweightedRMSNorm
from ..kimi_k25.modeling_kimi_k25 import (
    Kimi_K25CausalLMOutputWithPast,
    Kimi_K25ForConditionalGeneration,
    Kimi_K25Model,
    Kimi_K25ModelOutputWithPast,
    Kimi_K25MultimodalProjection,
    Kimi_K25PreTrainedModel,
    Kimi_K25VisionAttention,
    Kimi_K25VisionMLP,
    Kimi_K25VisionModel,
    Kimi_K25VisionPatchEmbed,
    Kimi_K25VisionPositionEmbeddings,
    Kimi_K25VisionRotaryEmbedding,
    get_vision_temporal_merge_index,
)
from ..shensi.modeling_shensi import ShensiAttentionResidual
from ..trocr.modeling_trocr import TrOCRAttention


@auto_docstring(checkpoint="louzongzhi/Shensi-VL-V1-N")
@strict
class ShensiVLVisionConfig(PreTrainedConfig):
    r"""
    latent_size (`int`, *optional*, defaults to 256):
        Width of the latent space used by the attention-residual projections.
    merge_kernel_size (`tuple[int] | list[int]`, *optional*):
        Kernel size for patch merging.
    num_cross_attention_heads (`int`, *optional*, defaults to 8):
        Number of attention heads in the vision-to-text feedback cross-attention.
    pos_emb_height (`int`, *optional*):
        Initial position embedding height.
    pos_emb_time (`int`, *optional*):
        Initial position embedding time dimension.
    pos_emb_width (`int`, *optional*):
        Initial position embedding width.
    initializer_range (`float`, *optional*):
        Standard deviation of the normal weight initialization.
    qkv_hidden_size (`int`, *optional*, defaults to 1536):
        Hidden size of the query, key and value projections; `qkv_hidden_size // num_attention_heads`
        sets the head dimension.
    hc_conv_kernels (`list[int]`, *optional*):
        Kernel sizes of the depthwise convolutions that create the hyper-connection streams.
    hc_mult (`int`, *optional*, defaults to 1):
        Total number of parallel hyper-connection streams.
    attn_res_block_layer_types (`list[str]`, *optional*):
        Per-layer attention-residual schedule, either `"block_write_layer"` or `"block_read_layer"`.
    attn_res_block_size (`int`, *optional*, defaults to 3):
        Number of consecutive layers between two attention-residual write layers. The write layer
        stores the residual stream in the bank and the following layers read it.
    attn_res_rank (`int`, *optional*):
        Rank of the depth-routing keys and channel gates.
    """

    model_type = "shensi_vl_vision"
    default_rope_type = "axial"

    # Core model dimensions
    hidden_size: int = 1024
    intermediate_size: int = 4096
    latent_size: int = 256
    num_hidden_layers: int = 27
    num_attention_heads: int = 12
    hidden_act: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Vision tower
    patch_size: int = 14
    merge_kernel_size: tuple[int, int] | list[int] = (2, 2)
    qkv_hidden_size: int = 1536
    num_cross_attention_heads: int = 8
    pos_emb_height: int = 64
    pos_emb_time: int = 4
    pos_emb_width: int = 64
    rope_parameters: dict | None = None

    # Hyper-connections
    hc_conv_kernels: list[int] | None = None
    hc_mult: int = 1

    # Attention residuals
    attn_res_block_layer_types: list[str] | None = None
    attn_res_block_size: int = 3
    attn_res_rank: int = 32

    def __post_init__(self, **kwargs):
        if self.hc_conv_kernels is None:
            self.hc_conv_kernels = [4 * (i + 1) for i in range(self.hc_mult - 1)]
        else:
            self.hc_mult = 1 + len(self.hc_conv_kernels)

        if self.attn_res_block_layer_types is None:
            self.attn_res_block_layer_types = [
                "block_write_layer" if i % self.attn_res_block_size == 0 else "block_read_layer"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="louzongzhi/Shensi-VL-V1-N")
@strict
class ShensiVLConfig(PreTrainedConfig):
    r"""
    num_attn_res_blocks (`int`, *optional*, defaults to 9):
        Number of attention-residual blocks: each block starts with a write layer that stores the
        residual stream in one bank row, and the following layers of the block read that row. The
        Interleaved Depth Coupling (IDC) schedule runs one vision-text block per attention-residual
        block.
    """

    model_type = "shensi_vl"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="shensi"),
        "vision_config": SubConfigSpec(config_class=ShensiVLVisionConfig),
    }

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 129264
    video_token_id: int = 129265
    vision_start_token_id: int = 129266
    vision_end_token_id: int = 129267
    tie_word_embeddings: bool = True
    num_attn_res_blocks: int = 9


class ShensiVLRMSNorm(DeepseekV4RMSNorm):
    pass


class ShensiVLUnweightedRMSNorm(DeepseekV4UnweightedRMSNorm):
    pass


class ShensiVLVisionPositionEmbeddings(Kimi_K25VisionPositionEmbeddings):
    def __init__(self, config):
        super().__init__()
        self.interpolation_mode = "bilinear"


class ShensiVLVisionPatchEmbed(Kimi_K25VisionPatchEmbed):
    def __init__(self, config):
        super().__init__()
        patch_size = (
            config.patch_size if not isinstance(config.patch_size, int) else (config.patch_size, config.patch_size)
        )
        self.proj = nn.Conv2d(3, config.hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)


class ShensiVLVisionRotaryEmbedding(Kimi_K25VisionRotaryEmbedding):
    def compute_axial_rope_parameters(
        config: ShensiVLVisionConfig, device=None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.qkv_hidden_size // config.num_attention_heads
        spatial_dim = dim // 2

        attention_factor = 1.0  # Unused in this type of RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
        return inv_freq.to(device), attention_factor


class ShensiVLVisionMLP(Kimi_K25VisionMLP):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)


class ShensiVLVisionAttention(Kimi_K25VisionAttention):
    def __init__(self, config: ShensiVLVisionConfig) -> None:
        super().__init__()
        self.head_dim = config.qkv_hidden_size // self.num_heads
        self.proj = nn.Linear(config.qkv_hidden_size, self.dim, bias=False)
        self.q_proj = nn.Linear(self.dim, config.qkv_hidden_size, bias=False)
        self.k_proj = nn.Linear(self.dim, config.qkv_hidden_size, bias=False)
        self.v_proj = nn.Linear(self.dim, config.qkv_hidden_size, bias=False)


class ShensiVLVisionAttentionResidual(ShensiAttentionResidual):
    pass


class ShensiVLVisionEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ShensiVLVisionConfig, layer_idx: int):
        super().__init__()
        self.self_attn = ShensiVLVisionAttention(config)
        self.mlp = ShensiVLVisionMLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = ShensiVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ShensiVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_write = config.attn_res_block_layer_types[layer_idx] == "block_write_layer"
        # How many bank rows already exist when this layer runs: every earlier write layer feeds one
        self.prev_valid_blocks = sum(
            1 for r in config.attn_res_block_layer_types[:layer_idx] if r == "block_write_layer"
        )
        self.self_attention_attn_res = ShensiVLVisionAttentionResidual(config)
        self.mlp_attn_res = ShensiVLVisionAttentionResidual(config)

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor | None,
        prefix_sum: torch.Tensor | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # State triple: `hidden_states` is the previous sub-layer output [N, D], `prefix_sum` the block's accumulated
        # residual stream [N, D], and `residual` the written bank [N, blocks, D]. `delta` is this layer's
        # contribution, None on the first layer of a block. A write layer starts a new block, so every block owns the
        # stream between two write layers
        delta = hidden_states - prefix_sum if hidden_states is not None else None

        # A write layer snapshots the stream into its bank row before self-attention runs, so later layers read the
        # state as it entered the block rather than after this layer changed it
        if self.is_write:
            written = prefix_sum.to(residual.dtype).unsqueeze(-2)
            residual = torch.cat(
                [
                    residual[..., : self.prev_valid_blocks, :],
                    written,
                    residual[..., self.prev_valid_blocks + 1 :, :],
                ],
                dim=-2,
            )

        # The attention-residual module mixes prefix and delta into the sub-layer input; the sub-layer output is then
        # folded back into the accumulated stream
        hidden_states, prefix_sum = self.self_attention_attn_res(
            prefix_sum,
            delta,
            residual,
            num_blocks=self.prev_valid_blocks,
        )
        if self.is_write:
            # The write layer starts a new block: its stream is now the only prefix
            prefix_sum = None

        attn_output = self.self_attn(hidden_states, **kwargs)
        hidden_states = hidden_states + attn_output

        if prefix_sum is None:
            prefix_sum = hidden_states
        else:
            prefix_sum = prefix_sum + hidden_states

        # Same contract for the MLP sub-layer; `num_blocks` includes the row just written above
        hidden_states, prefix_sum = self.mlp_attn_res(
            prefix_sum,
            prefix_sum,
            residual,
            num_blocks=self.prev_valid_blocks + self.is_write,
        )

        mlp_output = self.mlp(hidden_states)
        hidden_states = hidden_states + mlp_output

        prefix_sum = prefix_sum + hidden_states
        return hidden_states, prefix_sum, residual


class ShensiVLPreTrainedModel(Kimi_K25PreTrainedModel):
    _supports_static_cache = False
    _can_compile_fullgraph = False
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, ShensiVLCrossBlock):
            init.zeros_(module.keep_ratio)


class ShensiVLVisionModel(Kimi_K25VisionModel):
    def __init__(self, config: ShensiVLVisionConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [ShensiVLVisionEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = ShensiVLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Number of attention-residual blocks: each block starts with a write layer that stores the residual stream in
        # one bank row, and the following layers of the block read that row
        self.num_attn_res_blocks = config.attn_res_block_layer_types.count("block_write_layer")
        self.output_attn_res = ShensiVLVisionAttentionResidual(config)

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, ShensiVLVisionPositionEmbeddings):
            init.copy_(module.time_position_embeddings, module.compute_pos_embed())
            init.trunc_normal_(module.position_embeddings, mean=0.0)
        elif isinstance(module, ShensiVLVisionAttentionResidual):
            std = self.config.initializer_range
            init.zeros_(module.q_proj.weight)
            init.normal_(module.k_proj.weight, mean=0.0, std=std)
            init.normal_(module.gate_down.weight, mean=0.0, std=std)
            init.normal_(module.gate_up.weight, mean=0.0, std=std)
            init.zeros_(module.gate_up.bias)
            init.ones_(module.logit_scale)
            init.constant_(module.residual_gate, 0.1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        hidden_states = self.patch_embed(pixel_values, grid_thw=grid_thw, **kwargs)
        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1, kwargs=kwargs)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        cu_seqlens, max_seqlen = get_vision_attention_seqlens(
            grid_thw, self.config, merge_temporal=True, kwargs=kwargs
        )

        # Bank of block summaries: [N, num_blocks, D], one row per attention-residual block, filled by that block's
        # write layer. The first layer starts from the patch embeddings
        residual = hidden_states.new_empty(
            hidden_states.size(0),
            self.num_attn_res_blocks,
            hidden_states.size(1),
        )
        prefix_sum = hidden_states
        hidden_states = None
        for block in self.layers:
            hidden_states, prefix_sum, residual = block(
                hidden_states,
                residual,
                prefix_sum,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        # Read out the bank after the last block: the read path returns (output, updated), and the output is then
        # RMS-normalized and merged before the multimodal projection
        hidden_states, _ = self.output_attn_res(
            prefix_sum,
            hidden_states,
            residual,
            num_blocks=self.num_attn_res_blocks,
        )

        hidden_states = self.final_layernorm(hidden_states)
        merge_index = get_vision_temporal_merge_index(grid_thw, *self.merge_kernel_size, kwargs=kwargs)
        pooled_hidden_states = hidden_states[merge_index].mean(dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_hidden_states,
        )


class ShensiVLMultimodalProjection(Kimi_K25MultimodalProjection):
    def __init__(self, config):
        super().__init__()
        del self.pre_norm

        self.in_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, config.text_config.hidden_size, bias=False)
        self.post_norm = ShensiVLRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = self.out_proj(self.act(self.in_proj(hidden_states.view(batch_size, -1, self.hidden_size))))
        return self.post_norm(hidden_states)


class ShensiVLCrossAttention(TrOCRAttention):
    pass


class ShensiVLCrossBlock(GradientCheckpointingLayer):
    # Upper bound on (#patch queries x #key tokens) for one feedback attention call; longer batches are split into
    # several calls so peak memory stays bounded
    attn_budget = 1 << 23

    def __init__(
        self,
        config: ShensiVLConfig,
        block_idx: int,
        vision_modules: list,
        text_modules: list,
        adapter: dict,
    ):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        # Each block picks its own slice of both towers. The shared adapter is passed in a plain dict, which
        # `nn.Module.__setattr__` stores as a regular attribute, so its parameters are not registered again and appear
        # only once in the state dict
        self.vision_modules = vision_modules[block_idx]
        self.text_modules = text_modules[block_idx]
        self.adapter = adapter

        # The last block finalizes both towers; earlier blocks cross-attend the text states back into the vision
        # stream, gated by tanh(alpha_cross_attn), which starts at zero (feedback off)
        self.has_next_block = block_idx + 1 < config.num_attn_res_blocks
        self.cross_attn = (
            ShensiVLCrossAttention(
                config,
                embed_dim=config.vision_config.hidden_size,
                num_heads=config.vision_config.num_cross_attention_heads,
                kdim=config.text_config.hidden_size,
                vdim=config.text_config.hidden_size,
                bias=False,
            )
            if self.has_next_block
            else None
        )
        self.alpha_cross_attn = nn.Parameter(torch.zeros(1)) if self.has_next_block else None
        # Learned scalar mixing the projected features with the states already at the placeholders: 0 overwrites them,
        # 1 keeps them, and the value is shared by every placeholder of the batch
        self.keep_ratio = nn.Parameter(torch.zeros(1))

    def _run_layers(self, modules, metadata, **kwargs):
        # Run one tower slice, threading (hidden_states, prefix_sum, residual) through its layers; `metadata` is the
        # block's own copy, so updating it in place is safe when replayed
        for layer in modules["layers"]:
            outputs = layer(
                hidden_states=metadata["hidden_states"],
                prefix_sum=metadata["prefix_sum"],
                residual=metadata["residual"],
                prev_topk_indices=metadata.get("topk_indices"),
                **kwargs,
            )
            if len(outputs) == 4:
                hidden_states, prefix_sum, residual, topk_indices = outputs
                metadata["topk_indices"] = topk_indices
            else:
                hidden_states, prefix_sum, residual = outputs
            metadata.update(hidden_states=hidden_states, prefix_sum=prefix_sum, residual=residual)

    def _apply_output_attn_res(self, modules, metadata):
        # Final read-out of a tower: read the bank up to this block, then apply the tower's final norm. `block_idx +
        # 1` rows are visible because block i's write layer has already run
        hidden_states, _ = modules["output_attn_res"](
            metadata["prefix_sum"],
            metadata["hidden_states"],
            metadata["residual"],
            num_blocks=self.block_idx + 1,
        )
        metadata["hidden_states"] = modules["norm"](hidden_states)

    def _run_text_layers(self, modules, metadata):
        model = modules["model"]
        for layer in modules["layers"]:
            layer_blocks = model.bank_attn_res.read(metadata["blocks"], metadata["inputs_embeds"])
            hidden_states, partial_block, residual, topk_indices = layer(
                metadata["streams"],
                partial_block=metadata["partial_block"],
                blocks=layer_blocks,
                input_ids=metadata["input_ids"],
                attention_mask=metadata["attention_mask"],
                position_ids=metadata["position_ids"],
                past_key_values=metadata["past_key_values"],
                position_embeddings=None,
                prev_topk_indices=metadata["topk_indices"],
                rosa_token_ids=None,
            )
            metadata["streams"] = hidden_states
            metadata["partial_block"] = partial_block
            metadata["blocks"] = model.bank_attn_res.encode(residual[..., 1:, :])
            metadata["topk_indices"] = topk_indices

    def _apply_text_output(self, modules, metadata):
        model = modules["model"]
        num_blocks = model.num_attn_res_blocks
        encoded = model.bank_attn_res.write(metadata["blocks"], num_blocks, metadata["partial_block"])
        residual = model.bank_attn_res.read(encoded, metadata["inputs_embeds"])
        hidden_states = model.hc_head(metadata["streams"])
        hidden_states, _ = model.output_attn_res(
            metadata["partial_block"],
            hidden_states,
            residual,
            num_blocks=1 + num_blocks,
        )
        metadata["hidden_states"] = model.norm(hidden_states)

    def _cross_attend(self, hidden_states, keys, clips, gate, text_states=None, context_mask=None, clip_rows=None):
        clip_rows = clip_rows if clip_rows is not None else [None] * len(clips)
        extra = 0 if context_mask is None else int(context_mask.sum())
        # Group consecutive clips so that (#patch queries x #key tokens) stays under `attn_budget`. Each clip attends
        # only to its own keys, so a block-diagonal mask makes one call per group exactly equivalent to one call per
        # clip, just with fewer kernel launches
        groups, start, patches, tokens = [], 0, 0, 0
        for index, (patch_range, token_range) in enumerate(clips):
            patch_len = patch_range.stop - patch_range.start
            token_len = token_range.stop - token_range.start + extra
            if index > start and (patches + patch_len) * (tokens + token_len) > self.attn_budget:
                groups.append((start, index))
                start, patches, tokens = index, 0, 0
            patches += patch_len
            tokens += token_len
        groups.append((start, len(clips)))

        outputs = []
        for first, last in groups:
            group = clips[first:last]
            patch_start, patch_end = group[0][0].start, group[-1][0].stop
            token_start = group[0][1].start
            # Query: this group's vision patches. Key/value: the text states at the group's placeholder positions,
            # plus the shared question-region states packed below
            queries = hidden_states[patch_start:patch_end].unsqueeze(0)
            key_value_states = keys[token_start : group[-1][1].stop]

            # Each distinct text row is packed once; `offsets[row]` records where its states start, so the mask can
            # point every clip of that row at the right columns
            context_states = None
            offsets = {}
            if text_states is not None and context_mask is not None and clip_rows[first] is not None:
                pieces = []
                for row in dict.fromkeys(clip_rows[first:last]):
                    picked = text_states[row][context_mask[row]]
                    offsets[row] = (sum(p.shape[0] for p in pieces), picked)
                    pieces.append(picked)
                if pieces:
                    context_states = torch.cat(pieces, dim=0)

            attention_mask = None
            key_value_states = key_value_states.unsqueeze(0)
            if context_states is not None:
                key_value_states = torch.cat([key_value_states, context_states.unsqueeze(0)], dim=1)
            if len(group) > 1 or context_states is not None:
                # Block-diagonal mask: clip i sees its own text tokens plus the context columns of its row; `base`
                # shifts the packed context offset onto the key axis. Everything else is -inf, so clips in the same
                # group never leak into each other
                attention_mask = queries.new_full(
                    (1, 1, queries.shape[1], key_value_states.shape[1]), torch.finfo(queries.dtype).min
                )
                for offset, (patch_range, token_range) in enumerate(group):
                    rows = slice(patch_range.start - patch_start, patch_range.stop - patch_start)
                    cols = slice(token_range.start - token_start, token_range.stop - token_start)
                    attention_mask[0, 0, rows, cols] = 0
                    row = clip_rows[first + offset]
                    if row is not None and row in offsets:
                        start_col, picked = offsets[row]
                        base = group[-1][1].stop - token_start
                        qcols = slice(base + start_col, base + start_col + picked.shape[0])
                        attention_mask[0, 0, rows, qcols] = 0
            update, _ = self.cross_attn(queries, key_value_states=key_value_states, attention_mask=attention_mask)
            outputs.append(queries.squeeze(0) + gate * update.squeeze(0).to(queries.dtype))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        vision_metadata: dict | None,
        text_metadata: dict,
    ) -> tuple[dict | None, dict]:
        # Shallow copies: gradient checkpointing replays this forward, so the block must not mutate the caller's
        # metadata dicts in place
        vision_metadata = dict(vision_metadata) if vision_metadata is not None else None
        text_metadata = dict(text_metadata)

        # One Interleaved Depth Coupling (IDC) step: vision half (layers, last-block finalize), merge and project the
        # patches, write them into the text placeholders, then the text half. This injects vision features at several
        # depths instead of once, and every block but the last feeds the text states back into the vision stream
        # through the gated cross-attention, so deep vision layers are conditioned on what the text tower has already
        # read
        if vision_metadata is not None:
            self._run_layers(
                self.vision_modules,
                vision_metadata,
                cu_seqlens=vision_metadata["cu_seqlens"],
                max_seqlen=vision_metadata["max_seqlen"],
                position_embeddings=vision_metadata["position_embeddings"],
            )
            if not self.has_next_block:
                self._apply_output_attn_res(self.vision_modules, vision_metadata)

            # Merge each `merge_kernel_size` group of patches and project them to the text space: [N, kernel_volume,
            # D] -> [N, D_text]
            merge_indices = vision_metadata["merge_indices"]
            merged = torch.cat(
                [vision_metadata["hidden_states"][merge_index].mean(dim=1) for merge_index in merge_indices]
            )
            visual = self.adapter["module"](merged).squeeze(1)
            # Re-injection: `keep_ratio` blends the projected features with the placeholder states that already exist
            # there (0 = hard overwrite, 1 = keep the text states)
            keep = self.keep_ratio.clamp(0.0, 1.0)
            for mask, features in zip(
                vision_metadata["masks"], torch.split(visual, [index.size(0) for index in merge_indices])
            ):
                partial_block = text_metadata["partial_block"]
                features = features.to(partial_block.dtype)
                keep_here = keep.to(partial_block.dtype)
                mixed = (1.0 - keep_here) * features + keep_here * partial_block[mask.squeeze(-1)]
                text_metadata["partial_block"] = partial_block.masked_scatter(mask, mixed)
                streams = text_metadata["streams"]
                stream0 = streams[..., 0, :]
                keep_here = keep.to(stream0.dtype)
                mixed = (1.0 - keep_here) * features.to(stream0.dtype) + keep_here * stream0[mask.squeeze(-1)]
                stream0 = stream0.masked_scatter(mask, mixed)
                text_metadata["streams"] = torch.cat([stream0.unsqueeze(2), streams[..., 1:, :]], dim=2)

        # Text half: run the matching text layers on the stream updated with the vision features
        self._run_text_layers(self.text_modules, text_metadata)
        if not self.has_next_block:
            self._apply_text_output(self.text_modules, text_metadata)
            return vision_metadata, text_metadata

        if vision_metadata is not None:
            # Feedback: the vision stream reads the text states processed so far through the gated cross-attention, so
            # the next vision block is conditioned on the text
            gate = torch.tanh(self.alpha_cross_attn)
            hidden_states = vision_metadata["hidden_states"]
            updated = []
            for mask, clips, rows in zip(
                vision_metadata["masks"], vision_metadata["clip_layouts"], vision_metadata["clip_rows"]
            ):
                keys = text_metadata["partial_block"][mask.squeeze(-1)]
                updated.append(
                    self._cross_attend(
                        hidden_states,
                        keys,
                        clips,
                        gate,
                        text_states=text_metadata["partial_block"],
                        context_mask=vision_metadata["context_mask"],
                        clip_rows=rows,
                    )
                )
            vision_metadata["hidden_states"] = torch.cat(updated, dim=0)
        return vision_metadata, text_metadata


class ShensiVLModelOutputWithPast(Kimi_K25ModelOutputWithPast):
    pass


class ShensiVLModel(Kimi_K25Model):
    def __init__(self, config: ShensiVLConfig):
        super().__init__(config)

        self.language_model.num_attn_res_blocks = self.language_model.config.attn_res_block_layer_types.count(
            "block_write_layer"
        )

        modules = self._init_modules(self.vision_tower, self.language_model, self.mm_projector)
        # Number of attention-residual blocks: each block starts with a write layer that stores the residual stream in
        # one bank row, and the following layers of the block read that row. The Interleaved Depth Coupling (IDC)
        # schedule runs one vision-text block per attention-residual block
        self.blocks = nn.ModuleList(
            ShensiVLCrossBlock(config, block_idx, **modules) for block_idx in range(config.num_attn_res_blocks)
        )

    @staticmethod
    def _init_modules(vision_tower, language_model, mm_projector):
        # A new block starts at every write layer: `bounds` pairs consecutive write layers, so block i owns the layers
        # in [bounds[i], bounds[i + 1]) plus the tower's read-out module and final norm
        modules = {}
        for name, tower in (("vision_modules", vision_tower), ("text_modules", language_model)):
            starts = [index for index, layer in enumerate(tower.layers) if layer.is_write]
            bounds = [*starts, len(tower.layers)]
            block_layers = [
                [tower.layers[index] for index in range(start, end)] for start, end in zip(bounds, bounds[1:])
            ]
            # The text tower exposes `norm`, the vision tower `final_layernorm`
            norm = getattr(tower, "norm", None) or tower.final_layernorm
            modules[name] = [
                {"layers": layers, "output_attn_res": tower.output_attn_res, "norm": norm} for layers in block_layers
            ]
            if name == "text_modules":
                for module in modules[name]:
                    module["model"] = language_model

        # The adapter is shared by all blocks and passed in a plain dict, so it is stored as a regular attribute and
        # its parameters appear only once in the state dict
        return {**modules, "adapter": {"module": mm_projector}}

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        # The text tower owns the vocabulary tables, including the tied ROSA, indexer and hyper-connection embeddings,
        # so resize through it
        return self.language_model._resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        vision_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, **kwargs)
        image_embeds = self.mm_projector(vision_outputs.pooler_output).squeeze(1)
        # Pooler_output is [total_merged, kernel_volume, D]; the projector folds the kernel volume into the feature.
        # Split per image so each one yields (h / kh) * (w / kw) tokens, matching the per-image placeholder layout of
        # the text stream
        split_sizes = (
            (image_grid_thw[:, 1] // self.vision_tower.merge_kernel_size[0])
            .mul(image_grid_thw[:, 2] // self.vision_tower.merge_kernel_size[1])
            .tolist()
        )
        vision_outputs.pooler_output = torch.split(image_embeds, split_sizes)
        return vision_outputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | ShensiVLModelOutputWithPast:
        # Interleaved Depth Coupling (IDC): the two towers run block by block. Each block runs a slice of vision
        # layers, writes the projected patches into the image placeholder positions, runs the matching text layers,
        # and (except on the last block) feeds the text states back into the vision stream. Vision features therefore
        # enter the text stream at several depths, and deep vision layers can condition on the text read so far
        # instead of one-shot injection
        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        # Embed the text tokens only: the placeholders are overwritten by the projected features
        llm_input_ids = None
        if input_ids is not None:
            multimodal_mask = (input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = 0
        if inputs_embeds is None:
            # Keep the placeholder embeddings: they are overwritten in the streams below, and the `inputs_embeds` path
            # cannot mask them without the token ids
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Features may come pre-computed; otherwise the interleaved blocks encode the raw patches
        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        image_features = mm_encoder_outputs.get("image")
        if image_features is not None:
            image_embeds = torch.cat(image_features.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        video_features = mm_encoder_outputs.get("video")
        if video_features is not None:
            video_embeds = torch.cat(video_features.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if use_cache and past_key_values is None:
            past_key_values = self.language_model._create_cache()
        if past_key_values is not None:
            self.language_model._tie_indexed_caches(past_key_values)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_recurrent_attention_mask(
                config=self.language_model.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            if causal_mask is None:
                causal_mask = torch.ones(
                    inputs_embeds.shape[0], inputs_embeds.shape[1], dtype=torch.bool, device=inputs_embeds.device
                )
            causal_mask = causal_mask.bool()
        # Text metadata carries the hyper-connection streams, the encoded block bank and the running partial block
        # across interleaved blocks
        num_attn_res_blocks = self.language_model.num_attn_res_blocks
        streams = self.language_model.stream_expansion(inputs_embeds)
        text_metadata = {
            "streams": streams,
            "blocks": self.language_model.bank_attn_res.initialize(inputs_embeds, num_attn_res_blocks),
            "partial_block": torch.zeros_like(inputs_embeds),
            "inputs_embeds": inputs_embeds,
            "input_ids": llm_input_ids,
            "topk_indices": None,
            "position_ids": position_ids,
            "attention_mask": causal_mask,
            "past_key_values": past_key_values,
        }

        vision_metadata = None
        # Per-modality pairing used by the feedback: `merge_indices` maps merged tokens to patches, `clip_layouts`
        # pairs each sample's patch slice with its text-token slice, and `clip_rows` records which text row a clip may
        # read `context_mask` states from
        masks, merge_indices, clip_layouts, clip_rows = [], [], [], []
        context_mask = kwargs.pop("context_mask", None)
        if pixel_values is not None or pixel_values_videos is not None:
            image_mask, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            kernel_height, kernel_width = self.vision_tower.merge_kernel_size
            patch_start = 0
            for name, values, grid_thw, mask in (
                ("image", pixel_values, image_grid_thw, image_mask),
                ("video", pixel_values_videos, video_grid_thw, video_mask),
            ):
                if values is None:
                    continue
                token_start = 0
                # Cumulative ends of each sample's merged tokens and of its raw patches
                grid_ends = (grid_thw[:, 1] // kernel_height).mul(grid_thw[:, 2] // kernel_width).cumsum(0)
                patch_ends = grid_thw.prod(-1).cumsum(0)
                torch_compilable_check(
                    mask.sum() == grid_ends[-1],
                    f"{name.capitalize()} features and {name} tokens do not match, tokens: {mask.sum()}, "
                    f"features: {grid_ends[-1]}",
                )
                merge_indices.append(
                    get_vision_temporal_merge_index(grid_thw, kernel_height, kernel_width, kwargs=kwargs) + patch_start
                )
                clips = []
                patch_end, token_end = patch_start, token_start
                # trf-ignore: TRF056
                for patch_stop, token_stop in torch.stack((patch_ends, grid_ends), dim=1).tolist():
                    clips.append(
                        (slice(patch_end, patch_start + patch_stop), slice(token_end, token_start + token_stop))
                    )
                    patch_end, token_end = patch_start + patch_stop, token_start + token_stop
                masks.append(mask)
                clip_layouts.append(clips)
                # Map each sample's cumulative merged-token end back to the row that owns it
                row_ends = mask.squeeze(-1).sum(1).cumsum(0)
                # trf-ignore: TRF056
                clip_rows.append(torch.searchsorted(row_ends, grid_ends - 1, right=True).tolist())
                patch_start = patch_end

            multimodal_mask = (image_mask | video_mask).squeeze(-1)
            # The placeholders precede the question, so the causal text layers never see the question when the vision
            # stream reads them. Default to the tokens after the last placeholder; callers can pass their own
            # `context_mask` to choose the readable region
            if context_mask is not None:
                context_mask = context_mask.bool() & ~multimodal_mask
            elif input_ids is not None:
                positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                last_vision = (positions * multimodal_mask).max(dim=1, keepdim=True).values
                context_mask = (positions > last_vision) & ~multimodal_mask

            # Both modalities share one patch sequence: `clip_layouts` keeps the per-sample slices and `patch_start`
            # shifts the merge indices of the second modality accordingly
            vision_pixel_values = torch.cat(
                [values for values in (pixel_values, pixel_values_videos) if values is not None]
            )
            vision_grid_thw = torch.cat([grid for grid in (image_grid_thw, video_grid_thw) if grid is not None])
            vision_prefix_sum = self.vision_tower.patch_embed(vision_pixel_values, grid_thw=vision_grid_thw, **kwargs)
            cu_seqlens, max_seqlen = get_vision_attention_seqlens(
                vision_grid_thw, self.vision_tower.config, merge_temporal=True, kwargs=kwargs
            )
            vision_metadata = {
                "hidden_states": None,
                "prefix_sum": vision_prefix_sum,
                "residual": vision_prefix_sum.new_empty(
                    vision_prefix_sum.size(0),
                    self.vision_tower.num_attn_res_blocks,
                    vision_prefix_sum.size(1),
                ),
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
                "position_embeddings": self.vision_tower.rotary_emb(
                    vision_prefix_sum,
                    get_vision_position_ids(
                        vision_grid_thw,
                        spatial_merge_size=1,
                        kwargs=kwargs,
                    ),
                ),
                "masks": masks,
                "merge_indices": merge_indices,
                "clip_layouts": clip_layouts,
                "clip_rows": clip_rows,
                "context_mask": context_mask,
            }

        # Run the interleaved schedule: vision layers, reinjection, text layers, feedback
        for block in self.blocks:
            vision_metadata, text_metadata = block(
                vision_metadata,
                text_metadata,
            )

        return ShensiVLModelOutputWithPast(
            last_hidden_state=text_metadata["hidden_states"],
            past_key_values=text_metadata["past_key_values"],
        )


class ShensiVLCausalLMOutputWithPast(Kimi_K25CausalLMOutputWithPast):
    pass


class ShensiVLForConditionalGeneration(Kimi_K25ForConditionalGeneration):
    def _prepare_multimodal_encoder_kwargs_for_generation(self, model_kwargs):
        # The interleaved blocks need the raw patches: vision layers condition on the text states produced at the same
        # depth, which cannot be recovered from precomputed image features
        return model_kwargs

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        embeddings = self.model.language_model._resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of, mean_resizing
        )
        if new_num_tokens is None and pad_to_multiple_of is None:
            return embeddings
        # The output head sits on the wrapper, not on the text tower, so resize it here
        old_lm_head = self.get_output_embeddings()
        if old_lm_head is not None:
            new_lm_head = self._get_resized_lm_head(
                old_lm_head, embeddings.weight.shape[0], mean_resizing=mean_resizing
            )
            new_lm_head.requires_grad_(old_lm_head.weight.requires_grad)
            self.set_output_embeddings(new_lm_head)
        return embeddings


__all__ = [
    "ShensiVLConfig",
    "ShensiVLVisionConfig",
    "ShensiVLForConditionalGeneration",
    "ShensiVLModel",
    "ShensiVLPreTrainedModel",
    "ShensiVLVisionModel",
]
