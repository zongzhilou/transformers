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

from ..kimi_k25.configuration_kimi_k25 import Kimi_K25Config, Kimi_K25VisionConfig
from ..kimi_k25.modeling_kimi_k25 import (
    Kimi_K25CausalLMOutputWithPast,
    Kimi_K25ForConditionalGeneration,
    Kimi_K25Model,
    Kimi_K25ModelOutputWithPast,
    Kimi_K25MultimodalProjection,
    Kimi_K25PreTrainedModel,
    Kimi_K25VisionAttention,
    Kimi_K25VisionEncoderLayer,
    Kimi_K25VisionMLP,
    Kimi_K25VisionModel,
    Kimi_K25VisionPatchEmbed,
    Kimi_K25VisionPositionEmbeddings,
    Kimi_K25VisionRotaryEmbedding,
)


class ShensiVlVisionConfig(Kimi_K25VisionConfig):
    pass


class ShensiVlConfig(Kimi_K25Config):
    pass


class ShensiVlModelOutputWithPast(Kimi_K25ModelOutputWithPast):
    pass


class ShensiVlCausalLMOutputWithPast(Kimi_K25CausalLMOutputWithPast):
    pass


class ShensiVlVisionPositionEmbeddings(Kimi_K25VisionPositionEmbeddings):
    pass


class ShensiVlVisionPatchEmbed(Kimi_K25VisionPatchEmbed):
    pass


class ShensiVlVisionRotaryEmbedding(Kimi_K25VisionRotaryEmbedding):
    pass


class ShensiVlVisionMLP(Kimi_K25VisionMLP):
    pass


class ShensiVlVisionAttention(Kimi_K25VisionAttention):
    pass


class ShensiVlVisionEncoderLayer(Kimi_K25VisionEncoderLayer):
    pass


class ShensiVlPreTrainedModel(Kimi_K25PreTrainedModel):
    pass


class ShensiVlVisionModel(Kimi_K25VisionModel):
    pass


class ShensiVlMultimodalProjection(Kimi_K25MultimodalProjection):
    pass


class ShensiVlModel(Kimi_K25Model):
    pass


class ShensiVlForConditionalGeneration(Kimi_K25ForConditionalGeneration):
    pass


__all__ = [
    "ShensiVlConfig",
    "ShensiVlVisionConfig",
    "ShensiVlForConditionalGeneration",
    "ShensiVlModel",
    "ShensiVlPreTrainedModel",
    "ShensiVlVisionModel",
]
