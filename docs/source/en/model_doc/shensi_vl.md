<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-02.*


# ShensiVL

## Overview

ShensiVL couples a Kimi-K3 vision tower with a Shensi text tower through **Interleaved Depth Coupling
(IDC)**. Both towers are split into attention-residual blocks, and each block runs the two towers in an
interleaved schedule: the vision layers run first, the merged patches are projected into the text
embedding space and written into the image placeholder positions, and then the matching text layers
run. Every block but the last also feeds the text states of the question region back into the vision
stream through a gated cross-attention, so deeper vision layers are conditioned on the text read so
far. Vision features therefore enter the text stream at several depths instead of a single one-shot
injection.

This model was contributed by [louzongzhi](https://huggingface.co/louzongzhi).

## Usage examples

See the [model card](https://huggingface.co/louzongzhi/Shensi-VL-V1-N) for image + text usage examples.

## ShensiVLConfig

[[autodoc]] ShensiVLConfig

## ShensiVLVisionConfig

[[autodoc]] ShensiVLVisionConfig

## ShensiVLForConditionalGeneration

[[autodoc]] ShensiVLForConditionalGeneration

## ShensiVLModel

[[autodoc]] ShensiVLModel
    - forward

## ShensiVLPreTrainedModel

[[autodoc]] ShensiVLPreTrainedModel
    - forward

## ShensiVLVisionModel

[[autodoc]] ShensiVLVisionModel
    - forward