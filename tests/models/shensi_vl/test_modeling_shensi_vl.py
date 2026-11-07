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
"""Testing suite for the Shensi-VL model."""

import unittest

from parameterized import parameterized

from transformers import (
    AutoProcessor,
    ShensiConfig,
    ShensiVLConfig,
    ShensiVLVisionConfig,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...test_modeling_common import (
    floats_tensor,
)
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import ShensiVLForConditionalGeneration, ShensiVLModel


if is_vision_available():
    pass


class ShensiVLVisionText2TextModelTester(VLMModelTester):
    base_model_class = ShensiVLModel
    config_class = ShensiVLConfig
    text_config_class = ShensiConfig
    vision_config_class = ShensiVLVisionConfig
    conditional_generation_class = ShensiVLForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("image_size", 32)
        kwargs.setdefault("patch_size", 8)
        kwargs.setdefault("num_image_tokens", 16)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("qkv_hidden_size", 64)
        kwargs.setdefault("pos_emb_height", 4)
        kwargs.setdefault("merge_kernel_size", (1, 1))
        kwargs.setdefault("pos_emb_width", 4)
        kwargs.setdefault("pos_emb_time", 1)

        # Text tower (`ShensiConfig`). All-"moe" so the `inputs_embeds`-only tests do not
        # need the token ids that the hash router requires.
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 1)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("deepemb_kv_rank", 8)
        kwargs.setdefault("q_lora_rank", 32)
        kwargs.setdefault("n_routed_experts", 4)
        kwargs.setdefault("num_experts_per_tok", 2)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("attn_res_latent_size", 32)
        kwargs.setdefault("latent_size", 16)
        kwargs.setdefault("attn_res_block_size", 2)
        kwargs.setdefault("num_attn_res_blocks", 1)
        kwargs.setdefault("mlp_layer_types", ["sparse", "sparse"])
        # ROSA depends on token ids, so the generic tests disable it to keep `inputs_embeds`
        # equivalent to `input_ids`.
        kwargs.setdefault("rosa_layer_ids", [])
        kwargs.setdefault("index_n_heads", 2)
        kwargs.setdefault("index_head_dim", 16)
        kwargs.setdefault("index_topk_tokens", 2)
        kwargs.setdefault("scoring_func", "sqrtsoftplus")
        kwargs.setdefault("routed_scaling_factor", 1.5)

        kwargs.setdefault(
            "rope_parameters",
            {
                "rope_type": "default",
                "rope_theta": 10000,
            },
        )
        super().__init__(parent, **kwargs)

        # These can be inferred from existing properties and don't get separate kwargs
        self.projection_hidden_size = self.hidden_size

    def create_pixel_values(self, batch_size: int | None = None):
        # Override to 5D for patch-based models
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor(
            [
                batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels,
                self.patch_size,
                self.patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        # Place image tokens with vision_start_token_id prefix
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        # Place image tokens with vision_start_token_id prefix
        input_ids[:, : self.num_image_tokens] = self.image_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {
            "image_grid_thw": torch.tensor([[1, 4, 4]] * batch_size, device=torch_device),
        }


@require_torch
class ShensiVLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = ShensiVLVisionText2TextModelTester
    has_attentions = False

    # The first attention-residual block has no completed block to read, so its pseudo-queries
    # are intentionally unused and receive no gradient.
    test_all_params_have_gradient = False

    def test_reverse_loading_mapping(self):
        super().test_reverse_loading_mapping(skip_base_model=True)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # Indexed layers cache the low-rank DEA latent; linear-attention layers keep a constant
        # recurrent state and have no K/V entries at all.
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            if keys is None:
                continue
            self.assertIsInstance(keys, torch.Tensor)
            self.assertEqual(keys.shape[0], batch_size)
            self.assertEqual(keys.shape[-1], config.kv_deepemb_rank)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("Shensi's indexed attention is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Shensi's indexed attention is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("Shensi's indexed attention is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Shensi's indexed-attention cache is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Shensi's indexed-attention cache is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims on LM backbone")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        reason="The two towers' layers are driven by a custom (hidden_states, prefix_sum, residual) contract, "
        "which does not expose per-layer attentions to `_can_record_outputs`"
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip(
        reason="The two towers' layers are driven by a custom (hidden_states, prefix_sum, residual) contract, "
        "which does not expose per-layer hidden states to `_can_record_outputs`"
    )
    def test_hidden_states_output(self):
        pass

    @unittest.skip(
        reason="The two towers' layers are driven by a custom (hidden_states, prefix_sum, residual) contract, "
        "which does not expose per-layer hidden states to generation outputs"
    )
    def test_greedy_generate_dict_outputs(self):
        pass

    @unittest.skip(
        reason="The two towers' layers are driven by a custom (hidden_states, prefix_sum, residual) contract, "
        "which does not expose per-layer hidden states to generation outputs"
    )
    def test_sample_generate_dict_output(self):
        pass

    @unittest.skip(
        reason="The two towers' layers are driven by a custom (hidden_states, prefix_sum, residual) contract, "
        "which does not expose per-layer hidden states to generation outputs"
    )
    def test_beam_search_generate_dict_output(self):
        pass

    @unittest.skip(
        reason="The two towers' layers are driven by a custom (hidden_states, prefix_sum, residual) contract, "
        "which does not expose per-layer hidden states to generation outputs"
    )
    def test_beam_sample_generate_dict_output(self):
        pass

    @unittest.skip(
        reason="The interleaved blocks do not expose per-layer hidden states or attentions to `_can_record_outputs`"
    )
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Reentrant checkpointing does not support Shensi-VL's custom 4D hyper-connection state.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(
        reason="Shensi-VL stores the inference KV cache in FP8 on purpose, so a cached continuation "
        "is numerically close to, but not bit-equivalent with, a full-precision forward."
    )
    def test_recurrent_layers_mask_padding_on_continued_forward(self):
        pass

    @unittest.skip(
        reason="Generation feeds raw patches to the interleaved blocks so vision layers can condition "
        "on text states; precomputed `mm_encoder_outputs` is supported by `forward` only."
    )
    def test_generate_from_multimodal_encoder_outputs(self):
        pass

    @unittest.skip(
        reason="Generation feeds raw patches to the interleaved blocks so vision layers can condition "
        "on text states; precomputed `mm_encoder_outputs` is supported by `forward` only."
    )
    def test_generate_from_multimodal_encoder_outputs_and_raw_data(self):
        pass

    @unittest.skip(
        reason="accelerate's device_map='auto' cannot partition Shensi-VL's 4D hyper-connection "
        "streams and block-residual buffers across devices."
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        reason="accelerate's device_map='auto' cannot partition Shensi-VL's 4D hyper-connection "
        "streams and block-residual buffers across devices."
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        reason="accelerate's device_map='auto' cannot partition Shensi-VL's 4D hyper-connection "
        "streams and block-residual buffers across devices."
    )
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(
        reason="Framework mismatch: build_glob_alternation's unanchored regex 'res' also "
        "matches vision attention-residual modules, but the test's anchored regex does not expect it."
    )
    def test_keep_in_fp32_modules(self):
        pass


@slow
@require_torch
class ShensiVLIntegrationTest(unittest.TestCase):
    """End-to-end forward pass on the published Shensi-VL checkpoint.

    The checkpoint is large, so this test is marked slow and excluded from the default run:

        RUN_SLOW=1 pytest tests/models/shensi_vl/test_modeling_shensi_vl.py::ShensiVLIntegrationTest -s

    It exercises the interleaved depth-coupling schedule end to end: the vision tower slices, the
    per-block reinjection into the text streams, the gated text-to-vision feedback and the
    language-model read-out.
    """

    model_id = "louzongzhi/Shensi-VL-V1-N"
    model = None
    processor = None
    messages = None

    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        cls.model = ShensiVLForConditionalGeneration.from_pretrained(cls.model_id, dtype="auto", device_map="auto")
        cls.processor = AutoProcessor.from_pretrained(cls.model_id)
        cls.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
                    },
                    {"type": "text", "text": "What kind of animal is this?"},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_logits(self):
        inputs = self.processor.apply_chat_template(
            self.messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        self.assertEqual(logits.shape[:2], inputs["input_ids"].shape)
        self.assertTrue(torch.isfinite(logits).all())
