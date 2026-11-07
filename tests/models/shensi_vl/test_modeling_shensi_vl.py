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
"""Testing suite for the PyTorch Kimi2.6 model."""

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
    Expectations,
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
        kwargs.setdefault("latent_size", 16)
        kwargs.setdefault("attn_res_block_size", 2)
        kwargs.setdefault("num_attn_res_blocks", 1)
        kwargs.setdefault("mlp_layer_types", ["sparse", "sparse"])
        # ROSA depends on token ids, so the generic tests disable it to keep `inputs_embeds`
        # equivalent to `input_ids`.
        kwargs.setdefault("rosa_layer_ids", [])
        kwargs.setdefault("index_n_heads", 2)
        kwargs.setdefault("index_head_dim", 16)
        kwargs.setdefault("index_topk", 2)
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
            self.assertEqual(keys.shape[-1], config.deepemb_kv_rank)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV3 backbone is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("Deepseek-V3 backbone uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("Deepseek-V3 backbone uses MLA so it is not compatible with the standard cache format")
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
class KimiK25IntegrationTest(unittest.TestCase):
    model_id = "hf-internal-testing/kimi-k25-for-integration-test"
    model = None
    processor = None

    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        cls.model = ShensiVLForConditionalGeneration.from_pretrained(cls.model_id, device_map="auto")
        cls.processor = AutoProcessor.from_pretrained(cls.model_id)

        cls.message1 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        cls.message2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/coco_sample.png"
                        ),
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        cls.message3 = [{"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_logits_text_only(self):
        expectations = Expectations(
            {
                ("cuda", None): [
                    [[ 0.00000000, -0.02641739,  0.01707786, -0.08232787,  0.12016402, -0.15289734, -0.08252842,
                    0.15658118, -0.04173164,  0.10555173],
                    [ 0.00000000, -0.02639876,  0.01706616, -0.08229892,  0.12015337, -0.15289734, -0.08251060,
                    0.15658717, -0.04172155,  0.10556119],
                    [ 0.00000000, -0.02639606,  0.01708876, -0.08232962,  0.12015553, -0.15291165, -0.08253470,
                    0.15658627, -0.04173409,  0.10555199],
                    [ 0.00000000, -0.02638427,  0.01708288, -0.08230965,  0.12014508, -0.15289523, -0.08252528,
                    0.15660310, -0.04173251,  0.10554133],
                    [ 0.00000000, -0.02639318,  0.01706917, -0.08231460,  0.12016111, -0.15290320, -0.08252579,
                    0.15659124, -0.04173930,  0.10554294],
                    [ 0.00000000, -0.02638769,  0.01707737, -0.08231344,  0.12015510, -0.15289856, -0.08252031,
                    0.15659934, -0.04173541,  0.10554094],
                    [ 0.00000000, -0.02639529,  0.01708556, -0.08231322,  0.12014811, -0.15291123, -0.08251978,
                    0.15660490, -0.04173838,  0.10553965],
                    [ 0.00000000, -0.02639412,  0.01707170, -0.08232507,  0.12014882, -0.15289663, -0.08251097,
                    0.15659560, -0.04172803,  0.10553968],
                    [ 0.00000000, -0.02639644,  0.01707644, -0.08233012,  0.12013669, -0.15289611, -0.08249903,
                    0.15660135, -0.04172298,  0.10554458],
                    [ 0.00000000, -0.02639807,  0.01707577, -0.08232461,  0.12014168, -0.15288822, -0.08250540,
                    0.15660320, -0.04172593,  0.10553741]]
                ],
            }
        )  # fmt: skip
        expectations_mean = Expectations({("cuda", None): -0.014419052749872208})

        inputs = self.processor.apply_chat_template(
            self.message3, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        torch.testing.assert_close(logits[:, :10, :10].cpu(), torch.tensor(expectations.get_expectation()))
        torch.testing.assert_close(logits.mean().cpu(), torch.tensor(expectations_mean.get_expectation()))

    def test_model_logits(self):
        expectations = Expectations(
            {
                ("cuda", None): [
                    [[ 0.00000000, -0.02641739,  0.01707786, -0.08232787,  0.12016402, -0.15289734, -0.08252842,
                    0.15658118, -0.04173164,  0.10555173],
                    [ 0.00000000, -0.02639876,  0.01706616, -0.08229892,  0.12015337, -0.15289734, -0.08251060,
                    0.15658717, -0.04172155,  0.10556119],
                    [ 0.00000000, -0.02639606,  0.01708876, -0.08232962,  0.12015553, -0.15291165, -0.08253470,
                    0.15658627, -0.04173409,  0.10555199],
                    [ 0.00000000, -0.02639439,  0.01708424, -0.08231656,  0.12013876, -0.15291429, -0.08251298,
                    0.15660164, -0.04171884,  0.10554679],
                    [ 0.00000000, -0.02638364,  0.01708178, -0.08230615,  0.12013706, -0.15290739, -0.08251728,
                    0.15661213, -0.04172764,  0.10554501],
                    [ 0.00000000, -0.02639332,  0.01708583, -0.08231664,  0.12012445, -0.15291154, -0.08248951,
                    0.15662190, -0.04171957,  0.10554942],
                    [ 0.00000000,  0.08700177,  0.08366316,  0.07953200, -0.16923970,  0.06034253,  0.23461264,
                    0.23159909, -0.08662768, -0.01452735],
                    [ 0.00000000, -0.07244547,  0.02737196,  0.07369756,  0.11849800, -0.01468838, -0.08068197,
                    -0.22196104, -0.10857108,  0.00384381],
                    [ 0.00000000,  0.24396195,  0.07257971,  0.02183190,  0.05994213, -0.09902838, -0.05029112,
                    -0.09634171, -0.11046760,  0.02089387],
                    [ 0.00000000,  0.05029136, -0.03608268,  0.03495589,  0.02199023, -0.06974902,  0.11681847,
                    -0.12890734, -0.07075065,  0.12983331]]
                ],
            }
        )  # fmt: skip
        expectations_mean = Expectations({("cuda", None): 0.007999920286238194})

        inputs = self.processor.apply_chat_template(
            self.message1, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        torch.testing.assert_close(logits[:, :10, :10].cpu(), torch.tensor(expectations.get_expectation()))
        torch.testing.assert_close(logits.mean().cpu(), torch.tensor(expectations_mean.get_expectation()))

    def test_model_logits_batched(self):
        expectations = Expectations(
            {
                ("cuda", None): [
                    [[ 0.00000000, -0.02641735,  0.01707790, -0.08232785,  0.12016401, -0.15289739, -0.08252840,
                    0.15658122, -0.04173165,  0.10555175],
                    [ 0.00000000, -0.02639874,  0.01706615, -0.08229891,  0.12015339, -0.15289740, -0.08251059,
                    0.15658718, -0.04172156,  0.10556121],
                    [ 0.00000000, -0.02639605,  0.01708876, -0.08232963,  0.12015552, -0.15291169, -0.08253470,
                    0.15658626, -0.04173411,  0.10555198],
                    [ 0.00000000, -0.02639438,  0.01708425, -0.08231656,  0.12013876, -0.15291435, -0.08251297,
                    0.15660165, -0.04171887,  0.10554679],
                    [ 0.00000000, -0.02638364,  0.01708182, -0.08230612,  0.12013703, -0.15290746, -0.08251727,
                    0.15661217, -0.04172765,  0.10554502],
                    [ 0.00000000, -0.02639329,  0.01708584, -0.08231664,  0.12012445, -0.15291159, -0.08248951,
                    0.15662192, -0.04171958,  0.10554940],
                    [ 0.00000000,  0.08700177,  0.08366317,  0.07953200, -0.16923971,  0.06034254,  0.23461263,
                    0.23159909, -0.08662771, -0.01452734],
                    [ 0.00000000, -0.07244548,  0.02737197,  0.07369757,  0.11849802, -0.01468838, -0.08068197,
                    -0.22196105, -0.10857107,  0.00384379],
                    [ 0.00000000,  0.24396195,  0.07257971,  0.02183192,  0.05994214, -0.09902842, -0.05029113,
                    -0.09634172, -0.11046763,  0.02089386],
                    [ 0.00000000,  0.05029136, -0.03608267,  0.03495590,  0.02199023, -0.06974902,  0.11681848,
                    -0.12890735, -0.07075066,  0.12983333]],

                    [[ 0.00000000, -0.02641735,  0.01707790, -0.08232785,  0.12016401, -0.15289739, -0.08252840,
                    0.15658122, -0.04173165,  0.10555175],
                    [ 0.00000000, -0.02639874,  0.01706615, -0.08229891,  0.12015339, -0.15289740, -0.08251059,
                    0.15658718, -0.04172156,  0.10556121],
                    [ 0.00000000, -0.02639605,  0.01708876, -0.08232963,  0.12015552, -0.15291169, -0.08253470,
                    0.15658626, -0.04173411,  0.10555198],
                    [ 0.00000000, -0.02639438,  0.01708425, -0.08231656,  0.12013876, -0.15291435, -0.08251297,
                    0.15660165, -0.04171887,  0.10554679],
                    [ 0.00000000, -0.02638364,  0.01708182, -0.08230612,  0.12013703, -0.15290746, -0.08251727,
                    0.15661217, -0.04172765,  0.10554502],
                    [ 0.00000000, -0.02639329,  0.01708584, -0.08231664,  0.12012445, -0.15291159, -0.08248951,
                    0.15662192, -0.04171958,  0.10554940],
                    [ 0.00000000, -0.06525577,  0.08833339,  0.19255474, -0.08825377, -0.13921326,  0.14602648,
                    0.04482564, -0.14889751,  0.11028164],
                    [ 0.00000000, -0.00419279,  0.03873007,  0.31887013, -0.01151188, -0.30564448,  0.16670589,
                    0.19029431, -0.13194472,  0.02135594],
                    [ 0.00000000, -0.05691863,  0.09113241,  0.07550406,  0.09380092, -0.17235518,  0.20939635,
                    0.04311826, -0.04715816, -0.00544562],
                    [ 0.00000000, -0.01738080,  0.05235744,  0.22640616, -0.09752068, -0.16835804,  0.23356910,
                    -0.02671638, -0.13405795, -0.09033854]]
                ],
            }
        )  # fmt: skip
        expectations_mean = Expectations({("cuda", None): 0.007677852641791105})

        inputs = self.processor.apply_chat_template(
            [self.message1, self.message2],
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        torch.testing.assert_close(logits[:, :10, :10].cpu(), torch.tensor(expectations.get_expectation()))
        torch.testing.assert_close(logits.mean().cpu(), torch.tensor(expectations_mean.get_expectation()))
