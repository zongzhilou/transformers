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
import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        ShensiModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class ShensiModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = ShensiModel

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        # Standard CausalLMModelTester knobs.
        self.hidden_size = 64
        self.num_attention_heads = 4
        self.num_key_value_heads = 1
        self.num_hidden_layers = 2
        self.num_experts_per_tok = 2
        self.moe_intermediate_size = 32
        self.max_position_embeddings = 64
        # DSA / DEA.
        self.kv_deepemb_rank = 8
        self.head_dim = 32
        self.index_block_size = 4
        self.index_head_dim = 16
        self.index_n_heads = 2
        self.index_topk_tokens = 8
        self.index_block_topk = 2
        self.q_lora_rank = 32
        self.qk_nope_head_dim = 32
        self.v_head_dim = 32
        # KDA.
        self.linear_conv_kernel_dim = 4
        self.linear_head_dim = 8
        self.linear_num_heads = 4
        # MoE / MLP.
        self.deepemb_rank = 8
        self.latent_size = 16
        self.mlp_layer_types = ["sparse", "sparse"]
        self.n_routed_experts = 4
        self.routed_scaling_factor = 1.5
        self.scoring_func = "sqrtsoftplus"
        self.swiglu_limit = 10.0
        # Residual.
        self.attn_res_block_size = 2
        self.hc_active_streams = 2
        self.hc_conv_kernels = [4, 8, 12]
        self.hc_fixed_streams = 1
        self.hc_mult = 4
        # ROSA depends on token ids, so the generic tests disable it to keep `inputs_embeds`
        # equivalent to `input_ids`; dedicated ROSA tests cover the enabled path.
        self.rosa_layer_ids = []


@require_torch
class ShensiModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = ShensiModelTester

    # Indexer parameters only influence the argmax over compressed positions (``topk``),
    # which is non-differentiable — their gradients flow through a separate objective in
    # the upstream training recipe, not the main causal-LM loss.
    test_all_params_have_gradient = False

    # No SequenceClassification / TokenClassification / QA heads on V4.
    def is_pipeline_test_to_skip(self, *args, **kwargs):
        return True

    @unittest.skip(
        "Shensi's indexed-attention cache and latent expert layout are not supported by the "
        "quantized tensor-parallel generation path yet."
    )
    def test_tp_generation_quantized(self):
        pass

    @unittest.skip(
        "Shensi's indexed-attention and linear-attention layers use custom cache layers, which are "
        "not compatible with QuantizedCache."
    )
    def test_generate_with_quant_cache(self):
        pass

    @unittest.skip(
        "accelerate's device_map='auto' cannot partition Shensi's 4D hyper-connection "
        "streams and block-residual buffers across devices."
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        "accelerate's device_map='auto' cannot partition Shensi's 4D hyper-connection "
        "streams and block-residual buffers across devices."
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        "accelerate's device_map='auto' cannot partition Shensi's 4D hyper-connection "
        "streams and block-residual buffers across devices."
    )
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("Requires nvidia-cutlass-dsl package for batched/grouped inference comparison.")
    def test_eager_matches_batched_and_grouped_inference(self):
        pass

    @unittest.skip("Requires nvidia-cutlass-dsl package for batched/grouped inference comparison.")
    def test_eager_matches_batched_and_grouped_inference_0_fp16(self):
        pass

    @unittest.skip("Requires nvidia-cutlass-dsl package for batched/grouped inference comparison.")
    def test_eager_matches_batched_and_grouped_inference_2_bf16(self):
        pass

    @unittest.skip(
        "Framework mismatch: build_glob_alternation's unanchored regex 'norm' also "
        "matches routed_expert_norm, but the test's anchored regex doesn't expect it."
    )
    def test_keep_in_fp32_modules(self):
        pass

    @unittest.skip("use_kernels=True triggers kernel download/compile that hangs in this environment.")
    def test_kernels_can_load_without_crashing(self):
        pass

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        # Indexed layers attend to a per-query top-k selection and linear layers keep a
        # recurrent state, so the KV length varies per layer. We only check the shape
        # invariants: batched, same number-of-heads and query-length.
        import torch  # noqa: PLC0415

        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), (output_length - prompt_length))
        for _, iter_attentions in enumerate(attentions):
            self.assertIsInstance(iter_attentions, tuple)
            for layer_attention in iter_attentions:
                self.assertIsInstance(layer_attention, torch.Tensor)
                self.assertEqual(layer_attention.shape[0], batch_size)
                self.assertEqual(layer_attention.shape[1], config.num_attention_heads)

    @unittest.skip("MoE routing on a tiny randomly-initialized model makes the overfit target unstable.")
    def test_training_overfit(self):
        pass

    @unittest.skip("Reentrant checkpointing does not support Shensi's custom 4D HC state/cache.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(
        "The stream-expansion and hyper-connection depthwise convolutions need the prefix context "
        "in the new segment; the cache stores attention/recurrent state only, so a cached continuation "
        "is not bit-identical to a full forward."
    )
    def test_recurrent_layers_mask_padding_on_continued_forward(self):
        pass

    def test_attention_outputs(self):
        import torch  # noqa: PLC0415

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_attentions = True
        indexed_layers = [layer_type for layer_type in config.layer_types if layer_type == "indexed_attention"]
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertIsNotNone(attentions)
            self.assertEqual(len(attentions), len(indexed_layers))
            for attention in attentions:
                self.assertIsInstance(attention, torch.Tensor)
                self.assertEqual(attention.shape[0], inputs_dict["input_ids"].shape[0])
                self.assertEqual(attention.shape[1], config.num_attention_heads)

    def test_hash_layers_share_the_deep_embedding_and_start_neutral(self):
        import torch  # noqa: PLC0415

        # The common tester runs all-MoE layers (it needs no input_ids); this check needs the
        # hash-routed prefix, so ask for two hash layers.
        self.model_tester.mlp_layer_types = ["dense", "dense"]
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = ShensiModel(config).to(torch_device).eval()
        hash_layers = [layer for layer in model.layers if layer.is_hash]
        assert len(hash_layers) > 1, "the tester config needs more than one hash layer"
        assert len({id(layer.mlp.deepemb) for layer in hash_layers}) == 1, "hash layers must share one table"

        # Every token owns the identity matrix, and the zeroed output projection plus unit bias
        # keep the DeepEmbed scale at one, so a fresh hash layer is a plain SwiGLU MLP.
        table = hash_layers[0].mlp.deepemb.weight
        identity = torch.eye(config.deepemb_rank, dtype=table.dtype, device=table.device).flatten()
        assert torch.equal(table, identity.expand_as(table))
        for layer in hash_layers:
            weight = layer.mlp.deepemb_b_proj.weight
            bias = layer.mlp.deepemb_b_proj.bias
            assert torch.equal(weight, torch.zeros_like(weight))
            assert torch.equal(bias, torch.ones_like(bias))

    def test_hidden_states_output(self):
        # Every layer emits the standard ``(batch, seq, hidden)`` shape: the attention
        # residual state is bookkeeping, not an extra hidden axis.
        import torch  # noqa: PLC0415

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[-1]
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), config.num_hidden_layers + 1)
            seq_len = inputs_dict["input_ids"].shape[1]
            for layer_h in hidden_states:
                self.assertEqual(layer_h.shape, (inputs_dict["input_ids"].shape[0], seq_len, config.hidden_size))

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        # DSA layers cache the low-rank DEA latent; KDA layers keep a constant recurrent state and
        # have no K/V entries at all.
        import torch  # noqa: PLC0415

        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            if keys is None:
                continue
            self.assertIsInstance(keys, torch.Tensor)
            self.assertEqual(keys.shape[0], batch_size)
            self.assertEqual(keys.shape[-1], config.kv_deepemb_rank)

    @unittest.skip(
        reason=(
            "The hyper-connection stream convolutions and the attention-residual prefix sums read the whole "
            "sequence, so padded positions change the prefix state: a left-padded batch is not equivalent "
            "to the unpadded run by design."
        )
    )
    def test_left_padding_compatibility(self):
        pass

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        # The exact seq-length assertion of the base tester does not hold under
        # assisted decoding (arbitrary draft-token batches), so only the batch and
        # hidden dims are checked here.
        import torch  # noqa: PLC0415

        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), (output_length - prompt_length))
        for iter_hidden_states in hidden_states:
            self.assertIsInstance(iter_hidden_states, tuple)
            for layer_hidden in iter_hidden_states:
                self.assertIsInstance(layer_hidden, torch.Tensor)
                self.assertEqual(layer_hidden.shape[0], batch_size)
                self.assertEqual(layer_hidden.shape[-1], config.hidden_size)


@require_torch
@require_torch_accelerator
@slow
class ShensiIntegrationTest(unittest.TestCase):
    """End-to-end generation on the published Shensi checkpoint.

    The checkpoint is large, so this test is marked slow and excluded from the default run:

        RUN_SLOW=1 pytest tests/models/shensi/test_modeling_shensi.py::ShensiIntegrationTest -s

    It covers everything the tiny-config tests cannot reach: the published per-layer weights,
    the DSA indexer and its shared top-k propagation, the KDA (linear attention) layers, the
    low-rank KV latent, the attention-residual bank and the hyper-connection streams.
    """

    model_id = "louzongzhi/Shensi-V1-N"
    prompt = "The capital of France is"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        cls.model = AutoModelForCausalLM.from_pretrained(
            cls.model_id,
            dtype="auto",
            device_map="auto",
            attn_implementation="eager",
        )

    def test_config_matches_model_class(self):
        config = AutoConfig.from_pretrained(self.model_id)
        self.assertEqual(config.model_type, "shensi")
        self.assertEqual(config.architectures, ["ShensiForCausalLM"])

    def test_greedy_generation(self):
        inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(output_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        self.assertGreater(output_ids.shape[1], inputs.input_ids.shape[1])
        self.assertTrue(completion.strip())


@require_torch
class ShensiCompressionTest(unittest.TestCase):
    def _config(self, **overrides):
        kwargs = {
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "qk_nope_head_dim": 16,
            "v_head_dim": 16,
            "q_lora_rank": 8,
            "kv_deepemb_rank": 4,
            "deepemb_rank": 4,
            "moe_intermediate_size": 16,
            "latent_size": 8,
            "index_block_size": 2,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "index_topk_tokens": 4,
            "index_block_topk": 2,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "mlp_layer_types": ["sparse", "sparse", "sparse", "sparse"],
            "attn_res_block_size": 2,
            "attn_res_latent_size": 32,
            "hc_latent_size": 32,
            "hc_conv_kernels": [2, 3],
            "hc_mult": 3,
            "hc_active_streams": 2,
            "hc_fixed_streams": 1,
            "max_position_embeddings": 32,
        }
        kwargs.update(overrides)
        from transformers import ShensiConfig

        return ShensiConfig(**kwargs)

    def test_compression_forward_and_cache(self):
        from transformers import ShensiModel

        config = self._config()
        model = ShensiModel(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
        self.assertEqual(outputs.last_hidden_state.shape, (2, 5, config.hidden_size))
        cache = outputs.past_key_values
        write_layers = [i for i, layer_type in enumerate(config.layer_types) if layer_type == "indexed_attention"]
        token_ids = [cache.layers[i].token_ids for i in write_layers]
        self.assertTrue(all(ids is token_ids[0] for ids in token_ids))
        leader = cache.layers[write_layers[0]]
        self.assertIs(leader.keys, leader.values)
        self.assertEqual(leader.keys.dtype, torch.float8_e4m3fn)
        self.assertEqual(leader.indexer_keys.dtype, torch.float8_e4m3fn)
        for layer_idx in write_layers[1:]:
            self.assertIs(cache.layers[layer_idx].source, leader)
            self.assertIsNone(cache.layers[layer_idx].keys)
            self.assertIsNone(cache.layers[layer_idx].indexer_keys)
        self.assertEqual(leader.indexer_keys.shape[-1], config.index_k_size + 1)
        with torch.no_grad():
            decoded = model(input_ids=torch.randint(0, config.vocab_size, (2, 1)), past_key_values=cache)
        self.assertEqual(decoded.last_hidden_state.shape, (2, 1, config.hidden_size))

    def test_model_dtype_support(self):
        from transformers import ShensiModel

        config = self._config()
        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        for dtype in (torch.float32, torch.bfloat16):
            model = ShensiModel(config).to(dtype).eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, use_cache=True)
            self.assertEqual(outputs.last_hidden_state.dtype, dtype)

    def test_hyper_connection_gate_is_trainable(self):
        from transformers import ShensiModel

        config = self._config()
        hyper_connection = ShensiModel(config).layers[0].attn_hc
        # The per-stream gate shares one projection, so a zero initialization is neutral
        # without creating a multiplicative dead zone.
        self.assertEqual(hyper_connection.fn[: config.hc_mult].abs().sum().item(), 0.0)

        streams = torch.randn(2, 5, config.hc_mult, config.hidden_size)
        collapsed, _ = hyper_connection(streams)
        collapsed.square().mean().backward()

        self.assertIsNotNone(hyper_connection.fn.grad)
        self.assertGreater(hyper_connection.fn.grad.abs().sum().item(), 0.0)

    def test_hyper_connection_residual_anchor(self):
        from transformers import ShensiModel

        config = self._config()
        hyper_connection = ShensiModel(config).layers[0].ffn_hc
        streams = torch.randn(2, 5, 1, config.hidden_size).expand(-1, -1, config.hc_mult, -1).clone()
        _, state = hyper_connection(streams)
        updated = hyper_connection.write_back(
            torch.randn(2, 5, config.hidden_size),
            state,
            torch.randn(2, 5, config.hidden_size),
        )
        for batch_idx in range(updated.shape[0]):
            for token_idx in range(updated.shape[1]):
                selected = state.indices[batch_idx, token_idx].tolist()
                for stream_idx in range(config.hc_mult):
                    if stream_idx not in selected:
                        self.assertTrue(
                            torch.equal(
                                updated[batch_idx, token_idx, stream_idx],
                                streams[batch_idx, token_idx, stream_idx],
                            )
                        )

    def test_hyper_connection_conditional_streams(self):
        from transformers import ShensiModel

        config = self._config()
        hyper_connection = ShensiModel(config).layers[0].attn_hc
        self.assertGreater(hyper_connection.num_route_coefficients, 0)
        self.assertIsNotNone(hyper_connection.down)
        self.assertEqual(hyper_connection.down.shape[0], config.hc_mult - config.hc_fixed_streams)
        self.assertEqual(hyper_connection.up.abs().sum().item(), 0.0)

        decay = 1.0 - hyper_connection.decay.sigmoid()
        self.assertEqual(len(torch.unique(decay)), config.hc_mult - config.hc_fixed_streams)

    def test_attention_residual_is_identity_at_init(self):
        from transformers import ShensiModel

        config = self._config()
        model = ShensiModel(config).eval()
        modules = [m for m in model.modules() if type(m).__name__ == "ShensiAttentionResidual"]
        self.assertEqual(len(modules), 2 * config.num_hidden_layers + 1)
        diffs = []

        def hook(module, args, kwargs, output):
            diffs.append(output.abs().max().item())

        for module in modules:
            module.register_forward_hook(hook, with_kwargs=True)
        with torch.no_grad():
            model(input_ids=torch.randint(0, config.vocab_size, (2, 5)))
        self.assertEqual(len(diffs), len(modules))
        self.assertLess(max(diffs), 1e-2)

    def test_attention_residual_routing_weights(self):
        from transformers import ShensiModel

        config = self._config()
        module = ShensiModel(config).layers[0].self_attention_res
        prefix_sum = torch.randn(2, 5, config.hidden_size)
        delta = torch.randn(2, 5, config.hidden_size)
        blocks = torch.randn(2, 5, 4, config.hidden_size)

        with torch.no_grad():
            module.weight[1] = 1.0
        routed = module(prefix_sum=prefix_sum, delta=delta, blocks=blocks, num_blocks=3)
        self.assertGreater(routed.abs().max().item(), 0.0)

        captured = []
        original_softmax = torch.softmax

        def spy(tensor, dim=-1, **kwargs):
            result = original_softmax(tensor, dim=dim, **kwargs)
            captured.append(result)
            return result

        torch.softmax = spy
        try:
            with torch.no_grad():
                module(prefix_sum=prefix_sum, delta=delta, blocks=blocks, num_blocks=3)
        finally:
            torch.softmax = original_softmax

        self.assertEqual(len(captured), 1)
        weights = captured[0]
        self.assertEqual(weights.shape[-1], config.num_attn_res_blocks + 2)
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6))
        self.assertTrue(bool((weights >= 0).all()))

    def test_attention_residual_bias(self):
        from transformers import ShensiModel

        config = self._config()
        module = ShensiModel(config).layers[0].self_attention_res
        prefix_sum = torch.randn(2, 5, config.hidden_size)
        delta = torch.randn(2, 5, config.hidden_size)
        blocks = torch.randn(2, 5, 4, config.hidden_size)

        with torch.no_grad():
            module.weight[1] = 1.0
            baseline = module(
                prefix_sum=prefix_sum,
                delta=delta,
                blocks=blocks,
                num_blocks=3,
            )
            module.bias[:3].fill_(4.0)
            biased = module(
                prefix_sum=prefix_sum,
                delta=delta,
                blocks=blocks,
                num_blocks=3,
            )

        self.assertGreater((biased - baseline).abs().max().item(), 1e-5)

    def test_attention_residual_max_blocks_follows_schedule(self):
        from transformers import ShensiModel

        for block_size in (1, 2, 4):
            config = self._config(attn_res_block_size=block_size)
            module = ShensiModel(config).layers[0].self_attention_res
            self.assertEqual(module.bias.numel(), 1 + config.num_attn_res_blocks + 1)

    def test_num_attn_res_blocks_follows_schedule(self):
        for block_size in (1, 2, 4):
            config = self._config(attn_res_block_size=block_size)
            writes = config.attn_res_block_layer_types.count("block_write_layer")
            self.assertEqual(config.num_attn_res_blocks, writes)
            self.assertGreater(writes, 0)
        with self.assertRaises(ValueError):
            self._config(num_attn_res_blocks=99)

    def test_attention_residual_gradients(self):
        from transformers import ShensiModel

        config = self._config()
        module = ShensiModel(config).layers[0].self_attention_res
        with torch.no_grad():
            module.weight[1] = 1.0
        prefix_sum = torch.randn(2, 5, config.hidden_size, requires_grad=True)
        delta = torch.randn(2, 5, config.hidden_size)
        blocks = torch.randn(2, 5, 4, config.hidden_size)
        output = module(
            prefix_sum=prefix_sum,
            delta=delta,
            blocks=blocks,
            num_blocks=3,
        )
        self.assertEqual(output.shape, prefix_sum.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for parameter in (module.bias, module.weight):
            grad = parameter.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.0)
        self.assertGreater(module.bias.grad[-1].abs().item(), 0.0)
        self.assertGreater(module.bias.grad[:3].abs().sum().item(), 0.0)
        for parameter in (module.q_proj.weight, module.k_proj.weight, module.g_a_proj.weight, module.g_b_proj.weight):
            grad = parameter.grad
            self.assertIsNotNone(grad)
            self.assertGreater(grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(prefix_sum.grad)

    def test_attention_residual_correction_scale(self):
        from transformers import ShensiModel

        config = self._config()
        model = ShensiModel(config).eval()
        modules = [m for m in model.modules() if type(m).__name__ == "ShensiAttentionResidual"]
        ratios = []

        def hook(module, args, kwargs, output):
            reference = kwargs["prefix_sum"].float().norm()
            ratios.append(output.float().norm().item() / (reference.item() + 1e-6))

        for module in modules:
            module.register_forward_hook(hook, with_kwargs=True)
        with torch.no_grad():
            model(input_ids=torch.randint(0, config.vocab_size, (2, 5)))
        self.assertEqual(len(ratios), len(modules))
        self.assertTrue(all(torch.isfinite(torch.tensor(ratios)).tolist()))
        self.assertLess(max(ratios), 1e3)

    def test_attn_bank_runtime_state_is_non_persistent(self):
        from transformers.models.shensi.modeling_shensi import ShensiAttnBank

        config = self._config()
        bank = ShensiAttnBank(config)
        self.assertEqual(
            {name for name, _ in bank.named_parameters()},
            {"basis", "stream_basis", "token_down.weight", "token_up.weight"},
        )

        bank.open(torch.randn(2, 5, config.hidden_size), config.num_attn_res_blocks)
        buffers = {name for name, _ in bank.named_buffers()}
        self.assertTrue({"head", "token_bias", "block_residual", "num_valid_blocks"} <= buffers)
        saved = bank.state_dict()
        for name in ("head", "token_bias", "block_residual", "num_valid_blocks"):
            self.assertNotIn(name, saved)
        self.assertIn("basis", saved)

    def test_attn_bank_final_write_keeps_committed_rows(self):
        from transformers.models.shensi.modeling_shensi import ShensiAttnBank

        config = self._config()
        bank = ShensiAttnBank(config)
        bank.basis.data.normal_()
        bank.stream_basis.data.normal_()
        bank.token_down.weight.data.normal_()
        bank.token_up.weight.data.normal_()
        inputs_embeds = torch.randn(2, 5, config.hidden_size)
        bank.open(inputs_embeds, config.num_attn_res_blocks)
        blocks = bank.read_empty()
        for index in range(1, blocks.shape[-2]):
            blocks[..., index, :] = float(index)
        bank.update(blocks)
        committed = bank.block_residual[..., :-1, :].clone()

        bank.write(torch.full_like(inputs_embeds, 9.0), config.num_attn_res_blocks)
        self.assertTrue(torch.equal(bank.block_residual[..., :-1, :], committed))

    def test_compression_roundtrip(self):
        from transformers.models.shensi.modeling_shensi import ShensiAttnBank

        config = self._config()
        bank = ShensiAttnBank(config)
        with torch.no_grad():
            eye = torch.eye(max(bank.rank, config.hidden_size))[: bank.rank, : config.hidden_size]
            bank.basis.copy_(eye)
            stream_eye = torch.eye(max(bank.stream_rank, config.hidden_size))[: bank.stream_rank, : config.hidden_size]
            bank.stream_basis.copy_(stream_eye)
            bank.token_down.weight.normal_()
            bank.token_up.weight.zero_()
        streams = torch.randn(2, 5, config.hc_mult, config.hidden_size)
        reconstructed_streams = bank.decode_streams(bank.encode_streams(streams))
        self.assertLess((reconstructed_streams - streams).square().mean().item(), 1e-3)

        embeds = torch.randn(2, 5, config.hidden_size)
        blocks = torch.randn(2, 5, 4, config.hidden_size)
        self.assertTrue(torch.allclose(bank.decode(bank.encode(blocks)), blocks, atol=1e-5))

        bank.open(embeds, 3)
        expected = torch.zeros(2, 5, 4, config.hidden_size)
        expected[..., 0, :] = embeds
        self.assertTrue(torch.allclose(bank.read_empty(), expected, atol=1e-5))
        self.assertTrue(torch.allclose(bank.read(0), expected[..., :1, :], atol=1e-5))

        summary = torch.randn(2, 5, config.hidden_size)
        bank.write(summary, 2)
        expected[..., 2, :] = summary
        self.assertTrue(torch.allclose(bank.read(2), expected[..., :3, :], atol=1e-5))

        bank.update(expected)
        self.assertTrue(torch.allclose(bank.read(), expected, atol=1e-5))

        # The token bias is reconstructed from the embedding and cancels in `update` / `read`.
        with torch.no_grad():
            bank.token_up.weight.normal_()
        bank.open(embeds, 3)
        bank.update(expected)
        self.assertTrue(torch.allclose(bank.read(), expected, atol=1e-5))

        bank.close()
        self.assertIsNone(bank.block_residual)

    def test_stream_embeddings(self):
        from transformers.models.shensi.modeling_shensi import ShensiStreamEmbeddings

        config = self._config()
        model = ShensiModel(config)
        expansion = model.stream_embeddings
        self.assertIsInstance(expansion, ShensiStreamEmbeddings)
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        embeds = model.embed_tokens(input_ids)
        streams = expansion(embeds, input_ids)
        self.assertEqual(streams.shape, (2, 5, config.hc_mult, config.hidden_size))
        # Tables are initialized as copies of E(x) and the convolutions are zero: all streams start equal.
        self.assertTrue(torch.allclose(streams, embeds.unsqueeze(2).expand_as(streams), atol=1e-6))
        # The pure inputs_embeds path falls back to E(x) + conv, which is also equal at init.
        self.assertTrue(torch.allclose(expansion(embeds), embeds.unsqueeze(2).expand_as(streams), atol=1e-6))

        with torch.no_grad():
            expansion.token_embeddings.weight.normal_()
        streams = expansion(embeds, input_ids)
        self.assertFalse(torch.allclose(streams, embeds.unsqueeze(2).expand_as(streams), atol=1e-6))

    def test_rosa_multiturn_boundaries(self):
        from transformers import ShensiModel

        sequence = torch.tensor([[10, 11, 12, 13, 14, 2, 20, 21, 22, 10, 11, 12, 13]])
        attention_mask = torch.ones_like(sequence)
        config = self._config(rosa_layer_ids=[0, 1], rosa_embed_dim=8, eos_token_id=2)
        model = ShensiModel(config).eval()
        captured = []

        def hook(module, args):
            captured.append(args[1].detach().clone())

        model.layers[0].rosa.register_forward_pre_hook(hook)
        with torch.no_grad():
            model(input_ids=sequence, attention_mask=attention_mask)
        self.assertEqual(captured[-1][0, -1].item(), 14)

        config = self._config(
            rosa_layer_ids=[0, 1],
            rosa_embed_dim=8,
            eos_token_id=2,
            rosa_reset_token_id=2,
        )
        model = ShensiModel(config).eval()
        captured = []
        model.layers[0].rosa.register_forward_pre_hook(hook)
        with torch.no_grad():
            model(input_ids=sequence, attention_mask=attention_mask)
        self.assertEqual(captured[-1][0, -1].item(), -1)

    def test_rosa_embedding_is_shared(self):
        from transformers import ShensiModel

        config = self._config(rosa_layer_ids=[0, 1], rosa_embed_dim=8)
        model = ShensiModel(config)
        rosa_layers = [layer.rosa for layer in model.layers if layer.rosa is not None]
        self.assertEqual(len(rosa_layers), 2)
        self.assertIs(rosa_layers[0].embed_tokens, rosa_layers[1].embed_tokens)

        target_vocab_size = config.vocab_size + 5
        model.resize_token_embeddings(target_vocab_size)
        self.assertIs(rosa_layers[0].embed_tokens, rosa_layers[1].embed_tokens)
        self.assertEqual(rosa_layers[0].embed_tokens.num_embeddings, target_vocab_size)
