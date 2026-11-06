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

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-06.*


# Shensi

[Shensi](https://huggingface.co/louzongzhi/Shensi-V1-N) is a hybrid sparse-attention Mixture-of-Experts language
model. Each decoder layer runs either **indexed attention** (DeepSeek-style sparse attention with a hierarchical
indexer) or **gated linear attention** (Kimi Delta Attention). Feed-forward layers are a dense ReLU² MLP with a
token-conditioned DeepEmbed scale or a latent MoE. Two multi-depth mechanisms carry information across the stack: a
set of **hyper-connection streams** and a low-rank **attention-residual bank**. Selected layers also inject **ROSA**,
a parameterless suffix automaton that predicts the next token from the longest matching suffix seen so far.

## Architecture

### Hybrid attention

The per-layer attention plan is derived from the attention-residual write schedule: write layers run indexed
attention, read layers run gated linear attention. Indexed layers use a single low-rank KV latent and a hierarchical
indexer that ranks mean-pooled key blocks before refining the tokens inside the retained blocks. The top-k selection
is cached per query and shared with the following indexed layers when `indexer_types` marks them as `"shared"`.
Linear-attention layers keep a constant-size recurrent state instead of a growing KV cache.

### DeepEmbed key/value latents

Instead of materializing per-head keys and values, the indexed layers cache one low-rank latent per token. A token
embedding is added to the latent on the first indexed layer only, then the shared up-projection reconstructs the
full-width keys and values. During inference the latent is stored in FP8, which keeps the KV cache small at long
context.

### Hyper-connection streams

Residual connections are replaced by `hc_mult` parallel streams. Stream 0 is the shared token embedding and every
auxiliary stream owns its own embedding table plus a causal depthwise convolution for local context. Each sublayer
collapses the streams with a softmax `pre` projection, routes the active streams with learned coefficients and decay,
and writes the sublayer output back with a per-stream `post` projection and a low-rank adapter. Streams are compressed
to a low-rank basis between layers and expanded again by the hyper-connection head.

### Attention-residual blocks

Layers are grouped into attention-residual blocks. A write layer commits the block's accumulated sublayer output to a low-rank
bank and the following read layers route over the committed block summaries. Every sublayer applies a gated delta
rule with channel-wise decay, erase and write gates; the gates read the incoming update, which keeps a write from
gating itself.

Depth routing is a softmax over the committed block deltas, the current partial block and a learned null source, with
a learned per-source bias that lets the model keep or suppress each block independently. The routed branch is blended
with the plain accumulation through a ReZero scalar and rescaled to the plain RMS, so a fresh module is an exact
identity and the depth correction is added to the hyper-connection streams without changing their scale.

### ROSA

ROSA is a parameterless suffix automaton over the token ids: for each position it returns the token that followed the
longest matching suffix in the earlier context, or `-1` when the suffix is new. Its learned embedding and stream gate
are trained by the language-modeling loss and injected at selected depths. The automaton state is part of the
generation cache and supports cropping, beam reordering and batched continuation.

### Mixture-of-Experts

The leading layers are dense ReLU² MLPs with a token-conditioned DeepEmbed scale; the remaining layers are a latent
MoE that routes in a low-rank space, normalizes, and projects back. Router and expert pools are shared per
attention-residual block. Set `output_router_logits=True` to return the router logits and enable the load-balancing
auxiliary loss, and `erc_loss_coef` to weight the expert-router coupling regularizer used during training.

## Usage example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "louzongzhi/Shensi-V1-N"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

> [!TIP]
> Indexed attention builds a per-query sparse mask, so only the eager and SDPA backends are supported. Set
> `attn_implementation="eager"` when loading a checkpoint without SDPA support.

## ShensiConfig

[[autodoc]] ShensiConfig

## ShensiPreTrainedModel

[[autodoc]] ShensiPreTrainedModel
    - forward

## ShensiModel

[[autodoc]] ShensiModel
    - forward

## ShensiForCausalLM

[[autodoc]] ShensiForCausalLM
    - forward
