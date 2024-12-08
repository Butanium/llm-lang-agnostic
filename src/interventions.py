from __future__ import annotations
from dataclasses import dataclass
import torch as th
from torch.utils.data import DataLoader
from warnings import warn
from nnsight_utils import (
    get_layer_output,
    get_layer_input,
    get_attention,
    get_attention_output,
    get_next_token_probs,
    collect_activations,
    collect_activations_batched,
    get_num_layers,
    NNLanguageModel,
    GetModuleOutput,
    project_on_vocab,
)
from typing import Optional

__all__ = [
    "logit_lens",
    "TargetPrompt",
    "repeat_prompt",
    "TargetPromptBatch",
    "patchscope_lens",
    "patchscope_generate",
    "steer",
    "skip_layers",
    "patch_attention_lens",
    "patch_object_attn_lens",
    "object_lens",
]
from nnterp.interventions import (
    logit_lens,
    TargetPrompt,
    repeat_prompt,
    patchscope_lens,
    patchscope_generate,
    steer,
    skip_layers,
    patch_object_attn_lens,
)


def object_lens(
    nn_model,
    target_prompts,
    idx,
    source_prompts=None,
    hiddens=None,
    steering_vectors=None,
    num_patches=-1,
    scan=True,
    remote=False,
):
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]
    num_layers = get_num_layers(nn_model)
    if num_patches == -1:
        num_patches = num_layers
    if hiddens is None:
        if source_prompts is None:
            raise ValueError("Either source_prompts or hiddens must be provided")
        hiddens = collect_activations(
            nn_model,
            source_prompts,
            remote=remote,
        )
    if steering_vectors is not None:
        for i, (h, s) in enumerate(zip(hiddens, steering_vectors)):
            hiddens[i] = h + s
    probs_l = []
    for layer in range(num_layers):
        with nn_model.trace(target_prompts, scan=layer == 0 and scan, remote=remote):
            for target_layer in range(layer, min(layer + num_patches, num_layers)):
                get_layer_output(nn_model, target_layer)[:, idx] = hiddens[target_layer]
            probs = get_next_token_probs(nn_model).cpu().save()
            probs_l.append(probs)
    return (
        th.cat([p.value for p in probs_l], dim=0)
        .reshape(num_layers, len(target_prompts), -1)
        .transpose(0, 1)
    )
