from __future__ import annotations
import torch as th
import numpy as np

from nnsight_utils import (
    get_layer_output,
    get_next_token_probs,
    collect_activations,
    get_num_layers,
)
import nnsight as nns
from nnterp import collect_activations_batched


def object_lens(
    nn_model,
    target_prompts,
    idx,
    source_prompts=None,
    hiddens=None,
    steering_vectors=None,
    num_patches=-1,
    scan=False,
    remote=False,
    layers=None,
    batch_size=16,
):
    if layers is None:
        layers = list(range(get_num_layers(nn_model)))
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]
    num_layers = get_num_layers(nn_model)
    if num_patches == -1:
        num_patches = num_layers
    if hiddens is None and source_prompts is None:
        raise ValueError("Either source_prompts or hiddens must be provided")
    if steering_vectors is not None:
        for i, (h, s) in enumerate(zip(hiddens, steering_vectors)):
            hiddens[i] = h + s
    with nn_model.session(remote=remote):
        if hiddens is None:
            hiddens = collect_activations_batched(
                nn_model,
                list(source_prompts.flatten()),
                batch_size=batch_size,
                layers=layers,
                use_session=False,
            )
            hiddens = hiddens.transpose(0, 1)  # (all_prompts, layer, hidden_size)
            hiddens = hiddens.reshape(
                batch_size, source_prompts.shape[1], get_num_layers(nn_model), -1
            ).mean(dim=1)
        probs_l = nns.list().save()
        for layer in range(num_layers):
            with nn_model.trace(target_prompts, remote=remote):
                for target_layer in range(layer, min(layer + num_patches, num_layers)):
                    get_layer_output(nn_model, target_layer)[:, idx] = hiddens[
                        target_layer
                    ]
                probs = get_next_token_probs(nn_model).cpu().save()
                probs_l.append(probs)
    return (
        th.cat([p for p in probs_l], dim=0)
        .reshape(num_layers, len(target_prompts), -1)
        .transpose(0, 1)
    )
