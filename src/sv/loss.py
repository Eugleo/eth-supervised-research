from functools import partial
from typing import List, Optional

import einops
import polars as pl
import torch as t
from nnsight import LanguageModel
from rich.progress import Progress
from torch.utils.data import DataLoader, Dataset

from sv.vectors import SteeringVector

cross_entropy = t.nn.CrossEntropyLoss(reduction="none")


def spliced_llm_loss(
    model: LanguageModel,
    batch: dict,
    vector: SteeringVector,
):
    assert isinstance(model.device, t.device)
    cross_entropy.to(model.device)

    tokens = batch["input_ids"]
    is_prompt = batch["prompt_mask"]
    with model.trace(tokens, scan=False, validate=False) as _:  # type: ignore
        h = model.transformer.h[vector.layer].output[0]
        h[is_prompt] += vector.vector
        logits = einops.rearrange(model.output.logits, "b s tok -> b tok s").save()

    per_token_losses = cross_entropy(logits.value[:, :, :-1], tokens[:, 1:])
    per_token_losses[is_prompt[:, :-1]] = t.nan
    losses = per_token_losses.nanmean(-1)

    return losses


def _spliced_llm_loss_on_batch(
    model,
    batch,
    multiplier: float = 0,
    sv: Optional[SteeringVector] = None,
    kind="turner",
):
    criterion = t.nn.CrossEntropyLoss(reduction="none").to(model.device)
    prompts, completions = batch["prompt"], batch["completion"]
    sequences = [f"{p} {s}" for p, s in zip(prompts, completions)]
    tokens = model.tokenizer(sequences, return_tensors="pt", padding=True).input_ids

    completion_tokens = model.tokenizer(completions).input_ids
    completion_lengths = t.tensor([len(s) for s in completion_tokens])[:, None]
    total_length = tokens.size(1)
    indices = t.arange(total_length)[None, :]
    is_prompt = indices < total_length - completion_lengths

    with t.no_grad(), model.trace(sequences, scan=False, validate=False) as _:
        if sv and multiplier != 0:
            model.transformer.h[sv.layer].output[0][is_prompt] += multiplier * sv.vector
        logits = einops.rearrange(model.output.logits, "b s tok -> b tok s").save()

    tokens = tokens.to(logits.value.device)
    per_token_losses = criterion(logits.value[:, :, :-1], tokens[:, 1:])
    per_token_losses[is_prompt[:, :-1]] = t.nan
    per_sequence_losses = per_token_losses.nanmean(-1).tolist()

    return [
        {
            "index": index,
            "origin": origin,
            "loss": loss,
            "layer": sv.layer if sv else None,
            "multiplier": multiplier,
        }
        for index, origin, loss in zip(
            batch["index"], batch["origin"], per_sequence_losses
        )
    ]


@t.inference_mode()
def compute_spliced_llm_loss(
    model: LanguageModel,
    dataset: Dataset,
    steering_vectors: List[SteeringVector],
    multipliers: List[float],
    batch_size: int,
):
    loader = DataLoader(dataset, batch_size=batch_size)

    results = []
    with Progress() as progress:
        task_n = len(loader) * (1 + len(steering_vectors) * len(multipliers))
        task_computing = progress.add_task("Computing loss...", total=task_n)

        for batch in loader:
            score = partial(_spliced_llm_loss_on_batch, model=model, batch=batch)
            progress.update(task_computing, advance=1)
            results += [
                b | {"layer": sv.layer} for b in score() for sv in steering_vectors
            ]
            for sv in steering_vectors:
                for coeff in (c for c in multipliers if c != 0):
                    progress.update(task_computing, advance=1)
                    results += score(multiplier=coeff, sv=sv)

    return pl.DataFrame(results)
