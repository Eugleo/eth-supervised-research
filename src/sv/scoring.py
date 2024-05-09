import re
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


def activations(
    model: LanguageModel,
    batch: dict,
    layer: int,
    vector: Optional[SteeringVector] = None,
):
    assert isinstance(model.device, t.device)

    tokens = batch["input_ids"].to(model.device)
    is_prompt = batch["prompt_mask"].to(model.device)
    with model.trace(tokens, scan=False, validate=False) as _:  # type: ignore
        if vector is not None:
            h = model.transformer.h[vector.layer].output[0]
            h[is_prompt] += vector.vector
        activations = model.transformer.h[layer].output[0].t[-1].save()

    return activations.value


def has_keyword(text, keywords):
    if any(k in text for k in keywords):
        return any(re.findall(rf"\b{k}'?s?\b", text) for k in keywords)
    return False


def p_successful_rollout(
    model, prompt, sv, keywords, multipliers=[1], length=40, batch_size=200
):
    rollouts = {}
    batch = [prompt] * batch_size

    with t.no_grad(), model.generate(
        batch,
        max_new_tokens=length,
        scan=False,
        validate=False,
        pad_token_id=model.tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.8,
        temperature=1,
    ) as _:
        baseline_output = model.generator.output.save()
    baseline_decoded = model.tokenizer.batch_decode(
        baseline_output, skip_special_tokens=True
    )
    baseline_p_success = (
        sum(any(k in d for k in keywords) for d in baseline_decoded) / batch_size
    )
    rollouts[0] = baseline_p_success

    for multiplier in multipliers:
        with t.no_grad(), model.generate(
            batch,
            max_new_tokens=length,
            scan=False,
            validate=False,
            pad_token_id=model.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.8,
            temperature=1,
        ) as _:
            model.transformer.h[sv.layer].output[0][:] += multiplier * sv.vector
            steered_output = model.generator.output.save()
        decoded = model.tokenizer.batch_decode(steered_output, skip_special_tokens=True)
        rollouts[multiplier] = (
            sum(has_keyword(d, keywords) for d in decoded) / batch_size
        )

    return rollouts


def llm_loss(
    model: LanguageModel,
    batch: dict,
    vector: Optional[SteeringVector] = None,
):
    assert isinstance(model.device, t.device)
    cross_entropy.to(model.device)

    tokens = batch["input_ids"].to(model.device)
    is_prompt = batch["prompt_mask"].to(model.device)
    with model.trace(tokens, scan=False, validate=False) as _:  # type: ignore
        if vector is not None:
            h = model.transformer.h[vector.layer].output[0]
            h[is_prompt] += vector.vector
        logits = einops.rearrange(model.output.logits, "b s tok -> b tok s").save()

    per_token_losses = cross_entropy(logits.value[:, :, :-1], tokens[:, 1:])
    per_token_losses = t.where(is_prompt[:, :-1].bool(), t.nan, per_token_losses)
    losses = per_token_losses.nanmean(-1)

    return losses


def _annotated_loss(
    model,
    batch,
    multiplier: float = 0,
    sv: Optional[SteeringVector] = None,
):
    vector = SteeringVector(sv.layer, sv.vector * multiplier) if sv else None
    losses = llm_loss(model, batch, vector)
    n = len(batch["input_ids"])
    return [
        {
            "layer": layer,
            "multiplier": multiplier,
            "index": batch["index"][i],
            "origin": batch["origin"][i],
            "loss": losses[i],
        }
        for i in range(n)
        for layer in ([sv.layer] if sv else range(model.config.n_layer))
    ]


def _collate_fn(examples, tokenizer):
    texts = [x["prompt"] + " " + x["completion"] for x in examples]
    origins = [x["origin"] for x in examples]
    pos_mask = t.Tensor([origin != "neg" for origin in origins]).bool()

    output = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    completion_lens = t.tensor(
        [len(s) for s in tokenizer([x["completion"] for x in examples]).input_ids]
    )
    total_len = output.input_ids.size(1)
    indices = t.arange(total_len)
    prompt_mask = indices[None, :] < total_len - completion_lens[:, None]

    return output | {
        "index": [x["index"] for x in examples],
        "origin": origins,
        "prompt_mask": prompt_mask,
        "pos_mask": pos_mask,
    }


@t.inference_mode()
def compute_spliced_llm_loss(
    model: LanguageModel,
    dataset: Dataset,
    steering_vectors: List[SteeringVector],
    multipliers: List[float],
    batch_size: int,
):
    loader = DataLoader(
        dataset,
        collate_fn=partial(_collate_fn, tokenizer=model.tokenizer),
        batch_size=batch_size,
    )
    results = []
    with Progress() as progress:
        task_n = len(loader) * (1 + len(steering_vectors) * len(multipliers))
        task_computing = progress.add_task("Computing loss...", total=task_n)

        for batch in loader:
            loss = partial(_annotated_loss, model=model, batch=batch)
            results += loss()
            progress.update(task_computing, advance=1)
            for sv in steering_vectors:
                for multiplier in (m for m in multipliers if m != 0):
                    progress.update(task_computing, advance=1)
                    results += loss(multiplier=multiplier, sv=sv)

    return pl.DataFrame(results)
