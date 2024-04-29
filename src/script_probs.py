import os
from functools import partial
from pathlib import Path
from typing import List, Optional

import einops
import polars as pl
import sv.utils as utils
import torch as t
import typer
from nnsight import LanguageModel
from rich.progress import Progress
from sv.datasets import Dataset, EvalDataset
from sv.vectors import SteeringVector
from torch.utils.data import DataLoader
from typer import Argument, Option
from typing_extensions import Annotated

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def probs_for_prompts(model, prefixes, suffixes, sv, coeff):
    criterion = t.nn.CrossEntropyLoss(reduction="none").to(model.device)
    sequences = [f"{p} {s}" for p, s in zip(prefixes, suffixes)]
    tokens = model.tokenizer(sequences, return_tensors="pt", padding=True).input_ids

    total_length = tokens.size(1)
    indices = t.arange(total_length)[None, :]

    suffix_tokens = model.tokenizer(suffixes).input_ids
    suffix_lengths = t.tensor([len(s) for s in suffix_tokens])[:, None]
    is_prefix = indices < total_length - suffix_lengths

    with t.no_grad(), model.trace(sequences, scan=False, validate=False) as _:
        if sv and coeff != 0:
            h = model.transformer.h[sv.layer].output[0]
            h[is_prefix] += coeff * sv.vector
        logits = model.output.logits
        logits = einops.rearrange(logits, "b s tok -> b tok s").save()

    tokens = tokens.to(logits.device)
    losses = criterion(logits.value[:, :, :-1], tokens[:, 1:])

    prefix_tokens = model.tokenizer(prefixes).input_ids
    prefix_lengths = t.tensor([len(p) for p in prefix_tokens])[:, None]
    is_padding = indices < total_length - suffix_lengths - prefix_lengths

    losses[is_padding[:, :-1]] = 0
    return losses.mean(-1).exp().tolist()


def probs_for_batch(
    model, batch, coeff: float = 0, sv: Optional[SteeringVector] = None, kind="turner"
):
    if kind == "rimsky":
        prompts, neg_tokens, pos_tokens = (
            batch["prompt"],
            batch["neg_token"],
            batch["pos_token"],
        )
        with t.no_grad(), model.trace(scan=False, validate=False) as runner:
            with runner.invoke(prompts, scan=False) as _:
                if sv and coeff != 0:
                    model.transformer.h[sv.layer].output[0][:] += coeff * sv.vector
                output = model.lm_head.output.t[-1].softmax(dim=-1)
                neg_probs = output[t.arange(len(prompts)), neg_tokens].tolist().save()
                pos_probs = output[t.arange(len(prompts)), pos_tokens].tolist().save()

        return [
            {
                "q_num": q_num,
                "ordering": ordering,
                "measured_token": measured_token,
                "measured_prob": prob,
                "intervention_layer": sv.layer if sv else None,
                "intervention_coeff": coeff,
            }
            for measured_token, probs in [("neg", neg_probs), ("pos", pos_probs)]
            for q_num, ordering, prob in zip(batch["q_num"], batch["ordering"], probs)
        ]
    elif kind == "turner":
        prompts, neg, pos = (batch["prompt"], batch["neg"], batch["pos"])

        neg_losses = probs_for_prompts(model, prompts, neg, sv, coeff)
        pos_losses = probs_for_prompts(model, prompts, pos, sv, coeff)

        return [
            {
                "q_num": q_num,
                "ordering": "none",
                "measured_token": measured_token,
                "measured_prob": float(prob),
                "intervention_layer": sv.layer if sv else None,
                "intervention_coeff": coeff,
            }
            for measured_token, probs in [("neg", neg_losses), ("pos", pos_losses)]
            for q_num, prob in zip(batch["q_num"], probs)
        ]


@t.inference_mode()
def probs_for_dataset(
    model: LanguageModel,
    dataset: EvalDataset,
    steering_vectors: List[SteeringVector],
    coeffs: List[float],
):
    if 0 not in coeffs:
        raise ValueError("0 should be in the list of coefficients")
    batch_size = 1 if model.device == "cpu" else 64
    loader = DataLoader(dataset, batch_size=batch_size)
    results = []

    with Progress() as progress:
        task_n = len(loader) * (1 + len(steering_vectors) * (len(coeffs) - 1))
        task_computing = progress.add_task("Computing...", total=task_n)

        for batch in loader:
            invoke = partial(probs_for_batch, model=model, batch=batch)
            progress.update(task_computing, advance=1)
            results += [
                b | {"intervention_layer": sv.layer}
                for b in invoke()
                for sv in steering_vectors
            ]
            for sv in steering_vectors:
                for coeff in (c for c in coeffs if c != 0):
                    progress.update(task_computing, advance=1)
                    results += invoke(coeff=coeff, sv=sv)

    return pl.DataFrame(results)


def main(
    datasets: Annotated[List[str], Argument()],
    data_dir: Annotated[str, Option()] = "data",
    only_test: Annotated[bool, Option()] = True,
    coeffs: Annotated[List[float], Option()] = [],
    model_id: Annotated[str, Option("--model")] = "openai-community/gpt2",
    seed: Annotated[int, Option()] = 42,
    device: Annotated[str, Option()] = "cpu",
):
    if not coeffs:
        coeffs = t.linspace(-30, 30, steps=21).tolist()
        # coeffs = [-20, -10, 0, 10, 20]

    utils.set_seed(seed)
    model = LanguageModel(model_id, device_map=device, dispatch=True)

    print(f"Processing {len(datasets)} datasets and {len(coeffs)} coefficients")
    for dataset in datasets:
        print(f"Processing dataset {dataset}...")
        dataset_dir = Path(data_dir) / dataset
        vec_dir = dataset_dir / "vectors"
        vectors = [SteeringVector.load(path, device) for path in vec_dir.iterdir()]
        for set in ["test", "generate"]:
            if only_test and set != "test":
                continue

            data = Dataset.load(dataset_dir / f"{set}.json").for_evaluating(
                tokenizer=model.tokenizer, kind="turner"
            )
            probs = probs_for_dataset(model, data, vectors, coeffs)

            probs_dir = dataset_dir / "probs"
            probs_dir.mkdir(parents=True, exist_ok=True)
            probs.write_csv(probs_dir / f"{set}.csv")


if __name__ == "__main__":
    typer.run(main)
