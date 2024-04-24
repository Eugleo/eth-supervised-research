import os
from functools import partial
from pathlib import Path
from typing import List, Optional

import polars as pl
import sv.utils as utils
import torch as t
import typer
from nnsight import LanguageModel
from rich.progress import Progress
from sv.datasets import ContrastiveDataset
from sv.vectors import SteeringVector
from typer import Argument, Option
from typing_extensions import Annotated

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def probs_for_batch(
    model, batch, coeff: float = 0, sv: Optional[SteeringVector] = None
):
    # Here we can ignore "neg" since in the dataset there are
    # both (pos, neg) and (neg, pos) variants
    prompts = batch["pos"]["prompt"]
    neg_tokens, pos_tokens = batch["neg"]["answer_token"], batch["pos"]["answer_token"]

    with t.no_grad(), model.trace(scan=False, validate=False) as runner:
        with runner.invoke(prompts, scan=False) as _:
            if sv and coeff != 0:
                model.transformer.h[sv.layer].output[0][:] += coeff * sv.vector
            output = model.lm_head.output.t[-1].softmax(dim=-1)
            neg_probs = output[t.arange(len(prompts)), neg_tokens].tolist().save()
            pos_probs = output[t.arange(len(prompts)), pos_tokens].tolist().save()
            output_tokens = output.argmax(dim=-1).tolist().save()

    argmax_tokens = []
    for neg, pos, output in zip(neg_tokens, pos_tokens, output_tokens):
        if output == neg:
            argmax_tokens.append("neg")
        elif output == pos:
            argmax_tokens.append("pos")
        else:
            argmax_tokens.append("other")

    return [
        {
            "id": id,
            # If answer_first is True, then the first option is the positive one
            # Since we are only looking at the prompts from the "pos" POV
            "answer_was_first_option": answer_first
            if answer == "pos"
            else not answer_first,
            "measured_token": answer,
            "measured_prob": prob,
            "argmax_token": argmax_token,
            "intervention_layer": sv.layer if sv else None,
            "intervention_coeff": coeff,
        }
        for answer, probs in [("neg", neg_probs), ("pos", pos_probs)]
        for id, answer_first, argmax_token, prob in zip(
            batch["id"], batch["answer_is_first_option"], argmax_tokens, probs
        )
    ]


@t.inference_mode()
def probs_for_dataset(
    model: LanguageModel,
    dataset: ContrastiveDataset,
    steering_vectors: List[SteeringVector],
    coeffs: List[float],
):
    if 0 not in coeffs:
        raise ValueError("0 should be in the list of coefficients")

    loader = t.utils.data.DataLoader(dataset, batch_size=1)
    results = []

    with Progress() as progress:
        task_n = len(loader) * 2 * (1 + len(steering_vectors) * (len(coeffs) - 1))
        task_computing = progress.add_task("Computing...", total=task_n)

        for batch in loader:
            invoke = partial(probs_for_batch, model=model, batch=batch)
            progress.update(task_computing, advance=1)
            results += invoke()
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
):
    if not coeffs:
        coeffs = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]

    utils.set_seed(seed)
    model = LanguageModel(model_id)

    for dataset in datasets:
        print(f"Processing dataset {dataset}...")
        dataset_dir = Path(data_dir) / dataset
        vec_dir = dataset_dir / "vectors"
        vectors = [SteeringVector.load(path) for path in vec_dir.iterdir()]
        for set in ["test", "generate"]:
            if only_test and set != "test":
                continue

            data = ContrastiveDataset.load(dataset_dir / f"{set}.json")
            probs = probs_for_dataset(model, data, vectors, coeffs)

            probs_dir = dataset_dir / "probs"
            probs_dir.mkdir(parents=True, exist_ok=True)
            probs.write_csv(probs_dir / f"{set}.csv")


if __name__ == "__main__":
    typer.run(main)
