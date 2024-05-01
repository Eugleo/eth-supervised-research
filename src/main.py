import os
from pathlib import Path
from typing import Annotated, List

import polars as pl
import sv.utils as utils
import torch as t
import typer
from nnsight import LanguageModel
from rich.progress import Progress
from sv.datasets import Dataset, DictDataset
from sv.loss import compute_spliced_llm_loss
from sv.vectors import SteeringVector
from typer import Argument, Option

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

app = typer.Typer()


@app.command()
def compute_loss(
    datasets: Annotated[List[str], Argument()],
    multipliers: Annotated[List[float], Option()] = [],
    data_dir: Annotated[str, Option()] = "data",
    model_id: Annotated[str, Option("--model")] = "openai-community/gpt2",
    seed: Annotated[int, Option()] = 42,
    device: Annotated[str, Option()] = "cpu",
):
    if not multipliers:
        multipliers = t.linspace(-30, 30, steps=21).tolist()
    utils.set_seed(seed)
    model = LanguageModel(model_id, device_map=device, dispatch=True)

    already_increased = set()

    for dataset_name in datasets:
        if "openwebtext" in dataset_name:
            vector_dataset_name, dataset_definition = dataset_name.split("+")
            _, limit, prompt_len = dataset_definition.split("-")

            dataset = DictDataset.from_openwebtext(
                int(limit), int(prompt_len), model.tokenizer
            )

            vectors_version, vectors_dir = utils.current_version(
                Path(data_dir) / vector_dataset_name / "vectors"
            )
            vectors = [
                SteeringVector.load(path, device) for path in (vectors_dir).iterdir()
            ]
            print(
                f"Loading {limit} items in OWT and {vector_dataset_name} vectors v{vectors_version}..."
            )
        else:
            dataset_version, dataset_dir = utils.current_version(
                Path(data_dir) / dataset_name / "dataset"
            )

            vectors_version, vectors_dir = utils.current_version(
                Path(data_dir) / dataset_name / "vectors"
            )
            vectors = [
                SteeringVector.load(path, device) for path in (vectors_dir).iterdir()
            ]
            print(
                f"Loading {dataset_name} v{dataset_version} and vectors v{vectors_version}..."
            )

            dataset = Dataset.load(dataset_dir / "test.json").as_single_items()

        losses = compute_spliced_llm_loss(model, dataset, vectors, multipliers)

        if "openwebtext" in dataset_name:
            if vector_dataset_name in already_increased:
                scores_version, scores_dir = utils.current_version(
                    Path(data_dir) / vector_dataset_name / "scores"
                )
                already_increased.add(vector_dataset_name)
            else:
                scores_version, scores_dir = utils.next_version(
                    Path(data_dir) / vector_dataset_name / "scores"
                )
            filename = dataset_definition.replace("-", "_")
        else:
            if dataset_name in already_increased:
                scores_version, scores_dir = utils.current_version(
                    Path(data_dir) / dataset_name / "scores"
                )
                already_increased.add(dataset_name)
            else:
                scores_version, scores_dir = utils.next_version(
                    Path(data_dir) / dataset_name / "scores"
                )
            filename = "test"

        losses.write_csv(scores_dir / f"{filename}.csv")
        print(f"Saved scores as v{scores_version}...")


@app.command()
def generate_vectors(
    datasets: Annotated[List[str], Argument()],
    data_dir: Annotated[str, Option()] = "data",
    model_id: Annotated[str, Option("--model")] = "openai-community/gpt2",
    seed: Annotated[int, Option()] = 42,
    device: Annotated[str, Option()] = "cpu",
):
    utils.set_seed(seed)
    model = LanguageModel(model_id, device_map=device, dispatch=True)
    for dataset_name in datasets:
        dataset_version, dataset_dir = utils.current_version(
            Path(data_dir) / dataset_name / "dataset"
        )
        print(
            f"Generating vectors for v{dataset_version} of the {dataset_name} dataset..."
        )

        vector_version, vector_dir = utils.next_version(
            Path(data_dir) / dataset_name / "vectors"
        )
        dataset_name = Dataset.load(dataset_dir / "generate.json").as_prefixed_pairs()

        for vector in SteeringVector.generate_all(model, dataset_name):
            vector.save(vector_dir / f"layer_{vector.layer}.pt")
        print(f"Saved vectors as v{vector_version}...")


@app.command()
def create_dataset(
    datasets: Annotated[List[str], Argument()],
    kind: Annotated[str, Option()] = "gpt",
    data_dir: Annotated[str, Option()] = "data",
    seed: Annotated[int, Option()] = 42,
):
    utils.set_seed(seed)

    with Progress() as progress:
        task_n = len(datasets) * 2
        task_loading = progress.add_task("Creating datasets...", total=task_n)
        for name in datasets:
            load_dir = Path(data_dir) / name / "dataset" / "source"
            version, save_dir = utils.next_version(Path(data_dir) / name / "dataset")
            print(f"Creating dataset v{version} for {name}...")
            if kind == "rimsky":
                for set in ["generate", "test"]:
                    source_data = load_dir / f"{set}.json"
                    dataset = Dataset.from_rimsky(source_data)
                    dataset.save(save_dir / f"{set}.json")
                    progress.update(task_loading, advance=1)
            elif kind == "gpt":
                source_data = load_dir / "data.json"
                train, test = Dataset.from_gpt(source_data, test_size=50)
                train.save(save_dir / "generate.json")
                test.save(save_dir / "test.json")
                progress.update(task_loading, advance=2)


if __name__ == "__main__":
    app()
