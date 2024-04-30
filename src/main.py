import os
from pathlib import Path
from typing import Annotated, List

import sv.utils as utils
import typer
from nnsight import LanguageModel
from rich.progress import Progress
from sv.datasets import Dataset
from sv.vectors import SteeringVector
from typer import Argument, Option

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

app = typer.Typer()


def current_version(path: Path) -> tuple[int, Path]:
    with open(path / ".version") as f:
        version = int(f.read().strip())
        return version, path / f"v{version}"


def next_version(path: Path) -> tuple[int, Path]:
    previous_version = (
        0 if not (path / ".version").exists() else current_version(path)[0]
    )
    version = previous_version + 1
    with open(path / ".version", "w") as f:
        f.write(str(version))
    dir = path / f"v{version}"
    dir.mkdir(parents=True, exist_ok=True)
    return version, dir


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
        dataset_version, dataset_dir = current_version(
            Path(data_dir) / dataset_name / "dataset"
        )
        print(
            f"Generating vectors for v{dataset_version} of the {dataset_name} dataset..."
        )

        vector_version, vector_dir = next_version(
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
            version, save_dir = next_version(Path(data_dir) / name / "dataset")
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
