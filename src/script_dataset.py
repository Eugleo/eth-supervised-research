from pathlib import Path
from typing import Annotated, List

import typer
from rich.progress import Progress
from sv import utils
from sv.datasets import Dataset, Format
from typer import Argument, Option

app = typer.Typer()


@app.command()
def randomize(
    datasets: Annotated[List[str], Argument()],
    generate_size: Annotated[int, Option()],
    test_size: Annotated[int, Option()],
    format: Annotated[Format, Option()] = Format.repeated,
    name: Annotated[str, Option()] = "random",
    dataset_dir: Annotated[str, Option()] = "data",
    seed: Annotated[int, Option()] = 42,
):
    utils.set_seed(seed)

    data_paths = [
        Path(dataset_dir) / dataset / "original" / f"{set}.json"
        for dataset in datasets
        for set in ["generate", "test"]
    ]

    generate_dataset, test_dataset = Dataset.from_random_mix(
        data_paths, generate_size=generate_size, test_size=test_size, format=format
    )

    out_dir = Path(dataset_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_dataset.save(out_dir / "generate.json")
    test_dataset.save(out_dir / "test.json")


@app.command()
def load(
    datasets: Annotated[List[str], Argument()],
    format: Annotated[Format, Option()] = Format.repeated,
    data_dir: Annotated[str, Option()] = "data",
    seed: Annotated[int, Option()] = 42,
):
    utils.set_seed(seed)

    with Progress() as progress:
        task_n = len(datasets) * 2
        task_loading = progress.add_task("Loading and writing...", total=task_n)
        for name in datasets:
            dataset_dir = Path(data_dir) / name
            for set in ["generate", "test"]:
                orig_data = dataset_dir / "original" / f"{set}.json"
                dataset = Dataset.from_original(orig_data, format=format)
                dataset.save(dataset_dir / f"{set}.json")
                progress.update(task_loading, advance=1)


if __name__ == "__main__":
    app()
