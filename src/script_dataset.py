from pathlib import Path
from typing import Annotated, List

import typer
from nnsight import LanguageModel
from rich.progress import Progress
from sv import utils
from sv.datasets import ContrastiveDataset
from typer import Argument, Option

app = typer.Typer()


@app.command()
def randomize(
    datasets: Annotated[List[str], Argument()],
    generate_size: Annotated[int, Option()],
    test_size: Annotated[int, Option()],
    name: Annotated[str, Option()] = "random",
    dataset_dir: Annotated[str, Option()] = "data",
    model_id: Annotated[str, Option("--model")] = "openai-community/gpt2",
    seed: Annotated[int, Option()] = 42,
):
    utils.set_seed(seed)
    model = LanguageModel(model_id)

    data_paths = [
        Path(dataset_dir) / dataset / "original" / f"{set}.json"
        for dataset in datasets
        for set in ["generate", "test"]
    ]

    generate_dataset, test_dataset = ContrastiveDataset.from_random_mix(
        data_paths,
        model.tokenizer,
        generate_size=generate_size,
        test_size=test_size,
        answer_most_recent=True,
    )

    out_dir = Path(dataset_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_dataset.save(out_dir / "generate.json")
    test_dataset.save(out_dir / "test.json")


@app.command()
def load(
    datasets: Annotated[List[str], Argument()],
    dataset_dir: Annotated[str, Option()] = "data",
    model_id: Annotated[str, Option("--model")] = "openai-community/gpt2",
    seed: Annotated[int, Option()] = 42,
):
    utils.set_seed(seed)
    model = LanguageModel(model_id)

    with Progress() as progress:
        task_n = len(datasets) * 2
        task_loading = progress.add_task("Loading and writing...", total=task_n)
        for name in datasets:
            for set in ["generate", "test"]:
                data_path = Path(dataset_dir) / name / "original" / f"{set}.json"
                dataset = ContrastiveDataset.from_rimsky(
                    data_path, model.tokenizer, answer_most_recent=True
                )
                dataset.save(Path(dataset_dir) / name / f"{set}.json")
                progress.update(task_loading, advance=1)


if __name__ == "__main__":
    app()
