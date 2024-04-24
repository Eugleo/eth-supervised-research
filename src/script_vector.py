import os
from pathlib import Path
from typing import Annotated, List

import sv.utils as utils
import typer

# Assuming these are defined somewhere
from nnsight import LanguageModel
from sv.datasets import ContrastiveDataset
from sv.vectors import SteeringVector
from typer import Argument, Option

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def main(
    datasets: Annotated[List[str], Argument()],
    data_dir: Annotated[str, Option()] = "data",
    layers: Annotated[List[int], Option()] = list(range(12)),
    model_id: Annotated[str, Option("--model")] = "openai-community/gpt2",
    seed: Annotated[int, Option()] = 42,
):
    utils.set_seed(seed)
    model = LanguageModel(model_id)

    for dataset in datasets:
        print(f"Generating for {dataset}...")
        dataset_path = Path(data_dir) / dataset
        vector_dir = dataset_path / "vectors"
        vector_dir.mkdir(parents=True, exist_ok=True)
        dataset = ContrastiveDataset.load(dataset_path / "generate.json")

        for layer in layers:
            print(f"Processing layer {layer}...")
            vector = SteeringVector.generate(model, dataset, layer)
            vector.save(vector_dir / f"layer_{layer}.pt")


if __name__ == "__main__":
    typer.run(main)
