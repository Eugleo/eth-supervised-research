import os
from pathlib import Path
from typing import Annotated, List

import sv.utils as utils
import typer

# Assuming these are defined somewhere
from nnsight import LanguageModel
from sv.datasets import Dataset
from sv.vectors import SteeringVector
from typer import Argument, Option

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def main(
    datasets: Annotated[List[str], Argument()],
    data_dir: Annotated[str, Option()] = "data",
    model_id: Annotated[str, Option("--model")] = "openai-community/gpt2",
    seed: Annotated[int, Option()] = 42,
    device: Annotated[str, Option()] = "cpu",
):
    utils.set_seed(seed)
    model = LanguageModel(model_id, device_map=device, dispatch=True)

    for dataset in datasets:
        print(f"Generating for {dataset}...")
        dataset_path = Path(data_dir) / dataset
        vector_dir = dataset_path / "vectors"
        vector_dir.mkdir(parents=True, exist_ok=True)
        dataset = Dataset.load(dataset_path / "generate.json").for_vector_generation(
            model.tokenizer, kind="turner"
        )

        for vector in SteeringVector.generate_all(model, dataset):
            vector.save(vector_dir / f"layer_{vector.layer}.pt")


if __name__ == "__main__":
    typer.run(main)
