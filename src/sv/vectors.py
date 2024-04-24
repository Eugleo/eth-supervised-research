from dataclasses import dataclass
from pathlib import Path

import torch as t
from nnsight import LanguageModel
from rich.progress import track
from torch.utils.data import DataLoader

from sv.datasets import GenDataset


@dataclass
class SteeringVector:
    layer: int
    vector: t.Tensor

    @staticmethod
    def load(path: Path, device="cpu") -> "SteeringVector":
        data = t.load(path, map_location=device)
        return SteeringVector(layer=data["layer"], vector=data["vector"])

    def save(self, path: Path):
        t.save({"vector": self.vector, "layer": self.layer}, path)

    @staticmethod
    def generate(
        model: LanguageModel, dataset: GenDataset, layer: int
    ) -> "SteeringVector":
        batch_size = 1 if model.device == "cpu" else 32
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        differences = t.zeros(model.transformer.config.n_embd).to(model.device)  # type: ignore

        for batch in track(loader):
            with t.no_grad(), model.trace(scan=False, validate=False) as tracer:  # type: ignore
                with tracer.invoke(batch["pos"], scan=False) as _:
                    h_pos = model.transformer.h[layer].output[0].t[-1]

                with tracer.invoke(batch["neg"], scan=False) as _:
                    h_neg = model.transformer.h[layer].output[0].t[-1]

                difference = (h_pos - h_neg).sum(0).save()
            differences += difference

        steering_vector = differences / len(dataset)


        return SteeringVector(layer=layer, vector=steering_vector.cpu())
