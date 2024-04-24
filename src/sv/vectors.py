from dataclasses import dataclass
from pathlib import Path

import torch as t
from nnsight import LanguageModel
from rich.progress import track
from torch.utils.data import DataLoader

from sv.datasets import ContrastiveDataset


@dataclass
class SteeringVector:
    layer: int
    vector: t.Tensor

    @staticmethod
    def load(path: Path) -> "SteeringVector":
        data = t.load(path)
        return SteeringVector(layer=data["layer"], vector=data["vector"])

    def save(self, path: Path):
        t.save({"vector": self.vector, "layer": self.layer}, path)

    @staticmethod
    def generate(
        model: LanguageModel, dataset: ContrastiveDataset, layer: int
    ) -> "SteeringVector":
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        differences = t.zeros(model.transformer.config.n_embd)  # type: ignore

        for item in track(loader):
            with t.no_grad(), model.trace(scan=False, validate=False) as tracer:  # type: ignore
                with tracer.invoke(item["pos"]["complete"], scan=False) as _:
                    h_pos = model.transformer.h[layer].output[0].t[-1]

                with tracer.invoke(item["neg"]["complete"], scan=False) as _:
                    h_neg = model.transformer.h[layer].output[0].t[-1]

                difference = (h_pos - h_neg).sum(0).save()

            differences += difference

        steering_vector = differences / len(loader)

        return SteeringVector(layer=layer, vector=steering_vector)
