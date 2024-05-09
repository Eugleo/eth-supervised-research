from dataclasses import dataclass
from pathlib import Path

import torch as t
from nnsight import LanguageModel
from rich.progress import track
from torch.utils.data import DataLoader, Dataset


@dataclass
class SteeringVector:
    layer: int
    vector: t.Tensor

    def __mul__(self, other: float):
        return SteeringVector(layer=self.layer, vector=other * self.vector)

    def __rmul__(self, other: float):
        return self.__mul__(other)

    def __neg__(self):
        return -1 * self

    @staticmethod
    def load(path: Path, device="cpu") -> "SteeringVector":
        data = t.load(path, map_location=device)
        return SteeringVector(layer=data["layer"], vector=data["vector"])

    def save(self, path: Path):
        t.save({"vector": self.vector, "layer": self.layer}, path)

    @staticmethod
    def generate_all(model: LanguageModel, dataset: Dataset) -> list["SteeringVector"]:
        batch_size = 1 if model.device == "cpu" else 32
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        n_layers: int = model.transformer.config.n_layer  # type: ignore
        n_dims: int = model.transformer.config.n_embd  # type: ignore
        diff_by_layer = t.zeros(n_layers, n_dims).to(model.device)  # type: ignore

        for batch in track(loader):
            with t.no_grad(), model.trace(scan=False, validate=False) as tracer:  # type: ignore
                with tracer.invoke(batch["pos"]) as _:
                    for layer in range(n_layers):
                        h = model.transformer.h[layer].output[0].t[-1].sum(0).save()
                        diff_by_layer[layer] += h

                with tracer.invoke(batch["neg"]) as _:
                    for layer in range(n_layers):
                        h = model.transformer.h[layer].output[0].t[-1].sum(0).save()
                        diff_by_layer[layer] -= h

        steering_vectors = diff_by_layer / len(dataset)  # type: ignore

        return [
            SteeringVector(layer=i, vector=vec.cpu())
            for i, vec in enumerate(steering_vectors)
        ]
