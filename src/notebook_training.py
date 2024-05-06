# %%
from functools import partial
from pathlib import Path
from typing import Any

import jaxtyping as jt
import plotly.express as px
import polars as pl
import torch as t
import transformer_lens
from polars import col as c
from sae_lens import SparseAutoencoder
from sv import utils
from sv.vectors import SteeringVector
from transformer_lens import HookedTransformer
from transformer_lens import utils as tutils


def set_seed(seed: int) -> None:
    import os
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


set_seed(42)

dataset = "gender"
vector_layer = 8
sae_layer = vector_layer + 1
device = t.device("cpu")

# %%
autoencoder = SparseAutoencoder.from_pretrained(
    "gpt2-small-res-jb", f"blocks.{sae_layer}.hook_resid_pre"
)
autoencoder.to(device)


# %%
def load_vector(layer) -> jt.Float[t.Tensor, "dim"]:
    return SteeringVector.load(f"../data/gender/vectors/v1/layer_{layer}.pt").vector


def decompose_into_features(autoencoder, vector):
    return autoencoder(vector.unsqueeze(0)).feature_acts.squeeze()


original_vector = load_vector(dataset, vector_layer).to(device)
decomposed_vector = decompose_into_features(autoencoder, original_vector)


import os

import torch.nn.functional as F
import wandb
from datasets import load_dataset
from nnsight import LanguageModel
from rich.progress import Progress
from sv.datasets import Dataset, DictDataset
from sv.loss import spliced_llm_loss
from torch.utils.data import DataLoader
from transformer_lens.utils import tokenize_and_concatenate

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
tokenizer = model.tokenizer

_, dataset_dir = utils.current_version(Path("../data") / dataset / "dataset")
pos_train, neg_train = Dataset.load(dataset_dir / "generate.json").as_single_items()
pos_eval, neg_eval = Dataset.load(dataset_dir / "test.json").as_single_items()


def collate_fn(examples):
    texts = [x["prompt"] + " " + x["completion"] for x in examples]
    pos_mask = t.Tensor([x["origin"] == "pos" for x in examples]).bool()

    output = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    completion_lens = t.tensor(
        [len(s) for s in model.tokenizer([x["completion"] for x in examples]).input_ids]
    )
    total_len = output.input_ids.size(1)
    indices = t.arange(total_len)
    prompt_mask = indices[None, :] < total_len - completion_lens[:, None]

    return output | {"prompt_mask": prompt_mask, "pos_mask": pos_mask}


batch_size = 16

pos_train_loader = DataLoader(pos_train, collate_fn=collate_fn, batch_size=batch_size)
neg_train_loader = DataLoader(neg_train, collate_fn=collate_fn, batch_size=batch_size)
pos_eval_loader = DataLoader(pos_eval, collate_fn=collate_fn, batch_size=batch_size)
neg_eval_loader = DataLoader(neg_eval, collate_fn=collate_fn, batch_size=batch_size)


for p in model.parameters():
    p.requires_grad = False
model.eval()

for p in autoencoder.parameters():
    p.requires_grad = False
autoencoder.eval()

multipliers = t.nn.Parameter(decomposed_vector.clone().detach())
orig_vector = SteeringVector(vector_layer, original_vector)

epochs = 10
lr = 1e-2
optimizer = t.optim.Adam([multipliers], lr=lr)

wandb.init(project="eth-supervised-research")


@t.no_grad()
def eval_step(model, vector, pos_loader, neg_loader=None):
    pos_loss, neg_loss = 0, 0
    pos_n = len(pos_loader.dataset)
    n = pos_n + len(neg_loader.dataset) if neg_loader is not None else pos_n
    batches = (
        zip(pos_loader, neg_loader)
        if neg_loader is not None
        else ((i, None) for i in pos_loader)
    )
    for pos_batch, neg_batch in batches:
        pos_loss += spliced_llm_loss(model, pos_batch, vector).sum().item()
        if neg_batch is not None:
            neg_loss += spliced_llm_loss(model, neg_batch, vector).sum().item()

    avg_pos_loss = pos_loss / pos_n if pos_n > 0 else 0
    avg_neg_loss = neg_loss / (n - pos_n) if n - pos_n > 0 else 0
    avg_loss = (pos_loss - neg_loss) / n

    return avg_pos_loss, avg_neg_loss, avg_loss


vectors = []

old_vector = SteeringVector(vector_layer, orig_vector.vector * 3)
_, _, orig_loss = eval_step(model, old_vector, pos_eval_loader, neg_eval_loader)

new_vector = SteeringVector(
    vector_layer, multipliers @ autoencoder.W_dec + autoencoder.b_dec
)
pos_loss, neg_loss, loss = eval_step(
    model, new_vector, pos_eval_loader, neg_eval_loader
)


wandb.log(
    {
        "val/loss": loss,
        "val/pos_loss": pos_loss,
        "val/neg_loss": neg_loss,
        "val/improvement": orig_loss - loss,
        "n_seen": 0,
    }
)

n_seen = 0
with Progress() as progress:
    for epoch in range(epochs):
        train_task = progress.add_task(
            f"Epoch {epoch + 1}...", total=len(pos_train_loader.dataset)
        )
        batches = zip(pos_train_loader, neg_train_loader)
        for i, (pos_batch, neg_batch) in enumerate(batches):
            optimizer.zero_grad()
            new_vector = SteeringVector(
                vector_layer, multipliers @ autoencoder.W_dec + autoencoder.b_dec
            )
            pos_loss = spliced_llm_loss(model, pos_batch, new_vector).sum()
            neg_loss = spliced_llm_loss(model, neg_batch, new_vector).sum()

            penalty_l2 = ((new_vector.vector - original_vector) ** 2).sum()
            penalty_l1 = 0.05 * (t.abs(multipliers)).sum()

            loss = pos_loss - neg_loss + 0.01 * penalty_l2

            loss.backward()
            optimizer.step()
            n_seen += len(pos_batch["input_ids"]) + len(neg_batch["input_ids"])
            progress.update(train_task, advance=batch_size)

            similarity = F.cosine_similarity(new_vector.vector, original_vector, dim=0)

            wandb.log(
                {
                    "train/pos_loss": pos_loss.item() / batch_size,
                    "train/neg_loss": neg_loss.item() / batch_size,
                    "train/cos_similarity": similarity.item(),
                    "train/loss": loss / batch_size,
                    "n_seen": n_seen,
                }
            )
        progress.stop_task(train_task)

        with t.no_grad():
            eval_task = progress.add_task("Evaluating...", total=1)

            new_vector = SteeringVector(
                vector_layer, multipliers @ autoencoder.W_dec + autoencoder.b_dec
            )
            pos_loss, neg_loss, loss = eval_step(
                model, new_vector, pos_eval_loader, neg_eval_loader
            )
            progress.update(eval_task, advance=1)

            vectors.append(new_vector.vector.detach())

            wandb.log(
                {
                    "val/loss": loss,
                    "val/pos_loss": pos_loss,
                    "val/neg_loss": neg_loss,
                    "val/improvement": orig_loss - loss,
                    "n_seen": n_seen,
                }
            )
            progress.stop_task(eval_task)

wandb.finish()


# %%
def collate_fn_c4(examples):
    input_ids = t.stack([x["tokens"] for x in examples])
    prompt_mask = t.zeros_like(input_ids)
    prompt_mask[:, :64] = 1
    pos_mask = t.ones(len(examples)).bool()

    return {"input_ids": input_ids, "prompt_mask": prompt_mask, "pos_mask": pos_mask}


c4_data = load_dataset("NeelNanda/c4-10k", split="train[:2%]")
c4_tokenized = tokenize_and_concatenate(c4_data, model.tokenizer, max_length=128)  # type: ignore
c4_loader = DataLoader(
    c4_tokenized.with_format("torch"),  # type: ignore
    collate_fn=collate_fn_c4,
    batch_size=32,
)

# %%
baseline_loss = []
for multiplier in [1, 2, 3, 5, 10, 15, 30]:
    print(f"Multiplier: {multiplier}")
    vector = SteeringVector(vector_layer, multiplier * original_vector)
    _, _, loss = eval_step(model, vector, pos_eval_loader, neg_eval_loader)
    print(loss)
    # _, _, c4_loss = eval_step(model, vector, c4_loader)
    # baseline_loss.append(
    #     {
    #         "multiplier": multiplier,
    #         "c4_loss": c4_loss,
    #         "loss": loss,
    #         "vector": "static+scaled",
    #     }
    # )

# %%
final_vector = SteeringVector(vector_layer, vectors[5])


loss_vanilla = []
for multiplier in [1, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]:
    print(f"Multiplier: {multiplier}")
    vector = SteeringVector(vector_layer, multiplier * final_vector.vector)
    _, _, loss = eval_step(model, vector, pos_eval_loader, neg_eval_loader)
    print(loss)
    # _, _, c4_loss = eval_step(model, vector, c4_loader)
    # loss_vanilla.append(
    #     {
    #         "multiplier": multiplier,
    #         "c4_loss": c4_loss,
    #         "loss": loss,
    #         "vector": "vanilla",
    #     }
    # )

# %%
df = pl.DataFrame(baseline_loss + loss_vanilla).sort("c4_loss")

fig = px.line(
    df.to_pandas(),
    x="c4_loss",
    y="loss",
    markers=True,
    color="vector",
    hover_data=["multiplier"],
    title="Pareto frontier of Predictive loss v. Steering loss",
    labels={"c4_loss": "Predictive loss on C4", "loss": "Steering loss"},
)
fig.show()
fig.write_image("pareto_l2_fixed.pdf")

# %%
final_vec = SteeringVector(vector_layer, multipliers @ autoencoder.W_dec)
eval_step(model, c4_loader, final_vec)

# %%

_, vector_dir = utils.next_version(Path("../data") / dataset / "vectors")
final_vector.save(vector_dir / "layer_8.pt")

# %%
