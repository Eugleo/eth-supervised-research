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
gpt2 = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
gpt2.to(device)
gpt2.reset_hooks()

autoencoder = SparseAutoencoder.from_pretrained(
    "gpt2-small-res-jb", f"blocks.{sae_layer}.hook_resid_pre"
)
autoencoder.to(device)


# %%
def load_vector(dataset, layer) -> jt.Float[t.Tensor, "dim"]:
    dataset_dir = Path("../data") / dataset
    _, vector_dir = utils.current_version(dataset_dir / "vectors")
    return SteeringVector.load(vector_dir / f"layer_{layer}.pt").vector


def decompose_into_features(autoencoder, vector):
    return autoencoder(vector.unsqueeze(0)).feature_acts.squeeze()


# %%
original_vector = load_vector(dataset, vector_layer).to(device)
vector = decompose_into_features(autoencoder, original_vector)


# %%
import os

import torch.nn.functional as F
from nnsight import LanguageModel
from sv.datasets import Dataset, DictDataset
from sv.loss import spliced_llm_loss
from torch.utils.data import DataLoader

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

model = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
tokenizer = model.tokenizer

_, dataset_dir = utils.current_version(Path("../data") / dataset / "dataset")
train_data = Dataset.load(dataset_dir / "generate.json").as_single_items()
val_data = Dataset.load(dataset_dir / "test.json").as_single_items()


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


batch_size = 64

train_loader = DataLoader(
    train_data, collate_fn=collate_fn, batch_size=batch_size, shuffle=True
)
eval_loader = DataLoader(val_data, collate_fn=collate_fn, batch_size=batch_size)

import wandb
from rich.progress import track

for p in model.parameters():
    p.requires_grad = False
model.eval()

for p in autoencoder.parameters():
    p.requires_grad = False
autoencoder.eval()

multipliers = t.nn.Parameter(vector.clone().detach())
orig_vector = SteeringVector(vector_layer, original_vector)

epochs = 10
lr = 1e-2
optimizer = t.optim.Adam([multipliers], lr=lr)

wandb.init(project="eth-supervised-research")


@t.no_grad()
def eval_step(model, loader, vector):
    pos_loss, neg_loss = 0, 0
    n, pos_n = len(val_data), len(val_data) // 2
    for batch in track(loader, description="Evaluating..."):
        losses = spliced_llm_loss(model, batch, vector)

        is_pos = batch["pos_mask"]
        pos_loss += losses[is_pos].sum().item()
        neg_loss += losses[~is_pos].sum().item()

    loss = (pos_loss - neg_loss) / n

    return pos_loss / pos_n, neg_loss / pos_n, loss


n_seen = 0
for epoch in range(epochs):
    for i, batch in enumerate(track(train_loader, description=f"Epoch {epoch + 1}...")):
        optimizer.zero_grad()
        new_vector = SteeringVector(vector_layer, multipliers @ autoencoder.W_dec)
        losses = spliced_llm_loss(model, batch, new_vector)

        is_pos = batch["pos_mask"]
        positive_loss, negative_loss = losses[is_pos].sum(), losses[~is_pos].sum()

        similarity = F.cosine_similarity(new_vector.vector, original_vector, dim=0)

        loss = positive_loss - negative_loss

        wandb.log(
            {
                "train/pos_loss": positive_loss.item() / batch_size,
                "train/neg_loss": negative_loss.item() / batch_size,
                "train/cos_similarity": similarity.item(),
                "train/loss": loss.item() / batch_size,
                "n_seen": n_seen,
            }
        )

        loss.backward()
        optimizer.step()
        n_seen += len(batch["input_ids"])

    with t.no_grad():
        new_vector = SteeringVector(vector_layer, multipliers @ autoencoder.W_dec)
        pos_loss, neg_loss = 0, 0
        orig_vector_loss = 0
        n, pos_n = len(val_data), len(val_data) // 2
        for batch in track(eval_loader, description="Evaluating..."):
            losses = spliced_llm_loss(model, batch, new_vector)

            is_pos = batch["pos_mask"]
            pos_loss += losses[is_pos].sum().item()
            neg_loss += losses[~is_pos].sum().item()

            orig_losses = spliced_llm_loss(model, batch, orig_vector)
            orig_vector_loss += (
                orig_losses[is_pos].sum().item() - orig_losses[~is_pos].sum().item()
            )

        new_vector_loss = (pos_loss - neg_loss) / n
        wandb.log(
            {
                "val/loss": new_vector_loss,
                "val/pos_loss": positive_loss.item() / pos_n,
                "val/neg_loss": negative_loss.item() / pos_n,
                "val/improvement": (orig_vector_loss / n) - new_vector_loss,
                "n_seen": n_seen,
            }
        )

wandb.finish()

# %%
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate

data = load_dataset("NeelNanda/c4-10k", split="train")
tokenized_data = tokenize_and_concatenate(data, model.tokenizer, max_length=128)  # type: ignore


# %%
def collate_fn_c4(examples):
    input_ids = t.stack([x["tokens"] for x in examples])
    prompt_mask = t.zeros_like(input_ids)
    prompt_mask[:, :64] = 1
    pos_mask = t.ones(len(examples)).bool()

    return {"input_ids": input_ids, "prompt_mask": prompt_mask, "pos_mask": pos_mask}


c4_loader = DataLoader(
    tokenized_data.with_format("torch"),  # type: ignore
    collate_fn=collate_fn_c4,
    batch_size=32,
    shuffle=True,
)

losses = []
for multiplier in [1, 5, 10, 15, 20, 25, 30]:
    loss = 0
    vector = SteeringVector(vector_layer, multiplier * orig_vector.vector)
    pos_loss, _, _ = eval_step(model, eval_loader, vector)
    _, _, c4_loss = eval_step(model, c4_loader, vector)
    losses.append({"multiplier": multiplier, "c4_loss": loss, "pos_loss": pos_loss})

# %%
