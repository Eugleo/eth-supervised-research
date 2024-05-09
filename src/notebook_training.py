# %%
import os
import re
from functools import partial
from pathlib import Path

import plotly.express as px
import polars as pl
import torch as t
from datasets import load_dataset
from nnsight import LanguageModel
from polars import col as c
from rich import print as rprint
from rich.progress import Progress
from sae_lens import SparseAutoencoder
from sv import utils
from sv.datasets import Dataset
from sv.scoring import activations, llm_loss
from sv.vectors import SteeringVector
from torch.utils.data import DataLoader
from transformer_lens.utils import tokenize_and_concatenate

import wandb

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def compute_tradeoff(trainer, vector, multipliers, name):
    c4_data = load_dataset("NeelNanda/c4-10k", split="train[:1%]")
    c4_tokenized = tokenize_and_concatenate(
        c4_data, trainer.model.tokenizer, max_length=128
    )  # type: ignore
    c4_loader = DataLoader(
        c4_tokenized.with_format("torch"),  # type: ignore
        collate_fn=collate_fn_c4,
        batch_size=32,
    )
    successes = p_successful_rollout(
        trainer.model,
        "My favorite doll is a little",
        vector,
        gender_keywords,
        multipliers=multipliers,
    )
    results = []
    for multiplier in multipliers:
        with Progress() as progress:
            c4_loss = trainer.compute_loss(
                vector * multiplier, c4_loader, progress=progress
            )
            results.append(
                {
                    "multiplier": multiplier,
                    "c4_loss": c4_loss,
                    "p_success": successes[multiplier],
                    "vector": name,
                }
            )
    return results


# %%
baseline_vector = trainer.baseline_vector
vector = vectors_l2[0]

if "tradeoff_zero" not in locals():
    tradeoff_zero = compute_tradeoff(
        trainer, baseline_vector, name="zero", multipliers=[0]
    )

if "tradeoff_baseline" not in locals():
    tradeoff_baseline = compute_tradeoff(
        trainer,
        baseline_vector,
        name="original",
        multipliers=[-2, -1, 1, 2, 3],
    )

if "tradeoffs" not in locals():
    tradeoffs = tradeoff_baseline
ts = compute_tradeoff(
    trainer, vector, name="+p –p +n –n (e1)", multipliers=[-1, -0.5, 0.5, 0.75, 1]
)
assert isinstance(tradeoffs, list)
tradeoffs += ts


# %%
baseline_c4_loss = tradeoff_zero[0]["c4_loss"]
baseline_steering_loss = tradeoff_zero[0]["p_success"]

df = (
    pl.DataFrame(tradeoffs)
    .sort("vector", "p_success")
    .filter(c("c4_loss") < 3.75, c("vector").is_in(["original", "+p –p +n –n (e1)"]))
)

fig = px.line(
    df.to_pandas(),
    x="c4_loss",
    y="p_success",
    markers=True,
    color="vector",
    hover_data=["multiplier"],
    title="Pareto frontier of Predictive loss v. P(Success)",
    labels={"c4_loss": "Predictive loss on C4", "p_success": "P(Success)"},
)
fig.add_hline(y=baseline_steering_loss, line_dash="dash", annotation_text="Baseline")
fig.add_vline(x=baseline_c4_loss, line_dash="dash", annotation_text="Baseline")
fig.show()


# %%
from rich import print as rprint
from rich.table import Table

model = trainer.model
model.to("cuda")


sv = baseline_vector * 8
sv = vector * -2


def generate_freeform(start_prompt, vector):
    completions = Table(
        "Completion",
        show_lines=True,
        width=60,
    )
    for _ in range(5):
        with t.no_grad(), model.generate(
            start_prompt,
            max_new_tokens=40,
            scan=False,
            validate=False,
            pad_token_id=model.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.6,
            temperature=1,
        ) as _:
            model.transformer.h[vector.layer].output[0][:] += vector.vector
            steered_output = model.generator.output.save()
        prompt = model.tokenizer.batch_decode(steered_output.value)[0]
        completions.add_row(prompt)
    rprint(completions)


# sv = SteeringVector.load(vector_dir / "layer_8.pt", device="cuda") * 3
generate_freeform("My favorite doll is a little", sv)
