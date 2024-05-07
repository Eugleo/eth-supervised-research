# %%
import os
import re
from functools import partial
from pathlib import Path

import plotly.express as px
import polars as pl
import torch as t
import wandb
from datasets import load_dataset
from nnsight import LanguageModel
from polars import col as c
from rich import print as rprint
from rich.progress import Progress
from sae_lens import SparseAutoencoder
from sv import utils
from sv.datasets import Dataset
from sv.loss import llm_loss
from sv.vectors import SteeringVector
from torch.utils.data import DataLoader
from transformer_lens.utils import tokenize_and_concatenate

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

gender_keywords = [
    "dad",
    "father",
    "brother",
    "man",
    "men",
    "dude",
    "buddy",
    "boy",
    "guy",
    "gun",
    "prince",
    "king",
    "husband",
]


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


def has_keyword(text, keywords):
    if any(k in text for k in keywords):
        return any(re.findall(rf"\b{k}'?s?\b", text) for k in keywords)
    return False


def p_successful_rollout(
    model, prompt, sv, keywords, multipliers=[1], length=40, batch_size=200
):
    rollouts = {}
    batch = [prompt] * batch_size

    with t.no_grad(), model.generate(
        batch,
        max_new_tokens=length,
        scan=False,
        validate=False,
        pad_token_id=model.tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.8,
        temperature=1,
    ) as _:
        baseline_output = model.generator.output.save()
    baseline_decoded = model.tokenizer.batch_decode(
        baseline_output, skip_special_tokens=True
    )
    baseline_p_success = (
        sum(any(k in d for k in keywords) for d in baseline_decoded) / batch_size
    )
    rollouts[0] = baseline_p_success

    for multiplier in multipliers:
        with t.no_grad(), model.generate(
            batch,
            max_new_tokens=length,
            scan=False,
            validate=False,
            pad_token_id=model.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.8,
            temperature=1,
        ) as _:
            model.transformer.h[sv.layer].output[0][:] += multiplier * sv.vector
            steered_output = model.generator.output.save()
        decoded = model.tokenizer.batch_decode(steered_output, skip_special_tokens=True)
        rollouts[multiplier] = (
            sum(has_keyword(d, keywords) for d in decoded) / batch_size
        )

    return rollouts


def collate_fn_c4(examples):
    input_ids = t.stack([x["tokens"] for x in examples])
    prompt_mask = t.zeros_like(input_ids)
    prompt_mask[:, :64] = 1
    pos_mask = t.ones(len(examples)).bool()

    return {"input_ids": input_ids, "prompt_mask": prompt_mask, "pos_mask": pos_mask}


class Trainer:
    def __init__(self, config) -> None:
        sae_layer = config.vector_layer + 1

        self.autoencoder = SparseAutoencoder.from_pretrained(
            config.autoencoder, f"blocks.{sae_layer}.hook_resid_pre"
        )
        self.autoencoder.to(config.device)
        for p in self.autoencoder.parameters():
            p.requires_grad = False
        self.autoencoder.eval()

        self.model = LanguageModel(
            "openai-community/gpt2", device_map="cpu", dispatch=True
        )
        self.model.to(config.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.baseline_vector = SteeringVector.load(
            Path(config.baseline_vector_dir) / f"layer_{config.vector_layer}.pt",
            device=config.device,
        )  # type: ignore
        v = self.baseline_vector.vector
        feature_acts = self.autoencoder(v.unsqueeze(0)).feature_acts.squeeze()

        self.config = config

        self.multipliers = t.nn.Parameter(t.zeros_like(feature_acts))
        self.optimizer = t.optim.Adam([self.multipliers], lr=config.lr)

    @property
    def vector(self):
        return SteeringVector(
            self.config.vector_layer,
            self.multipliers @ self.autoencoder.W_dec + self.baseline_vector.vector,
        )

    def get_all_metrics(self, vector, pos_loader, neg_loader, eval_c4_loader, progress):
        add_pos_loss, add_neg_loss, add_acc = self.eval(
            vector, pos_loader, neg_loader, progress=progress
        )
        sub_poss_loss, sub_neg_loss, sub_acc = self.eval(
            -1 * vector, pos_loader, neg_loader, progress=progress
        )
        c4_loss = self.compute_loss(vector, eval_c4_loader, progress=progress)
        p_success = p_successful_rollout(
            self.model,
            "My favorite doll is a little",
            vector,
            gender_keywords,
        )[1]

        return {
            "c4_loss": c4_loss,
            "p_success": p_success,
            "add_pos_loss": add_pos_loss,
            "add_neg_loss": add_neg_loss,
            "add_acc": add_acc,
            "sub_pos_loss": sub_poss_loss,
            "sub_neg_loss": sub_neg_loss,
            "sub_acc": sub_acc,
        }

    def get_c4_loader(self, split):
        data = load_dataset("NeelNanda/c4-10k", split=split)
        tokenized = tokenize_and_concatenate(
            data,  # type: ignore
            self.model.tokenizer,  # type: ignore
            max_length=128,  # type: ignore
        )
        loader = DataLoader(
            tokenized.with_format("torch"),  # type: ignore
            collate_fn=collate_fn_c4,
            batch_size=32,
        )
        return loader

    def get_wanb_dict(self, base, our):
        return {
            "val/∆(add_acc)": our["add_acc"] - base["add_acc"],
            "val/∆(sub_acc)": base["sub_acc"] - our["sub_acc"],
            "val/∆(add_pos_loss)": base["add_pos_loss"] - our["add_pos_loss"],
            "val/∆(add_neg_loss)": our["add_neg_loss"] - base["add_neg_loss"],
            "val/∆(sub_pos_loss)": our["sub_pos_loss"] - base["sub_pos_loss"],
            "val/∆(sub_neg_loss)": base["sub_neg_loss"] - our["sub_neg_loss"],
            "val/∆(c4_loss)": base["c4_loss"] - our["c4_loss"],
            "val/∆(p_success)": our["p_success"] - base["p_success"],
        }

    def train(self, pos_loader, neg_loader, pos_eval_loader, neg_eval_loader):
        train_c4_loader = self.get_c4_loader("train[:1%]")
        eval_c4_loader = self.get_c4_loader("train[1%:2%]")

        strong_baseline = 3 * self.baseline_vector

        with Progress() as progress:
            evaluate = partial(
                self.get_all_metrics,
                pos_loader=pos_eval_loader,
                neg_loader=neg_eval_loader,
                eval_c4_loader=eval_c4_loader,
                progress=progress,
            )

            base = evaluate(strong_baseline)
            rprint(f"Baseline: {base}")

            our = evaluate(self.vector)
            metrics = self.get_wanb_dict(base, our)
            wandb.log(metrics | {"n_seen": 0})

            n_seen = 0
            for epoch in range(self.config.epochs):
                train_task = progress.add_task(
                    f"Epoch {epoch + 1}...", total=len(pos_loader.dataset)
                )
                batches = zip(pos_loader, neg_loader, train_c4_loader)
                for pos_batch, neg_batch, neutral_batch in batches:
                    batch_size = len(pos_batch["input_ids"])
                    self.optimizer.zero_grad()

                    v = self.vector

                    add_pos_loss = llm_loss(self.model, pos_batch, v).sum()
                    add_neg_loss = llm_loss(self.model, neg_batch, v).sum()

                    sub_pos_loss = llm_loss(self.model, pos_batch, -1 * v).sum()
                    sub_neg_loss = llm_loss(self.model, neg_batch, -1 * v).sum()

                    c4_loss = llm_loss(self.model, neutral_batch, v).sum()

                    penalty_l1 = self.multipliers.abs().sum()
                    penalty_l2 = self.multipliers.pow(2).sum()

                    loss = (
                        self.config.alpha_p * add_pos_loss
                        - self.config.alpha_n * add_neg_loss
                        + self.config.sigma_p * sub_neg_loss
                        - self.config.sigma_n * sub_pos_loss
                        + self.config.gamma * c4_loss
                        + self.config.l1 * penalty_l1
                        + self.config.l2 * penalty_l2
                    )

                    loss.backward()
                    self.optimizer.step()

                    n_seen += len(pos_batch["input_ids"]) + len(neg_batch["input_ids"])
                    progress.update(train_task, advance=self.config.batch_size)

                    wandb.log({"train/loss": loss / batch_size, "n_seen": n_seen})
                progress.stop_task(train_task)

                our = evaluate(self.vector)
                metrics = self.get_wanb_dict(base, our)
                wandb.log(metrics | {"n_seen": n_seen, "epoch": epoch + 1})

    @t.no_grad()
    def eval(self, vector, pos_loader, neg_loader, progress=None):
        assert len(pos_loader.dataset) == len(neg_loader.dataset)
        n, correct, pos_loss, neg_loss = len(pos_loader.dataset), 0, 0, 0

        if progress is not None:
            task = progress.add_task("Evaluating...", total=n)

        for pos_batch, neg_batch in zip(pos_loader, neg_loader):
            pos_losses = llm_loss(self.model, pos_batch, vector)
            neg_losses = llm_loss(self.model, neg_batch, vector)
            correct += (pos_losses < neg_losses).count_nonzero().item()
            pos_loss += pos_losses.sum().item()
            neg_loss += neg_losses.sum().item()
            if progress is not None:
                progress.update(task, advance=len(pos_losses) + len(neg_losses))

        if progress is not None:
            progress.stop_task(task)

        avg_pos_loss = pos_loss / n
        avg_neg_loss = neg_loss / n
        avg_loss = correct / pos_loss

        return avg_pos_loss, avg_neg_loss, avg_loss

    def compute_loss(self, vector, loader, progress=None):
        n, loss = len(loader.dataset), 0
        if progress is not None:
            task = progress.add_task("Evaluating...", total=n)

        for batch in loader:
            loss += llm_loss(self.model, batch, vector).sum().item()
            if progress is not None:
                progress.update(task, advance=len(batch["input_ids"]))

        if progress is not None:
            progress.stop_task(task)

        avg_loss = loss / n

        return avg_loss


def collate_fn(examples, tokenizer):
    texts = [x["prompt"] + " " + x["completion"] for x in examples]
    pos_mask = t.Tensor([x["origin"] == "pos" for x in examples]).bool()

    output = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

    completion_lens = t.tensor(
        [len(s) for s in tokenizer([x["completion"] for x in examples]).input_ids]
    )
    total_len = output.input_ids.size(1)
    indices = t.arange(total_len)
    prompt_mask = indices[None, :] < total_len - completion_lens[:, None]

    return output | {"prompt_mask": prompt_mask, "pos_mask": pos_mask}


def get_loaders(trainer):
    _, dataset_dir = utils.get_version(
        Path("../data") / trainer.config.dataset / "dataset"
    )
    pos_train, neg_train = Dataset.load(dataset_dir / "generate.json").as_single_items()
    pos_eval, neg_eval = Dataset.load(dataset_dir / "test.json").as_single_items()

    collate = partial(collate_fn, tokenizer=trainer.model.tokenizer)
    batch_size = trainer.config.batch_size

    pos_loader = DataLoader(
        pos_train, collate_fn=collate, batch_size=batch_size, shuffle=True
    )
    neg_loader = DataLoader(
        neg_train, collate_fn=collate, batch_size=batch_size, shuffle=True
    )
    pos_eval_loader = DataLoader(pos_eval, collate_fn=collate, batch_size=batch_size)
    neg_eval_loader = DataLoader(neg_eval, collate_fn=collate, batch_size=batch_size)

    return pos_loader, neg_loader, pos_eval_loader, neg_eval_loader


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
def run_training():
    wandb.init()

    config = wandb.config
    set_seed(config.seed)

    trainer = Trainer(config)
    pos_train, neg_train, pos_eval, neg_eval = get_loaders(trainer)
    trainer.train(pos_train, neg_train, pos_eval, neg_eval)

    wandb.finish()


# %%
sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val/∆(c4_loss)"},
    "parameters": {
        "lr": {"values": [1e-1, 1e-2, 1e-3, 1e-4]},
        "l1": {"min": 0.0, "max": 1.0},
        "l2": {"min": 0.0, "max": 1.0},
        "alpha_p": {"min": 0.0, "max": 1.0},
        "alpha_n": {"min": 0.0, "max": 1.0},
        "sigma_p": {"min": 0.0, "max": 1.0},
        "sigma_n": {"min": 0.0, "max": 1.0},
        "gamma": {"min": 0.0, "max": 1.0},
        "dataset": {"value": "gender"},
        "epochs": {"value": 20},
        "batch_size": {"value": 64},
        "vector_layer": {"value": 8},
        "autoencoder": {"value": "gpt2-small-res-jb"},
        "model": {"value": "openai-community/gpt2"},
        "device": {"value": "cuda"},
        "baseline_vector_dir": {"value": "../data/gender/vectors/v1"},
        "seed": {"value": 42},
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="eth-supervised-research")
wandb.agent(sweep_id, function=run_training)


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
