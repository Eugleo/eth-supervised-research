from functools import partial
from pathlib import Path

import torch as t
from nnsight import LanguageModel
from rich import print as rprint
from rich.progress import Progress
from sae_lens import SparseAutoencoder

import wandb
from sv.datasets import C4DataModule, PairDataModule
from sv.scoring import activations, llm_loss, p_successful_rollout
from sv.vectors import SteeringVector

GENDER_KEYWORDS = [
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


def _prep_model(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


class Trainer:
    def __init__(self, cfg) -> None:
        sae_layer = cfg.vector_layer + 1
        self.autoencoder = SparseAutoencoder.from_pretrained(
            cfg.autoencoder, f"blocks.{sae_layer}.hook_resid_pre"
        ).to(cfg.device)
        _prep_model(self.autoencoder)

        self.model = LanguageModel(
            "openai-community/gpt2", device_map="cpu", dispatch=True
        ).to(cfg.device)
        _prep_model(self.model)

        basline_path = Path(cfg.baseline_vector_dir) / f"layer_{cfg.vector_layer}.pt"
        self.baseline_vector = SteeringVector.load(basline_path, device=cfg.device)  # type: ignore

        init = (3 * self.baseline_vector.vector).detach().clone()
        self.vector_data = t.nn.Parameter(init)
        self.optimizer = t.optim.Adam([self.vector_data], lr=cfg.lr)

        self.config = cfg

    @property
    def vector(self):
        return SteeringVector(self.config.vector_layer, self.vector_data)

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
            GENDER_KEYWORDS,
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

    def get_wanb_dict(self, base, our):
        return {
            "val/add_pos_loss": our["add_pos_loss"],
            "val/add_neg_loss": our["add_neg_loss"],
            "val/sub_pos_loss": our["sub_pos_loss"],
            "val/sub_neg_loss": our["sub_neg_loss"],
            "val/∆(add_acc)": our["add_acc"] - base["add_acc"],
            "val/∆(sub_acc)": base["sub_acc"] - our["sub_acc"],
            "val/∆(c4_loss)": base["c4_loss"] - our["c4_loss"],
            "val/∆(p_success)": our["p_success"] - base["p_success"],
            "val/goodness": min(
                base["c4_loss"] - our["c4_loss"], our["p_success"] - base["p_success"]
            ),
        }

    def train(self, pair_module: PairDataModule, neutral_module: C4DataModule):
        strong_baseline = 3 * self.baseline_vector

        pair_module.setup()
        pos_loader, neg_loader = pair_module.train_loaders()
        pos_eval_loader, neg_eval_loader = pair_module.eval_loaders()

        neutral_module.setup()
        train_c4_loader = neutral_module.train_dataloader()
        eval_c4_loader = neutral_module.eval_dataloader()

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
                    f"Epoch {epoch + 1}...",
                    total=len(pos_loader.dataset),  # type: ignore
                )
                batches = zip(pos_loader, neg_loader, train_c4_loader)
                for pos_batch, neg_batch, neutral_batch in batches:
                    batch_size = len(pos_batch["input_ids"])
                    self.optimizer.zero_grad()

                    m, v, cfg = self.model, self.vector, self.config

                    add_neg = activations(m, neg_batch, cfg.act_layer, v)
                    null_pos = activations(m, pos_batch, cfg.act_layer)

                    sub_pos = activations(m, pos_batch, cfg.act_layer, -v)
                    null_neg = activations(m, neg_batch, cfg.act_layer)

                    add_neut = activations(m, neutral_batch, cfg.act_layer, v)
                    sub_neut = activations(m, neutral_batch, cfg.act_layer, -v)
                    null_neut = activations(m, neutral_batch, cfg.act_layer)

                    loss = (
                        cfg.alpha * (add_neg - null_pos).pow(2).sum()
                        + cfg.alpha * (sub_pos - null_neg).pow(2).sum()
                        + cfg.omega * (add_neut - null_neut).pow(2).sum()
                        + cfg.omega * (sub_neut - null_neut).pow(2).sum()
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
