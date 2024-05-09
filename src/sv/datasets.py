import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import datasets.arrow_dataset as ad
import torch as t
import torch.utils.data as td
from datasets import load_dataset
from transformer_lens.utils import tokenize_and_concatenate
from transformers import PreTrainedTokenizer


class C4DataModule:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prefix_len: int = 64,
        max_len: int = 128,
        percent_size: int = 1,
        batch_size: int = 32,
    ):
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._prefix_len = prefix_len
        self._percent_size = percent_size
        self._batch_size = batch_size

    def setup(self):
        train_split = f"train[:{self._percent_size}%]"
        self._train_data = load_dataset("NeelNanda/c4-10k", split=train_split)
        eval_split = f"train[{self._percent_size}%:{self._percent_size + 1}%]"
        self._eval_data = load_dataset("NeelNanda/c4-10k", split=eval_split)

    def _collate_fn(self, examples):
        input_ids = t.stack([x["tokens"] for x in examples])
        prompt_mask = t.zeros_like(input_ids)
        prompt_mask[:, : self._prefix_len] = 1
        pos_mask = t.ones(len(examples)).bool()

        return {
            "input_ids": input_ids,
            "prompt_mask": prompt_mask,
            "pos_mask": pos_mask,
        }

    def _dataloader(self, data: ad.Dataset):
        tokenized = tokenize_and_concatenate(data, self._tokenizer, max_length=128)  # type:ignore
        loader = td.DataLoader(
            tokenized.with_format("torch"),  # type: ignore
            collate_fn=self._collate_fn,
            batch_size=self._batch_size,
        )
        return loader

    def train_dataloader(self):
        assert isinstance(self._train_data, ad.Dataset)
        return self._dataloader(self._train_data)

    def eval_dataloader(self):
        assert isinstance(self._eval_data, ad.Dataset)
        return self._dataloader(self._eval_data)


class PairDataModule:
    def __init__(
        self, dataset_dir: Path, tokenizer: PreTrainedTokenizer, batch_size: int = 64
    ):
        self._dataset_dir = dataset_dir
        self._tokenizer = tokenizer
        self._batch_size = batch_size

    def setup(self):
        self._pos_train_data, self._neg_train_data = Dataset.load(
            self._dataset_dir / "generate.json"
        ).as_single_items()
        self._pos_eval_data, self._neg_eval_data = Dataset.load(
            self._dataset_dir / "test.json"
        ).as_single_items()

    def _collate_fn(self, examples):
        texts = [x["prompt"] + " " + x["completion"] for x in examples]
        pos_mask = t.Tensor([x["origin"] == "pos" for x in examples]).bool()

        output = self._tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )

        completion_tokens = self._tokenizer(
            [x["completion"] for x in examples]
        ).input_ids
        completion_lens = t.tensor([len(s) for s in completion_tokens])
        total_len = output.input_ids.size(1)
        indices = t.arange(total_len)
        prompt_mask = indices[None, :] < total_len - completion_lens[:, None]

        return output | {"prompt_mask": prompt_mask, "pos_mask": pos_mask}

    def _loader(self, dataset):
        loader = td.DataLoader(
            dataset, collate_fn=self._collate_fn, batch_size=self._batch_size
        )
        return loader

    def train_loaders(self):
        pos_loader = self._loader(self._pos_train_data)
        neg_loader = self._loader(self._neg_train_data)

        return pos_loader, neg_loader

    def eval_loaders(self):
        pos_loader = self._loader(self._pos_eval_data)
        neg_loader = self._loader(self._neg_eval_data)

        return pos_loader, neg_loader


class DictDataset(td.Dataset):
    _prompts: List[dict]

    def __init__(self, prompts: List[dict]):
        self._prompts = prompts

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, idx: int) -> dict:
        return self._prompts[idx]

    @staticmethod
    def from_openwebtext(
        limit: int, prompt_len: int, tokenizer: PreTrainedTokenizer, max_len: int = 128
    ) -> "DictDataset":
        dataset = load_dataset(
            "Skylion007/openwebtext", streaming=True, trust_remote_code=True
        )["train"]  # type: ignore
        items = []
        for i, item in enumerate(dataset):
            assert isinstance(item, dict)
            if i >= limit:
                break
            tokens = tokenizer(
                item["text"], return_tensors="pt", truncation=True
            ).input_ids
            prompt = tokenizer.batch_decode(tokens[:, :prompt_len])[0].strip()
            completion = tokenizer.batch_decode(tokens[:, prompt_len:max_len])[
                0
            ].strip()
            items.append(
                {
                    "index": i,
                    "prompt": prompt,
                    "completion": completion,
                    "origin": f"openwebtext-{limit}-{prompt_len }",
                }
            )
        return DictDataset(items)


@dataclass
class Item:
    question: str
    pos: str
    neg: str

    def __hash__(self) -> int:
        return hash((self.question, self.pos, self.neg))

    def to_dict(self) -> dict:
        return {"question": self.question, "pos": self.pos, "neg": self.neg}

    @staticmethod
    def from_dict(data: dict) -> "Item":
        return Item(question=data["question"], pos=data["pos"], neg=data["neg"])

    @staticmethod
    def from_rimsky_item(rimsky_item: dict, matching=None) -> "Item":
        match = re.match(
            r"(.+?)(?:Choices:)?\s+\(A\)(.+)\s+\(B\)(.+)",
            rimsky_item["question"],
            flags=re.DOTALL,
        )
        if not match:
            raise ValueError(f"Invalid question: {rimsky_item['question']}")

        if matching is None:
            if rimsky_item["answer_matching_behavior"] == "(A)":
                matching = "first"
            else:
                matching = "second"

        question, a, b = [text.strip() for text in match.groups()]
        pos, neg = (a, b) if matching == "first" else (b, a)

        return Item(question, pos, neg)


class Dataset:
    _items: List[Item]

    def __init__(self, items: List[Item]):
        self._items = items

    def __len__(self):
        return len(self._items)

    def as_prefixed_pairs(self) -> DictDataset:
        pairs = [
            {
                "pos": item.question + " " + item.pos,
                "neg": item.question + " " + item.neg,
            }
            for item in self._items
        ]
        dataset = DictDataset(pairs)
        return dataset

    def as_single_items(self) -> tuple[DictDataset, DictDataset]:
        pos_prompts = [
            {
                "index": i,
                "prompt": item.question,
                "completion": item.pos,
                "origin": "pos",
            }
            for i, item in enumerate(self._items)
        ]
        neg_prompts = [
            {
                "index": i,
                "prompt": item.question,
                "completion": item.neg,
                "origin": "neg",
            }
            for i, item in enumerate(self._items)
        ]
        pos_dataset = DictDataset(pos_prompts)
        neg_dataset = DictDataset(neg_prompts)
        return pos_dataset, neg_dataset

    @staticmethod
    def from_rimsky(path: Path) -> "Dataset":
        with open(path) as f:
            rimsky_items = json.load(f)

        items = [Item.from_rimsky_item(i) for i in rimsky_items]
        dataset = Dataset(items)

        return dataset

    @staticmethod
    def from_gpt(path: Path, test_size: int) -> tuple["Dataset", "Dataset"]:
        with open(path) as f:
            rimsky_items = json.load(f)

        items = [Item.from_rimsky_item(i, matching="first") for i in rimsky_items]
        if test_size > len(items):
            raise ValueError(f"Not enough items for chosen test size: {len(items)}")
        if len(set(items)) < len(items):
            raise ValueError("Items are not unique")

        random.shuffle(items)
        train = Dataset(items[:-test_size])
        test = Dataset(items[-test_size:])

        return train, test

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({"items": [item.to_dict() for item in self._items]}, f)

    @staticmethod
    def load(path: Path) -> "Dataset":
        with open(path) as f:
            data = json.load(f)
            return Dataset([Item.from_dict(item) for item in data["items"]])
