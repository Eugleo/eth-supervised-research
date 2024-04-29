import json
import random
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Literal, Tuple, TypedDict

import jsonpickle
import torch.utils.data as td
from transformers import PreTrainedTokenizer


class ContrastivePair(TypedDict):
    pos: str
    neg: str


class GenDataset(td.Dataset):
    _pairs: List[ContrastivePair]

    def __init__(self, pairs: List[ContrastivePair]):
        self._pairs = pairs

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx: int) -> ContrastivePair:
        return self._pairs[idx]


class Prompt(TypedDict):
    q_num: int
    prompt: str
    ordering: Literal["pos,neg", "neg,pos"]
    pos_token: int
    neg_token: int


class EvalDataset(td.Dataset):
    _prompts: List[Prompt]

    def __init__(self, prompts: List[Prompt]):
        self._prompts = prompts

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, idx: int) -> Prompt:
        return self._prompts[idx]


class RimskyItem(TypedDict):
    question: str
    answer_matching_behavior: str
    answer_not_matching_behavior: str


class Format(str, Enum):
    rimsky = "rimsky"
    repeated = "repeated"
    combined = "combined"


@dataclass
class Item:
    question: str
    pos: str
    neg: str

    def ordered(self, first: str, second: str) -> tuple[str, str, str]:
        return self.question, getattr(self, first), getattr(self, second)

    def __hash__(self) -> int:
        return hash((self.question, self.pos, self.neg))

    @staticmethod
    def from_rimsky_item(rimsky_item: RimskyItem, matching=None) -> "Item":
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

    def get_tokens(
        self, ordering: tuple[str, str], format: Format, tokenizer: PreTrainedTokenizer
    ) -> Tuple[int, str, int, str]:
        first, second = ordering
        if format == Format.rimsky or format == Format.combined:
            first_token: int = tokenizer(" A").input_ids[0]
            second_token: int = tokenizer(" B").input_ids[0]
        elif format == Format.repeated:
            first_token: int = tokenizer(f" {getattr(self, first)}").input_ids[0]
            second_token: int = tokenizer(f" {getattr(self, second)}").input_ids[0]

        pos_token, neg_token = (
            (first_token, second_token)
            if first == "pos"
            else (second_token, first_token)
        )
        pos_token_str, neg_token_str = (
            tokenizer.batch_decode([pos_token])[0],
            tokenizer.batch_decode([neg_token])[0],
        )

        return pos_token, pos_token_str, neg_token, neg_token_str


class Dataset:
    _items: List[Item]
    _format: Format

    def __init__(self, items: List[Item], format: Format):
        self._items = items
        self._format = format
        if format == Format.repeated:
            self.make_prompt = Dataset._repeated_prompt
        elif format == Format.rimsky:
            self.make_prompt = Dataset._rimsky_prompt
        elif format == Format.combined:
            self.make_prompt = Dataset._combined_prompt
        else:
            raise ValueError(f"Invalid format: {format}")

    def __len__(self):
        return len(self._items)

    def for_generating(self, tokenizer: PreTrainedTokenizer, kind: str) -> GenDataset:
        pairs: List[ContrastivePair] = []
        for item in self._items:
            if kind == "rimsky":
                for first, second in [("pos", "neg"), ("neg", "pos")]:
                    _, pos_token_str, _, _ = item.get_tokens(
                        (first, second), self._format, tokenizer
                    )
                    pos_prompt = self.make_prompt(*item.ordered(first, second))
                    _, _, _, neg_token_str = item.get_tokens(
                        (second, first), self._format, tokenizer
                    )
                    neg_prompt = self.make_prompt(*item.ordered(second, first))
                    pairs.append(
                        {
                            "pos": pos_prompt + pos_token_str,
                            "neg": neg_prompt + neg_token_str,
                        }
                    )
            elif kind == "turner":
                pairs.append(
                    {
                        "pos": item.question + " " + item.pos,
                        "neg": item.question + " " + item.neg,
                    }
                )

        dataset = GenDataset(pairs)
        return dataset

    def for_evaluating(self, tokenizer: PreTrainedTokenizer, kind: str) -> EvalDataset:
        prompts: List[Prompt] = []
        for i, item in enumerate(self._items):
            if kind == "rimsky":
                for first, second in [("pos", "neg"), ("neg", "pos")]:
                    prompt = self.make_prompt(*item.ordered(first, second))
                    pos_token, _, neg_token, _ = item.get_tokens(
                        (first, second), self._format, tokenizer
                    )
                    ordering = "pos,neg" if first == "pos" else "neg,pos"
                    prompts.append(
                        {
                            "q_num": i,
                            "prompt": prompt,
                            "ordering": ordering,
                            "pos_token": pos_token,
                            "neg_token": neg_token,
                        }
                    )
            elif kind == "turner":
                prompts.append(
                    {
                        "q_num": i,
                        "prompt": item.question,
                        "pos": item.pos,
                        "neg": item.neg,
                    }
                )

        dataset = EvalDataset(prompts)
        return dataset

    @staticmethod
    def _rimsky_prompt(question: str, first: str, second: str):
        return "\n\n".join(
            [
                f"{question}",
                f"Option A: {first}",
                f"Option B: {second}",
                "Answer: Option",
            ]
        )

    @staticmethod
    def _combined_prompt(question: str, first: str, second: str):
        return "\n\n".join(
            [
                f"Q: {question}\nA: Option A: {first}",
                f"Q: {question}\nA: Option B: {second}",
                f"Q: {question}\nA: I choose option",
            ]
        )

    @staticmethod
    def _repeated_prompt(question: str, first: str, second: str):
        return "\n\n".join(
            [
                f"Q: {question}\nA: {first}",
                f"Q: {question}\nA: {second}",
                f"Q: {question}\nA:",
            ]
        )

    @staticmethod
    def from_original(path: Path, format: Format) -> "Dataset":
        with open(path) as f:
            rimsky_items = json.load(f)

        items = [Item.from_rimsky_item(i) for i in rimsky_items]
        dataset = Dataset(items, format=format)

        return dataset

    @staticmethod
    def from_gpt_generated(
        path: Path, test_size: int, format: Format
    ) -> tuple["Dataset", "Dataset"]:
        with open(path) as f:
            rimsky_items = json.load(f)

        items = [Item.from_rimsky_item(i, matching="first") for i in rimsky_items]
        if test_size > len(items):
            raise ValueError(f"Not enough items for chosen test size: {len(items)}")
        if len(set(items)) < len(items):
            raise ValueError("Items are not unique")

        random.shuffle(items)
        train = Dataset(items[:-test_size], format=format)
        test = Dataset(items[-test_size:], format=format)

        return train, test

    @staticmethod
    def from_randomized_original(path: Path, format: Format) -> "Dataset":
        with open(path) as f:
            rimsky_items = json.load(f)

        questions, pos, neg = [], [], []
        for item in [Item.from_rimsky_item(i) for i in rimsky_items]:
            questions.append(item.question)
            pos.append(item.pos)
            neg.append(item.neg)
        random.shuffle(questions)
        random.shuffle(pos)
        random.shuffle(neg)

        items: List[Item] = [
            Item(question, pos, neg) for question, pos, neg in zip(questions, pos, neg)
        ]
        dataset = Dataset(items, format=format)

        return dataset

    @staticmethod
    def from_random_mix(
        paths: List[Path], generate_size: int, test_size: int, format: Format
    ) -> tuple["Dataset", "Dataset"]:
        data = []
        for path in paths:
            with open(path) as f:
                data += json.load(f)

        questions, answers = [], []
        for item in [Item.from_rimsky_item(i) for i in data]:
            questions.append(item.question)
            answers += [item.pos, item.neg]
        random.shuffle(questions)
        random.shuffle(answers)

        if generate_size + test_size > len(questions):
            raise ValueError(f"Not enough items: {len(questions)}")

        datasets = []
        for start, size in [(0, generate_size), (generate_size, test_size)]:
            items: List[Item] = [
                Item(question, pos, neg)
                for question, pos, neg in zip(
                    questions[start : start + size],
                    answers[start::2],
                    answers[start + 1 :: 2],
                )
            ]
            datasets.append(Dataset(items, format=format))

        return datasets[0], datasets[1]

    def save(self, path: Path):
        with open(path, "w") as f:
            f.write(jsonpickle.encode({"format": self._format, "items": self._items}))  # type: ignore

    @staticmethod
    def load(path: Path) -> "Dataset":
        with open(path) as f:
            data: dict = jsonpickle.decode(f.read())  # type: ignore
            return Dataset(data["items"], format=data["format"])
