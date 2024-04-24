import json
import random
import re
from pathlib import Path
from typing import Iterable, List, TypedDict

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RimskyItem(TypedDict):
    question: str
    answer_matching_behavior: str
    answer_not_matching_behavior: str


class Question(TypedDict):
    prompt: str
    answer_token: int
    complete: str


class PosNegPair(TypedDict):
    id: int
    pos: Question
    neg: Question


class ContrastiveDataset(Dataset):
    _data: List[PosNegPair]

    def __init__(self, data: List[PosNegPair]):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> PosNegPair:
        return self._data[idx]

    @staticmethod
    def _parse_rimsky_item(item: RimskyItem):
        match = re.match(
            r"(.+?)(?:Choices:)?\s+\(A\)(.+)\s+\(B\)(.+)",
            item["question"],
            flags=re.DOTALL,
        )
        if not match:
            raise ValueError(f"Invalid question: {item['question']}")

        question, a, b = [text.strip() for text in match.groups()]
        a, b = f" ^ {a}", f" $ {b}"
        pos, neg = (a, b) if item["answer_matching_behavior"] == "(A)" else (b, a)

        return question, pos, neg

    @staticmethod
    def _rimsky_prompt(question: str, first: str, second: str):
        return "\n\n".join(
            [
                f"Question: {question}",
                f"Choices:\nA:{first}\nA:{second}",
                "Answer:\nA:",
            ]
        )

    @staticmethod
    def _prompt(question: str, first: str, second: str):
        return "\n\n".join(
            [
                f"Q: {question}\nA:{first}",
                f"Q: {question}\nA:{second}",
                f"Q: {question}\nA:",
            ]
        )

    @staticmethod
    def _build_dataset(
        questions_with_answers: Iterable[tuple[str, str, str]],
        answer_most_recent: bool,
        tokenizer: PreTrainedTokenizer,
        rimsky_format=False,
    ):
        items = []
        for id, (question, pos, neg) in enumerate(questions_with_answers):
            pair = {"id": id}
            for first, second, label in [(pos, neg, "neg"), (neg, pos, "pos")]:
                answer_token = tokenizer(second).input_ids[0]
                if not answer_most_recent:
                    first, second = second, first
                if rimsky_format:
                    prompt = ContrastiveDataset._rimsky_prompt(question, second, first)
                else:
                    prompt = ContrastiveDataset._prompt(question, first, second)
                pair[label] = {  # type: ignore
                    "prompt": prompt,
                    "answer_token": answer_token,
                    "complete": prompt + tokenizer.decode([answer_token]),
                }
            items.append(pair)
        return items

    @staticmethod
    def from_rimsky(
        path: Path,
        tokenizer: PreTrainedTokenizer,
        answer_most_recent=True,
        rimsky_format=False,
    ) -> "ContrastiveDataset":
        with open(path) as f:
            data = json.load(f)

        questions_with_answers = (
            ContrastiveDataset._parse_rimsky_item(i) for i in data
        )
        items = ContrastiveDataset._build_dataset(
            questions_with_answers, answer_most_recent, tokenizer, rimsky_format
        )
        dataset = ContrastiveDataset(items)

        return dataset

    @staticmethod
    def from_random_mix(
        paths: List[Path],
        tokenizer: PreTrainedTokenizer,
        generate_size: int,
        test_size: int,
        answer_most_recent=True,
    ) -> tuple["ContrastiveDataset", "ContrastiveDataset"]:
        data = []
        for path in paths:
            with open(path) as f:
                data += json.load(f)

        questions, answers = [], []
        for question, pos, neg in [
            ContrastiveDataset._parse_rimsky_item(i) for i in data
        ]:
            questions.append(question)
            answers += [pos, neg]
        random.shuffle(questions)
        random.shuffle(answers)

        if generate_size + test_size > len(questions):
            raise ValueError(
                f"Not enough questions to generate the requested sizes: {len(questions)}"
            )

        datasets = []
        for start, size in [(0, generate_size), (generate_size, test_size)]:
            questions_with_answers = zip(
                questions[start : start + size],
                answers[start::2],
                answers[start + 1 :: 2],
            )
            items = ContrastiveDataset._build_dataset(
                questions_with_answers, answer_most_recent, tokenizer
            )
            datasets.append(ContrastiveDataset(items))

        return datasets[0], datasets[1]

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self._data, f)

    @staticmethod
    def load(path: Path):
        with open(path) as f:
            return ContrastiveDataset(json.load(f))
