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
        pos, neg = (a, b) if item["answer_matching_behavior"] == "(A)" else (b, a)

        return question, pos, neg

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
    def _build_dataset(
        questions_with_answers: Iterable[tuple[str, str, str]],
        tokenizer: PreTrainedTokenizer,
    ):
        A_tok, B_tok = tokenizer(" A").input_ids[0], tokenizer(" B").input_ids[0]
        items = []
        for id, (question, pos, neg) in enumerate(questions_with_answers):
            for answer_is_first_option in [True, False]:
                pair = {"id": id, "answer_is_first_option": answer_is_first_option}
                for label, answer, other in [("pos", pos, neg), ("neg", neg, pos)]:
                    first, second = (
                        (answer, other) if answer_is_first_option else (other, answer)
                    )
                    answer_token = A_tok if answer_is_first_option else B_tok
                    prompt = ContrastiveDataset._rimsky_prompt(question, first, second)
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
    ) -> "ContrastiveDataset":
        with open(path) as f:
            data = json.load(f)

        questions_with_answers = (
            ContrastiveDataset._parse_rimsky_item(i) for i in data
        )
        items = ContrastiveDataset._build_dataset(questions_with_answers, tokenizer)
        dataset = ContrastiveDataset(items)

        return dataset

    @staticmethod
    def from_random_mix(
        paths: List[Path],
        tokenizer: PreTrainedTokenizer,
        generate_size: int,
        test_size: int,
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
            items = ContrastiveDataset._build_dataset(questions_with_answers, tokenizer)
            datasets.append(ContrastiveDataset(items))

        return datasets[0], datasets[1]

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self._data, f)

    @staticmethod
    def load(path: Path):
        with open(path) as f:
            return ContrastiveDataset(json.load(f))
