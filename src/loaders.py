# src/loaders.py
from datasets import load_dataset
from typing import Iterable, Dict


def load_gsm8k(split="test", limit=None) -> Iterable[Dict]:
    ds = load_dataset("gsm8k", "main")[split]
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        yield {
            "id": ex.get("question", "")[:64] + f"::{i}",
            "question": ex["question"],
            "answer": ex["answer"],
            "task": "math",
        }


def load_strategyqa(split="validation", limit=None) -> Iterable[Dict]:
    ds = load_dataset("strategyqa")["validation"]
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        yield {
            "id": ex.get("question", "")[:64] + f"::{i}",
            "question": ex["question"],
            "answer": "yes" if ex["answer"] else "no",
            "task": "qa",
        }


def get_loader(name: str):
    if name.lower() == "gsm8k":
        return load_gsm8k
    elif name.lower() == "strategyqa":
        return load_strategyqa
    else:
        raise ValueError(f"Unknown dataset: {name}")