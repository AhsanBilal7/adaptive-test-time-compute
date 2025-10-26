# src/metrics.py
from typing import List


def accuracy(predictions: List[str], gold: List[str]) -> float:
    if not predictions or not gold or len(predictions) != len(gold):
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    return correct / len(predictions)