# src/io_utils.py
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_jsonl(items: Iterable[Dict[str, Any]], out_path: str | Path):
    out = ensure_dir(Path(out_path).parent)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def set_seed(seed: int):
    random.seed(seed)