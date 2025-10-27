# src/loaders.py
from datasets import load_dataset
from typing import Iterable, Dict, Optional


def load_gsm8k(split="test", limit=None) -> Iterable[Dict]:
    ds = load_dataset("gsm8k", "main")[split]
    for i, ex in enumerate(ds):
        if limit and i >= limit and limit > 0:
            break
        yield {
            "id": ex.get("question", "")[:64] + f"::{i}",
            "question": ex["question"],
            "answer": ex["answer"],
            "task": "math",
        }


def load_aime24(split: str = "test", limit: Optional[int] = None) -> Iterable[Dict]:
    """
    Load AIME 2024 from HuggingFaceH4/aime_2024 and yield {id, question, answer, task}.
    Attempts to map various possible field names for question/answer.
    """
    ds_dict = load_dataset("HuggingFaceH4/aime_2024")
    if split not in ds_dict:
        # Fall back to the first available split
        split = list(ds_dict.keys())[0]
    ds = ds_dict[split]

    # Helpers to extract fields with fallbacks
    def get_question(ex: Dict) -> str:
        for k in ("question", "problem", "prompt", "input", "query"):
            if k in ex and isinstance(ex[k], str) and ex[k].strip():
                return ex[k]
        # As a last resort, join all string fields (unlikely needed)
        return " ".join(str(v) for v in ex.values() if isinstance(v, str)).strip()

    def get_answer(ex: Dict) -> str:
        # Prefer a clean final answer if present
        for k in ("final_answer", "answer", "solution", "target", "output"):
            if k in ex and isinstance(ex[k], str) and ex[k].strip():
                return ex[k]
        # Try numeric fields
        for k in ex.keys():
            if "answer" in k.lower() and ex[k] is not None:
                return str(ex[k])
        return ""

    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        q = get_question(ex)
        a = get_answer(ex)
        yield {
            "id": (q or "")[:64] + f"::{i}",
            "question": q,
            "answer": a,
            "task": "aime",
        }


def load_aime25(split: str = "test", limit: Optional[int] = None) -> Iterable[Dict]:
    """
    Load AIME 2025. Adjust the dataset path/name as needed.
    This is a placeholder - update with the actual dataset name when available.
    """
    try:
        # Try common dataset names
        possible_names = [
            "yentinglin/aime_2025",
        ]
        
        ds_dict = None
        for name in possible_names:
            try:
                ds_dict = load_dataset(name, "default")
                break
            except:
                continue
        
        if ds_dict is None:
            print(f"Warning: Could not load AIME 2025 dataset. Tried: {possible_names}")
            return
            
        if split not in ds_dict:
            split = list(ds_dict.keys())[0]
        ds = ds_dict[split]
        
        def get_question(ex: Dict) -> str:
            for k in ("question", "problem", "prompt", "input", "query"):
                if k in ex and isinstance(ex[k], str) and ex[k].strip():
                    return ex[k]
            return " ".join(str(v) for v in ex.values() if isinstance(v, str)).strip()
        
        def get_answer(ex: Dict) -> str:
            for k in ("final_answer", "answer", "solution", "target", "output"):
                if k in ex and isinstance(ex[k], str) and ex[k].strip():
                    return ex[k]
            for k in ex.keys():
                if "answer" in k.lower() and ex[k] is not None:
                    return str(ex[k])
            return ""
        
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            q = get_question(ex)
            a = get_answer(ex)
            yield {
                "id": (q or "")[:64] + f"::{i}",
                "question": q,
                "answer": a,
                "task": "aime",
            }
    except Exception as e:
        print(f"Error loading AIME 2025: {e}")
        return


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
    name_lower = name.lower()
    if name_lower == "gsm8k":
        return load_gsm8k
    elif name_lower == "strategyqa":
        return load_strategyqa
    elif name_lower in ["aime24", "aime_24", "aime2024"]:
        return load_aime24
    elif name_lower in ["aime25", "aime_25", "aime2025"]:
        return load_aime25
    else:
        raise ValueError(f"Unknown dataset: {name}")