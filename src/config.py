# src/config.py
import os
import yaml
from typing import Any, Dict


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def env_default(model_name: str) -> str:
    if "${OLLAMA_MODEL}" in model_name:
        return os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
    return model_name


def apply_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg["model"]["name"] = env_default(cfg["model"]["name"])
    return cfg