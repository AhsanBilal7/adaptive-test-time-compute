# src/main.py
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import load_yaml, apply_env
from src.io_utils import ensure_dir, save_jsonl, set_seed
from src.loaders import get_loader
from src.templates import build_iterative_prompt, build_prompt
from src.ollama_client import OllamaClient
from src.generation import generate_greedy, generate_k_paths
from src.selection import pick_sweet_spot, self_consistency_majority, iterative_reasoning_selection
from src.parse import split_reason_answer, parse_gsm8k_numeric, parse_yesno, extract_final_answer
from src.metrics import accuracy


def run(cfg):
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["io"]["out_dir"])
    
    m = cfg["model"]
    client = OllamaClient(
        model=m["name"],
        temperature=m["temperature"],
        top_p=m["top_p"],
        max_tokens=m["max_tokens"],
        stop=m.get("stop", [])
    )
    
    ds_name = cfg["dataset"]["name"]
    split = cfg["dataset"]["split"]
    limit = cfg["dataset"]["limit"]
    loader = get_loader(ds_name)
    data_iter = list(loader(split=split, limit=limit))
    
    preds_iterative, golds = [], []
    preds_greedy, preds_sc = [], []
    traces = []
    
    K = cfg["reasoning"]["K"]
    low, high = cfg["reasoning"]["sweet_spot"]["low"], cfg["reasoning"]["sweet_spot"]["high"]
    prefer_center = cfg["reasoning"]["sweet_spot"]["prefer_center"]
    max_steps = cfg["reasoning"].get("max_steps", 10)
    
    answers_log = []
    for ex in tqdm(data_iter, desc=f"Running {ds_name}"):
        q = ex["question"]
        gt_raw = ex["answer"]
        kind = "math" if ex["task"] == "math" else "qa"
        
        def normalize_answer(ans_raw: str) -> str:
            if kind == "math":
                return parse_gsm8k_numeric(ans_raw)
            else:
                return parse_yesno(ans_raw)
        
        # Iterative reasoning selection
        final_reasoning, step_trace = iterative_reasoning_selection(
            client=client,
            question=q,
            kind=kind,
            K=K,
            low=low,
            high=high,
            prefer_center=prefer_center,
            max_steps=max_steps
        )
        
        # Extract final answer from the complete reasoning
        final_answer_raw = extract_final_answer(final_reasoning, kind=kind)
        final_answer = normalize_answer(final_answer_raw)
        
        # Baseline: Greedy
        prompt = build_prompt(q, kind=kind)
        greedy_text = generate_greedy(client, prompt)
        g_reason, g_ans_raw = split_reason_answer(greedy_text)
        g_ans = normalize_answer(g_ans_raw)
        
        # Baseline: Self-consistency (optional)
        sc_ans = None
        if cfg["baselines"]["self_consistency"]:
            chains = generate_k_paths(client, prompt, K=K)
            sc_answers = []
            for ch in chains:
                _, ans_raw = split_reason_answer(ch)
                sc_answers.append(normalize_answer(ans_raw))
            sc_ans = self_consistency_majority(sc_answers)
        
        # Gold answer
        if kind == "math":
            gold = parse_gsm8k_numeric(gt_raw)
        else:
            gold = parse_yesno(gt_raw)
        
        # Store predictions
        preds_iterative.append(final_answer)
        preds_greedy.append(g_ans)
        if sc_ans is not None:
            preds_sc.append(sc_ans)
        golds.append(gold)
        
        answers_log.append({
            "id": ex["id"],
            "gt_raw": gt_raw,
            "gold": gold,
            "answer_raw": final_answer_raw,
            "answer": final_answer,
        })

        print(f"Q: {q} : answers_log {answers_log[-1]}")
        
        if cfg["io"]["save_traces"]:
            traces.append({
                "id": ex["id"],
                "question": q,
                "gold": gold,
                "iterative_selection": {
                    "final_answer": final_answer,
                    "final_reasoning": final_reasoning,
                    "steps": step_trace,
                },
                "greedy": {"answer": g_ans, "snippet": g_reason},
                "self_consistency": {"answer": sc_ans} if sc_ans is not None else None
            })
    
    acc_iterative = accuracy(preds_iterative, golds)
    acc_greedy = accuracy(preds_greedy, golds)
    results = {
        "dataset": ds_name,
        "n": len(golds),
        "acc_iterative_selection": acc_iterative,
        "acc_greedy": acc_greedy,
    }
    if preds_sc:
        results["acc_self_consistency"] = accuracy(preds_sc, golds)
    
    print(json.dumps(results, indent=2))
    
    if cfg["io"]["save_jsonl"]:
        save_jsonl(traces, Path(cfg["io"]["out_dir"]) / "traces.jsonl")
        with open(Path(cfg["io"]["out_dir"]) / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        save_jsonl(answers_log, Path(cfg["io"]["out_dir"]) / "answers.jsonl")
        with open(Path(cfg["io"]["out_dir"]) / "answers.json", "w") as f:
            json.dump(answers_log, f, indent=2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--dataset", type=str, default=None, help="override dataset.name")
    return ap.parse_args()


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    cfg = apply_env(cfg)
    ensure_dir(cfg["io"]["out_dir"])
    run(cfg)