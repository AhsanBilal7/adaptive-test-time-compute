# experiment_lzw_strategies.py
"""
Experiment to compare different LZW compression-based selection strategies.
Tests: lowest, highest, median, and sweet-spot selection.
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from src.config import load_yaml, apply_env
from src.io_utils import ensure_dir, save_jsonl, set_seed
from src.loaders import get_loader
from src.templates import build_iterative_prompt, build_final_answer_prompt
from src.ollama_client import OllamaClient
from src.generation import generate_greedy, generate_k_paths
from src.selection import (
    pick_sweet_spot, 
    pick_lowest_compression, 
    pick_highest_compression, 
    pick_median_compression,
    score_chains
)
from src.parse import has_answer_marker, extract_final_answer, parse_gsm8k_numeric, parse_yesno
from src.metrics import accuracy


def iterative_reasoning_with_strategy(
    client,
    question: str,
    kind: str,
    K: int,
    max_steps: int,
    strategy: str,
    low: float = 0.60,
    high: float = 1.20,
    prefer_center: bool = True
) -> tuple[str, list[dict], list[float]]:
    """
    Iteratively build reasoning using specified selection strategy.
    
    Args:
        strategy: One of ['lowest', 'highest', 'median', 'sweet_spot']
    
    Returns:
        final_reasoning: Complete reasoning chain
        step_trace: List of step information
        all_ratios: All LZW ratios seen during generation
    """
    accumulated_reasoning = ""
    step_trace = []
    all_ratios = []
    
    for step_num in range(max_steps):
        prompt = build_iterative_prompt(question, accumulated_reasoning, kind=kind)
        candidates = generate_k_paths(client, prompt, K=K)
        
        contains_answer = [has_answer_marker(c, kind=kind) for c in candidates]
        
        # Select based on strategy
        ratios = score_chains(candidates)
        all_ratios.extend(ratios)
        
        if strategy == 'lowest':
            idx, ratio = pick_lowest_compression(candidates)
        elif strategy == 'highest':
            idx, ratio = pick_highest_compression(candidates)
        elif strategy == 'median':
            idx, ratio = pick_median_compression(candidates)
        elif strategy == 'sweet_spot':
            idx, ratio = pick_sweet_spot(candidates, low, high, prefer_center)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        selected = candidates[idx]
        
        step_trace.append({
            "step": step_num,
            "selected_text": selected,
            "ratio": ratio,
            "all_ratios": ratios,
            "has_answer": contains_answer[idx],
            "strategy": strategy,
        })
        
        accumulated_reasoning = accumulated_reasoning + " " + selected if accumulated_reasoning else selected
        
        if contains_answer[idx]:
            break
    
    # Final answer extraction
    final_prompt = build_final_answer_prompt(question, accumulated_reasoning, kind=kind)
    final_answer_text = generate_greedy(client, final_prompt)
    
    final_reasoning = accumulated_reasoning + "\n\n" + final_answer_text.strip()
    
    step_trace.append({
        "step": len(step_trace),
        "selected_text": final_answer_text.strip(),
        "ratio": score_chains([final_answer_text])[0],
        "all_ratios": [score_chains([final_answer_text])[0]],
        "has_answer": True,
        "is_final_extraction": True,
        "strategy": strategy,
    })
    
    return final_reasoning.strip(), step_trace, all_ratios


def run_experiment(cfg, strategy: str):
    """Run experiment with a specific selection strategy."""
    print(f"\n{'='*60}")
    print(f"Running experiment with strategy: {strategy.upper()}")
    print(f"{'='*60}\n")
    
    set_seed(cfg["seed"])
    
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
    
    preds, golds = [], []
    traces = []
    all_lzw_ratios = []
    
    K = cfg["reasoning"]["K"]
    low, high = cfg["reasoning"]["sweet_spot"]["low"], cfg["reasoning"]["sweet_spot"]["high"]
    prefer_center = cfg["reasoning"]["sweet_spot"]["prefer_center"]
    max_steps = cfg["reasoning"].get("max_steps", 10)
    
    for ex in tqdm(data_iter, desc=f"{strategy}"):
        q = ex["question"]
        gt_raw = ex["answer"]
        kind = "math" if ex["task"] == "math" else "qa"
        
        def normalize_answer(ans_raw: str) -> str:
            if kind == "math":
                return parse_gsm8k_numeric(ans_raw)
            else:
                return parse_yesno(ans_raw)
        
        # Run with strategy
        final_reasoning, step_trace, ratios = iterative_reasoning_with_strategy(
            client=client,
            question=q,
            kind=kind,
            K=K,
            max_steps=max_steps,
            strategy=strategy,
            low=low,
            high=high,
            prefer_center=prefer_center
        )
        
        all_lzw_ratios.extend(ratios)
        
        final_answer_raw = extract_final_answer(final_reasoning, kind=kind)
        final_answer = normalize_answer(final_answer_raw)
        
        if kind == "math":
            gold = parse_gsm8k_numeric(gt_raw)
        else:
            gold = parse_yesno(gt_raw)
        
        preds.append(final_answer)
        golds.append(gold)
        
        if cfg["io"]["save_traces"]:
            traces.append({
                "id": ex["id"],
                "question": q,
                "gold": gold,
                "strategy": strategy,
                "final_answer": final_answer,
                "final_reasoning": final_reasoning,
                "steps": step_trace,
                "lzw_ratios": ratios,
            })
    
    acc = accuracy(preds, golds)
    avg_lzw = sum(all_lzw_ratios) / len(all_lzw_ratios) if all_lzw_ratios else 0.0
    
    results = {
        "strategy": strategy,
        "dataset": ds_name,
        "n": len(golds),
        "accuracy": acc,
        "avg_lzw_ratio": avg_lzw,
        "min_lzw_ratio": min(all_lzw_ratios) if all_lzw_ratios else 0.0,
        "max_lzw_ratio": max(all_lzw_ratios) if all_lzw_ratios else 0.0,
    }
    
    print(f"\nResults for {strategy}:")
    print(json.dumps(results, indent=2))
    
    return results, traces


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--strategies", type=str, nargs="+", 
                       default=["lowest", "median", "highest", "sweet_spot"],
                       help="Selection strategies to test")
    args = parser.parse_args()
    
    cfg = load_yaml(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    cfg = apply_env(cfg)
    
    # Ensure output directory
    base_out_dir = Path(cfg["io"]["out_dir"])
    ensure_dir(base_out_dir)
    
    all_results = []
    
    # Run experiments for each strategy
    for strategy in args.strategies:
        strategy_out_dir = base_out_dir / f"strategy_{strategy}"
        ensure_dir(strategy_out_dir)
        cfg["io"]["out_dir"] = str(strategy_out_dir)
        
        results, traces = run_experiment(cfg, strategy)
        all_results.append(results)
        
        # Save individual results
        if cfg["io"]["save_jsonl"]:
            save_jsonl(traces, strategy_out_dir / "traces.jsonl")
            with open(strategy_out_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
    
    # Save combined results
    combined_results = {
        "dataset": cfg["dataset"]["name"],
        "limit": cfg["dataset"]["limit"],
        "K": cfg["reasoning"]["K"],
        "max_steps": cfg["reasoning"]["max_steps"],
        "strategies": all_results,
    }
    
    with open(base_out_dir / "combined_results.json", "w") as f:
        json.dump(combined_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for r in all_results:
        print(f"{r['strategy']:15s} | Accuracy: {r['accuracy']:.4f} | Avg LZW: {r['avg_lzw_ratio']:.4f}")
    print("="*60)
    
    print(f"\nResults saved to: {base_out_dir}")


if __name__ == "__main__":
    main()
