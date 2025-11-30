# BALROG/balrog/agents/math_eval.py

import re
import time
import json
import csv
from pathlib import Path
from typing import Optional, Dict, List, Any

from datasets import load_dataset
from omegaconf import DictConfig

from src.math_agent_core import MathAgent

from src.math_agent_core import get_field_from_completion

from BALROG.balrog.client import create_llm_client
from src.reasoners.prompts_templates import *

def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    
    answer = str(answer).strip()
    answer = re.sub(r"\\boxed\{(.*?)\}", r"\1", answer)
    answer = re.sub(r"\$(.+?)\$", r"\1", answer)
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.replace("\\frac", "frac")
    
    return answer.lower().strip()


def _extract_number(text: str):
    match = re.search(r"-?\d+(\.\d+)?", text)
    if match:
        return float(match.group())
    return None


def check_answer_equivalence(pred: str, gold: str) -> bool:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if pred_norm == gold_norm:
        return True
    
    pred_num = _extract_number(pred_norm)
    gold_num = _extract_number(gold_norm)
    
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-6
    
    return False


def evaluate_math_dataset(
    agent_factory,
    output_dir: str,
    num_workers: int = 1,
    max_problems: Optional[int] = None,
    split: str = "train",
    problem_types: Optional[List[str]] = None,
    difficulty_levels: Optional[List[str]] = None,
    agent_config: Optional[Dict] = None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading MATH dataset from Hugging Face (qwedsacf/competition_math)")
    
    ds = load_dataset("qwedsacf/competition_math")
    
    print(f"[INFO] Dataset splits available: {list(ds.keys())}")
    dataset = ds[split]
    
    print(f"[INFO] Loaded {len(dataset)} problems from {split} split")
    
    if problem_types:
        dataset = dataset.filter(lambda x: x["type"] in problem_types)
        print(f"[INFO] Filtered to {len(dataset)} problems of types: {problem_types}")
    
    if difficulty_levels:
        dataset = dataset.filter(lambda x: x["level"] in difficulty_levels)
        print(f"[INFO] Filtered to {len(dataset)} problems of levels: {difficulty_levels}")
    
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
        print(f"[INFO] Limited to {len(dataset)} problems")
    
    agent = agent_factory()
    
    results = []
    correct = 0
    total = 0
    
    stats_by_type = {}
    stats_by_level = {}
    
    csv_path = output_dir / (
        f"mode-{agent_config['mode']}_planner-{agent_config['use_planner']}_"
        f"toolsel-{agent_config['use_tool_selector']}_computesel-{agent_config['use_compute_selector']}_"
        f"fixedtool-{agent_config['fixed_tool']}_fixedcompute-{agent_config['fixed_compute']}_"
        f"pf-{agent_config['planning_frequency']}_results_live.csv"
    )
    csv_headers = [
        "problem_id", "problem_type", "level", "gold_answer", "predicted_answer", 
        "is_correct", "cumulative_accuracy", "time_seconds"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
    
    print(f"[INFO] CSV results will be saved to: {csv_path}")
    
    start_time = time.time()
    
    for idx in range(len(dataset)):
        problem_data = dataset[idx]
        
        problem = problem_data["problem"]
        gold_answer = problem_data["solution"]
        problem_type = problem_data["type"]
        level = problem_data["level"]
        
        gold_answer_extracted = normalize_answer(gold_answer)
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", gold_answer)
        if boxed_match:
            gold_answer_extracted = normalize_answer(boxed_match.group(1))
        
        print(f"\n{'='*60}")
        print(f"[INFO] Problem {idx + 1}/{len(dataset)}")
        print(f"Type: {problem_type}, Level: {level}")
        print(f"Problem: {problem[:150]}...")
        
        agent.reset()
        
        problem_start = time.time()
        response = agent.solve(
        problem,
        planning_prompt_template=PLANNING_PROMPT_TEMPLATE,
        math_system_prompt=MATH_SYSTEM_PROMPT,
        tool_selector_prompt=TOOL_SELECTOR_PROMPT,
        compute_selector_prompt=COMPUTE_SELECTOR_PROMPT,
        reactive_instruction_prompt=REACTIVE_INSTRUCTION_PROMPT,
        cot_instruction_prompt=COT_INSTRUCTION_PROMPT,
        prm_scoring_prompt=PRM_SCORING_PROMPT,
        final_answer_system_prompt=FINAL_ANSWER_SYSTEM_PROMPT,
        )
        problem_time = time.time() - problem_start
        
        is_correct = check_answer_equivalence(response.answer, gold_answer_extracted)
        
        if is_correct:
            correct += 1
        total += 1
        
        if problem_type not in stats_by_type:
            stats_by_type[problem_type] = {"correct": 0, "total": 0}
        stats_by_type[problem_type]["total"] += 1
        if is_correct:
            stats_by_type[problem_type]["correct"] += 1
        
        if level not in stats_by_level:
            stats_by_level[level] = {"correct": 0, "total": 0}
        stats_by_level[level]["total"] += 1
        if is_correct:
            stats_by_level[level]["correct"] += 1
        
        accuracy = correct / total * 100

        print(f"Reasoning: {response.reasoning}")
        print(f"Plan: {response.plan}")
        print(f"Predicted: {response.answer}")
        print(f"Gold: {gold_answer_extracted}")
        print(f"Correct: {'✅' if is_correct else '❌'}")
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Time: {problem_time:.2f}s")
        
        result = {
            "problem_id": idx,
            "problem": problem,
            "problem_type": problem_type,
            "level": level,
            "gold_answer": gold_answer_extracted,
            "gold_solution": gold_answer,
            "predicted_answer": response.answer,
            "reasoning": response.reasoning,
            "plan": response.plan,
            "is_correct": is_correct,
            "time_seconds": problem_time,
            "tools_used": response.metadata.get("tools_used", []),
            "compute_configs_used": response.metadata.get("compute_configs_used", []),
            "metadata": response.metadata
        }
        results.append(result)
        
        csv_row = {
            "problem_id": idx,
            "problem_type": problem_type,
            "level": level,
            "gold_answer": gold_answer_extracted,
            "predicted_answer": response.answer,
            "is_correct": is_correct,
            "cumulative_accuracy": f"{accuracy:.2f}",
            "time_seconds": f"{problem_time:.2f}"
        }
        
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writerow(csv_row)
    
    total_time = time.time() - start_time
    
    final_accuracy = correct / total * 100 if total > 0 else 0
    
    accuracy_by_type = {
        ptype: (stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0)
        for ptype, stats in stats_by_type.items()
    }
    
    accuracy_by_level = {
        level: (stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0)
        for level, stats in stats_by_level.items()
    }
    
    agent_stats = agent.get_stats()
    agent_stats.update({
        "total_problems": total,
        "correct": correct,
        "accuracy": final_accuracy,
        "total_time_seconds": total_time,
        "avg_time_per_problem": total_time / total if total > 0 else 0,
        "stats_by_type": stats_by_type,
        "stats_by_level": stats_by_level,
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_level": accuracy_by_level
    })
    
    final_results = {
        "statistics": agent_stats,
        "results": results
    }
    
    final_path = output_dir / (
        f"mode-{agent_config['mode']}_planner-{agent_config['use_planner']}_"
        f"toolsel-{agent_config['use_tool_selector']}_computesel-{agent_config['use_compute_selector']}_"
        f"fixedtool-{agent_config['fixed_tool']}_fixedcompute-{agent_config['fixed_compute']}_"
        f"pf-{agent_config['planning_frequency']}_results_final.json"
    )

    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total problems: {total}")
    print(f"Correct: {correct}")
    print(f"Overall Accuracy: {final_accuracy:.2f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg time per problem: {total_time / total:.2f}s")
    print(f"\nAccuracy by Problem Type:")
    for ptype, acc in accuracy_by_type.items():
        print(f"  {ptype}: {acc:.2f}% ({stats_by_type[ptype]['correct']}/{stats_by_type[ptype]['total']})")
    print(f"\nAccuracy by Difficulty Level:")
    for level, acc in accuracy_by_level.items():
        print(f"  {level}: {acc:.2f}% ({stats_by_level[level]['correct']}/{stats_by_level[level]['total']})")
    print(f"\nResults saved to: {final_path}")
    print(f"Live CSV results saved to: {csv_path}")
    
    return final_results


def create_agent_from_config(config: Dict):
    def client_factory():
        return create_llm_client(DictConfig(config["client"]))
    
    agent = MathAgent(
        client_factory=client_factory(),
        # mode=config["agent"]["mode"],
        # planning_frequency=config["agent"].get("planning_frequency"),
        use_planner=config["agent"]["use_planner"],
        use_tool_selector=config["agent"]["use_tool_selector"],
        use_compute_selector=config["agent"]["use_compute_selector"],
        fixed_tool=config["agent"].get("fixed_tool"),
        fixed_compute=config["agent"].get("fixed_compute"),
        remember_cot=config["agent"]["remember_cot"],
        max_text_history=config["agent"]["max_text_history"],
        # max_image_history=config["agent"]["max_image_history"],
        # domain=config["agent"].get("domain", "math")
    )
    
    return agent


def main():
    config = {
        "agent": {
            "mode": "dynamic",
            "use_planner": True,
            "use_tool_selector": True,
            "use_compute_selector": True,
            "remember_cot": True,
            "max_text_history": 16,
            "max_image_history": 0,
            "fixed_tool": None,
            "fixed_compute": None,
            "planning_frequency": None,
            "domain": "math"
        },
        "eval": {
            "num_workers": 16,
            "output_dir": "multi_tool_results",
            "max_problems": 250,
            "split": "train",
            "problem_types": None,
            "difficulty_levels": None
        },
        "client": {
            "client_name": "ollama",
            "base_url": "http://localhost:11434/v1",
            "model_id": "gemma3n:e4b",
            "generate_kwargs": {
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "timeout": 60,
            "max_retries": 5,
            "delay": 2,
            "alternate_roles": False
        }
    }
    
    agent_factory = lambda: create_agent_from_config(config)
    
    results = evaluate_math_dataset(
        agent_factory=agent_factory,
        output_dir=config["eval"]["output_dir"],
        num_workers=config["eval"]["num_workers"],
        max_problems=config["eval"].get("max_problems"),
        split=config["eval"]["split"],
        problem_types=config["eval"].get("problem_types"),
        difficulty_levels=config["eval"].get("difficulty_levels"),
        agent_config=config["agent"]
    )
    
    return results


if __name__ == "__main__":
    main()
