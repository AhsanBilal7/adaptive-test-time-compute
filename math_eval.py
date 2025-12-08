import re
import time
import json
import csv
import yaml
import argparse
import signal
from pathlib import Path
from typing import Optional, Dict, List, Any

from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from src.math_agent_core import MathAgent
from src.math_agent_core import get_field_from_completion
from BALROG.balrog.client import create_llm_client
from src.help_functions.prompts_templates import *

import sympy
from sympy.parsing.latex import parse_latex
import logging

eval_logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ===== LMEval Functions (from minerva_math) =====

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} expression from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    """Remove \\boxed{} wrapper from a string."""
    if s is None:
        return ""
    
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    
    if not s.startswith(left):
        return s

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    Copied from Lewkowycz et al. (2022) - Appendix D
    """
    if final_answer is None:
        return ""
    
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    Check if two normalized latex strings are mathematically equivalent.
    From LMEval minerva_math.
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    """Extract answer from 'Final Answer: The final answer is X. I hope it is correct.' format"""
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?)\. I hope it is correct\.",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


def check_answer_equivalence(pred: str, gold: str) -> bool:
    """
    Check if prediction matches ground truth using LMEval's approach:
    1. Extract unnormalized answer from prediction (handles "Final Answer: ..." format)
    2. Normalize both answers
    3. Check exact string match
    4. Check mathematical equivalence using is_equiv
    """
    # Extract unnormalized answer from prediction if it's in the competition format
    pred_extracted = get_unnormalized_answer(pred)
    
    # If extraction failed, use the raw prediction
    if pred_extracted == "[invalidanswer]":
        pred_extracted = pred
    
    # Normalize both answers
    pred_normalized = normalize_final_answer(pred_extracted)
    gold_normalized = normalize_final_answer(gold)
    
    # Check exact string match first
    if pred_normalized == gold_normalized:
        return True
    
    # Check mathematical equivalence using is_equiv
    return is_equiv(pred_normalized, gold_normalized)


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
    
    print(f"[INFO] Loading MATH dataset from Hugging Face (HuggingFaceH4/MATH-500)")
    
    ds = load_dataset("HuggingFaceH4/MATH-500")
    
    print(f"[INFO] Dataset splits available: {list(ds.keys())}")
    dataset = ds[split]
    
    print(f"[INFO] Loaded {len(dataset)} problems from {split} split")
    
    # Use 'subject' instead of 'type'
    if problem_types:
        dataset = dataset.filter(lambda x: x["subject"] in problem_types)
        print(f"[INFO] Filtered to {len(dataset)} problems of subjects: {problem_types}")
    
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
    
    # NEW: Additional tracking metrics
    tool_usage_stats = {}
    compute_strategy_stats = {}
    reasoning_length_stats = []
    verification_stats = {"attempted": 0, "passed": 0, "failed": 0}
    plan_usage_stats = {"with_plan": 0, "without_plan": 0}
    compute_param_distribution = {}
    error_analysis = {"extraction_failures": 0, "timeout_errors": 0}
    
    csv_path = output_dir / (
        f"mode-{agent_config['mode']}_planner-{agent_config['use_planner']}_"
        f"toolsel-{agent_config['use_tool_selector']}_computesel-{agent_config['use_compute_selector']}_"
        f"fixedtool-{agent_config['fixed_tool']}_fixedcompute-{agent_config['fixed_compute']}_"
        f"pf-{agent_config['planning_frequency']}_results_live.csv"
    )

    csv_headers = [
        "problem_id", "problem_type", "level",
        "gold_answer", "gold_answer_normalized",
        "predicted_answer_structured", 
        "predicted_answer_normalized",
        "is_correct", "cumulative_accuracy", "time_seconds",
        "reasoning_length", "word_count", "tools_used", "compute_strategy"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
    
    print(f"[INFO] CSV results will be saved to: {csv_path}")
    
    start_time = time.time()
    
    for idx in range(len(dataset)):
        problem_data = dataset[idx]
        
        problem = problem_data["problem"]
        gold_answer_raw = problem_data["answer"]
        problem_type = problem_data["subject"]
        level = problem_data["level"]
        solution = problem_data["solution"]
        
        # Extract and normalize ground truth answer
        # First try to extract from boxed format if present in solution
        boxed_answer = last_boxed_only_string(solution)
        if boxed_answer:
            gold_answer_extracted = remove_boxed(boxed_answer)
        else:
            # Fall back to the provided answer field
            gold_answer_extracted = gold_answer_raw
        
        gold_answer_normalized = normalize_final_answer(gold_answer_extracted)
        
        print(f"\n{'='*60}")
        print(f"[INFO] Problem {idx + 1}/{len(dataset)}")
        print(f"Subject: {problem_type}, Level: {level}")
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
            final_answer_user_prompt=FINAL_ANSWER_USER_PROMPT,
            unstructured_final_answer_system_prompt=UNSTRUCTURED_FINAL_ANSWER_SYSTEM_PROMPT,
            unstructured_final_answer_user_prompt=UNSTRUCTURED_FINAL_ANSWER_USER_PROMPT,
            direct_solve_prompt=DIRECT_SOLVE_PROMPT,
            direct_solve_system_prompt=DIRECT_SOLVE_SYSTEM_PROMPT,
            tool_selector_system_prompt=TOOL_SELECTOR_SYSTEM_PROMPT,
            compute_selector_system_prompt=COMPUTE_SELECTOR_SYSTEM_PROMPT,
        )
        problem_time = time.time() - problem_start
        
        # Extract unnormalized answer from unstructured response
        predicted_answer_normalized = get_unnormalized_answer(response.answer_unstructured)
        
        # If extraction failed, fall back to structured answer
        if predicted_answer_normalized == "[invalidanswer]":
            predicted_answer_normalized = response.answer
            error_analysis["extraction_failures"] += 1
        
        # Check equivalence using LMEval approach
        is_correct = check_answer_equivalence(response.answer_unstructured, gold_answer_extracted)
        
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
        
        # NEW: Track tool usage
        for tool in response.metadata.get("tools_used", []):
            tool_usage_stats[tool] = tool_usage_stats.get(tool, 0) + 1
        
        # NEW: Track compute strategies
        for config in response.metadata.get("compute_configs_used", []):
            strategy = config.get("strategy", "unknown")
            param = config.get("param", 0)
            
            compute_strategy_stats[strategy] = compute_strategy_stats.get(strategy, 0) + 1
            
            # Track parameter distribution
            key = f"{strategy}_param_{param}"
            compute_param_distribution[key] = compute_param_distribution.get(key, 0) + 1
        
        # NEW: Track reasoning length
        reasoning_length = len(response.reasoning) if response.reasoning else 0
        word_count = len(response.reasoning.split()) if response.reasoning else 0
        reasoning_length_stats.append({
            "problem_id": idx,
            "length": reasoning_length,
            "word_count": word_count,
            "is_correct": is_correct
        })
        
        # NEW: Track plan usage
        if response.plan:
            plan_usage_stats["with_plan"] += 1
        else:
            plan_usage_stats["without_plan"] += 1
        
        # NEW: Track verification usage
        compute_metadata = response.metadata.get("compute_metadata", [])
        for meta in compute_metadata:
            if meta.get("tool") in ["numeric_verifier", "verifier"]:
                verification_stats["attempted"] += 1
                result = meta.get("result", {})
                if isinstance(result, dict) and result.get("is_valid"):
                    verification_stats["passed"] += 1
                else:
                    verification_stats["failed"] += 1
        
        accuracy = correct / total * 100

        print(f"\n--- RESPONSE DETAILS ---")
        print(f"Reasoning: {response.reasoning[:200]}..." if len(response.reasoning) > 200 else f"Reasoning: {response.reasoning}")
        print(f"Plan: {response.plan}")
        print(f"\n--- ANSWERS ---")
        print(f"Predicted (structured): {response.answer}")
        print(f"Predicted (unstructured/raw): {response.answer_unstructured[:200]}...")
        print(f"Predicted (normalized/extracted): {predicted_answer_normalized}")
        print(f"Gold (raw): {gold_answer_extracted}")
        print(f"Gold (normalized): {gold_answer_normalized}")
        print(f"\n--- EVALUATION ---")
        print(f"Correct: {'✅' if is_correct else '❌'}")
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Time: {problem_time:.2f}s")
        print(f"Reasoning Length: {reasoning_length} chars, {word_count} words")
        
        result = {
            "problem_id": idx,
            "problem": problem,
            "problem_type": problem_type,
            "level": level,
            "gold_answer": gold_answer_extracted,
            "gold_answer_normalized": gold_answer_normalized,
            "gold_answer_raw": gold_answer_raw,
            "gold_solution": solution,
            "predicted_answer_structured": response.answer,
            "predicted_answer_unstructured": response.answer_unstructured,
            "predicted_answer_normalized": predicted_answer_normalized,
            "reasoning": response.reasoning,
            "reasoning_length": reasoning_length,
            "word_count": word_count,
            "plan": response.plan,
            "is_correct": is_correct,
            "time_seconds": problem_time,
            "tools_used": response.metadata.get("tools_used", []),
            "compute_configs_used": response.metadata.get("compute_configs_used", []),
            "metadata": response.metadata
        }
        results.append(result)
        
        # Get primary compute strategy for CSV
        primary_compute_strategy = "none"
        if response.metadata.get("compute_configs_used"):
            primary_compute_strategy = response.metadata["compute_configs_used"][0].get("strategy", "none")
        
        csv_row = {
            "problem_id": idx,
            "problem_type": problem_type,
            "level": level,
            "gold_answer": gold_answer_extracted,
            "gold_answer_normalized": gold_answer_normalized,
            "predicted_answer_structured": response.answer,
            "predicted_answer_normalized": predicted_answer_normalized,
            "is_correct": is_correct,
            "cumulative_accuracy": f"{accuracy:.2f}",
            "time_seconds": f"{problem_time:.2f}",
            "reasoning_length": reasoning_length,
            "word_count": word_count,
            "tools_used": ",".join(response.metadata.get("tools_used", [])),
            "compute_strategy": primary_compute_strategy
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
    
    # NEW: Calculate additional aggregate statistics
    
    # Tool usage percentages
    total_tool_calls = sum(tool_usage_stats.values())
    tool_usage_percentages = {
        tool: (count / total_tool_calls * 100) if total_tool_calls > 0 else 0
        for tool, count in tool_usage_stats.items()
    }
    
    # Compute strategy percentages
    total_compute_calls = sum(compute_strategy_stats.values())
    compute_strategy_percentages = {
        strategy: (count / total_compute_calls * 100) if total_compute_calls > 0 else 0
        for strategy, count in compute_strategy_stats.items()
    }
    
    # Reasoning length analysis
    avg_reasoning_length = sum(r["length"] for r in reasoning_length_stats) / len(reasoning_length_stats) if reasoning_length_stats else 0
    avg_word_count = sum(r["word_count"] for r in reasoning_length_stats) / len(reasoning_length_stats) if reasoning_length_stats else 0
    
    # Reasoning length by correctness
    correct_reasoning_lengths = [r["length"] for r in reasoning_length_stats if r["is_correct"]]
    incorrect_reasoning_lengths = [r["length"] for r in reasoning_length_stats if not r["is_correct"]]
    
    avg_correct_reasoning_length = sum(correct_reasoning_lengths) / len(correct_reasoning_lengths) if correct_reasoning_lengths else 0
    avg_incorrect_reasoning_length = sum(incorrect_reasoning_lengths) / len(incorrect_reasoning_lengths) if incorrect_reasoning_lengths else 0
    
    # Compute intensity metrics
    total_compute_params = sum(
        config.get("param", 0)
        for result in results
        for config in result.get("compute_configs_used", [])
    )
    avg_compute_params_per_problem = total_compute_params / total if total > 0 else 0
    
    # Accuracy by compute strategy
    accuracy_by_compute_strategy = {}
    for result in results:
        for config in result.get("compute_configs_used", []):
            strategy = config.get("strategy", "unknown")
            if strategy not in accuracy_by_compute_strategy:
                accuracy_by_compute_strategy[strategy] = {"correct": 0, "total": 0}
            accuracy_by_compute_strategy[strategy]["total"] += 1
            if result["is_correct"]:
                accuracy_by_compute_strategy[strategy]["correct"] += 1
    
    for strategy in accuracy_by_compute_strategy:
        stats = accuracy_by_compute_strategy[strategy]
        stats["accuracy"] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    # Accuracy by tool
    accuracy_by_tool = {}
    for result in results:
        for tool in result.get("tools_used", []):
            if tool not in accuracy_by_tool:
                accuracy_by_tool[tool] = {"correct": 0, "total": 0}
            accuracy_by_tool[tool]["total"] += 1
            if result["is_correct"]:
                accuracy_by_tool[tool]["correct"] += 1
    
    for tool in accuracy_by_tool:
        stats = accuracy_by_tool[tool]
        stats["accuracy"] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    # Accuracy with/without plan
    accuracy_with_plan = sum(1 for r in results if r["is_correct"] and r["plan"]) / plan_usage_stats["with_plan"] * 100 if plan_usage_stats["with_plan"] > 0 else 0
    accuracy_without_plan = sum(1 for r in results if r["is_correct"] and not r["plan"]) / plan_usage_stats["without_plan"] * 100 if plan_usage_stats["without_plan"] > 0 else 0
    
    # Verification effectiveness
    verification_effectiveness = {
        "total_problems_with_verification": verification_stats["attempted"],
        "verification_pass_rate": (verification_stats["passed"] / verification_stats["attempted"] * 100) if verification_stats["attempted"] > 0 else 0,
        **verification_stats
    }
    
    # Time analysis
    problem_times = [r["time_seconds"] for r in results]
    time_analysis = {
        "avg_time": sum(problem_times) / len(problem_times) if problem_times else 0,
        "min_time": min(problem_times) if problem_times else 0,
        "max_time": max(problem_times) if problem_times else 0,
        "median_time": sorted(problem_times)[len(problem_times)//2] if problem_times else 0,
        "total_time": total_time
    }
    
    # Time by correctness
    correct_times = [r["time_seconds"] for r in results if r["is_correct"]]
    incorrect_times = [r["time_seconds"] for r in results if not r["is_correct"]]
    
    time_by_correctness = {
        "avg_time_correct": sum(correct_times) / len(correct_times) if correct_times else 0,
        "avg_time_incorrect": sum(incorrect_times) / len(incorrect_times) if incorrect_times else 0
    }
    
    # Compute efficiency
    compute_efficiency = {
        "accuracy_per_compute_param": final_accuracy / avg_compute_params_per_problem if avg_compute_params_per_problem > 0 else 0,
        "correct_per_second": correct / total_time if total_time > 0 else 0,
        "total_compute_params_used": total_compute_params,
        "avg_compute_params_per_problem": avg_compute_params_per_problem
    }
    
    # Error analysis
    error_analysis["extraction_failure_rate"] = (error_analysis["extraction_failures"] / total * 100) if total > 0 else 0
    
    agent_stats = agent.get_stats()
    agent_stats.update({
        # Existing metrics
        "total_problems": total,
        "correct": correct,
        "accuracy": final_accuracy,
        "total_time_seconds": total_time,
        "avg_time_per_problem": total_time / total if total > 0 else 0,
        "stats_by_type": stats_by_type,
        "stats_by_level": stats_by_level,
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_level": accuracy_by_level,
        
        # NEW: Tool usage metrics
        "tool_usage": {
            "counts": tool_usage_stats,
            "percentages": tool_usage_percentages,
            "accuracy_by_tool": accuracy_by_tool
        },
        
        # NEW: Compute strategy metrics
        "compute_strategies": {
            "counts": compute_strategy_stats,
            "percentages": compute_strategy_percentages,
            "param_distribution": compute_param_distribution,
            "accuracy_by_strategy": accuracy_by_compute_strategy,
            "total_compute_params": total_compute_params,
            "avg_params_per_problem": avg_compute_params_per_problem
        },
        
        # NEW: Reasoning analysis
        "reasoning_analysis": {
            "avg_length_chars": avg_reasoning_length,
            "avg_word_count": avg_word_count,
            "avg_length_correct": avg_correct_reasoning_length,
            "avg_length_incorrect": avg_incorrect_reasoning_length,
            "length_correlation_with_accuracy": avg_correct_reasoning_length / avg_incorrect_reasoning_length if avg_incorrect_reasoning_length > 0 else 0,
            "detailed_stats": reasoning_length_stats
        },
        
        # NEW: Planning analysis
        "planning_analysis": {
            "usage": plan_usage_stats,
            "accuracy_with_plan": accuracy_with_plan,
            "accuracy_without_plan": accuracy_without_plan,
            "plan_effectiveness_gain": accuracy_with_plan - accuracy_without_plan
        },
        
        # NEW: Verification metrics
        "verification": verification_effectiveness,
        
        # NEW: Time analysis
        "time_analysis": {
            **time_analysis,
            **time_by_correctness,
            "time_efficiency": {
                "seconds_per_correct_answer": total_time / correct if correct > 0 else 0,
                "time_overhead_for_incorrect": time_by_correctness["avg_time_incorrect"] - time_by_correctness["avg_time_correct"]
            }
        },
        
        # NEW: Compute efficiency
        "compute_efficiency": compute_efficiency,
        
        # NEW: Error analysis
        "error_analysis": error_analysis,
        
        # NEW: Configuration summary
        "configuration": {
            "mode": agent_config.get("mode"),
            "use_planner": agent_config.get("use_planner"),
            "use_tool_selector": agent_config.get("use_tool_selector"),
            "use_compute_selector": agent_config.get("use_compute_selector"),
            "fixed_tool": agent_config.get("fixed_tool"),
            "fixed_compute": agent_config.get("fixed_compute"),
            "planning_frequency": agent_config.get("planning_frequency")
        }
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
    
    print(f"\n--- ACCURACY BY PROBLEM SUBJECT ---")
    for ptype, acc in accuracy_by_type.items():
        print(f"  {ptype}: {acc:.2f}% ({stats_by_type[ptype]['correct']}/{stats_by_type[ptype]['total']})")
    
    print(f"\n--- ACCURACY BY DIFFICULTY LEVEL ---")
    for level, acc in accuracy_by_level.items():
        print(f"  {level}: {acc:.2f}% ({stats_by_level[level]['correct']}/{stats_by_level[level]['total']})")
    
    print(f"\n--- TOOL USAGE ---")
    for tool, count in tool_usage_stats.items():
        percentage = tool_usage_percentages[tool]
        acc = accuracy_by_tool.get(tool, {}).get("accuracy", 0)
        print(f"  {tool}: {count} times ({percentage:.1f}%), Accuracy: {acc:.2f}%")
    
    print(f"\n--- COMPUTE STRATEGIES ---")
    for strategy, count in compute_strategy_stats.items():
        percentage = compute_strategy_percentages[strategy]
        acc = accuracy_by_compute_strategy.get(strategy, {}).get("accuracy", 0)
        print(f"  {strategy}: {count} times ({percentage:.1f}%), Accuracy: {acc:.2f}%")
    
    print(f"\n--- PLANNING EFFECTIVENESS ---")
    print(f"  With plan: {plan_usage_stats['with_plan']} problems, Accuracy: {accuracy_with_plan:.2f}%")
    print(f"  Without plan: {plan_usage_stats['without_plan']} problems, Accuracy: {accuracy_without_plan:.2f}%")
    print(f"  Plan effectiveness gain: {accuracy_with_plan - accuracy_without_plan:+.2f}%")
    
    print(f"\n--- REASONING ANALYSIS ---")
    print(f"  Avg reasoning length: {avg_reasoning_length:.0f} chars, {avg_word_count:.0f} words")
    print(f"  Avg length (correct): {avg_correct_reasoning_length:.0f} chars")
    print(f"  Avg length (incorrect): {avg_incorrect_reasoning_length:.0f} chars")
    
    print(f"\n--- COMPUTE EFFICIENCY ---")
    print(f"  Total compute params used: {total_compute_params}")
    print(f"  Avg compute params per problem: {avg_compute_params_per_problem:.2f}")
    print(f"  Accuracy per compute param: {compute_efficiency['accuracy_per_compute_param']:.2f}")
    print(f"  Correct answers per second: {compute_efficiency['correct_per_second']:.4f}")
    
    print(f"\n--- ERROR ANALYSIS ---")
    print(f"  Extraction failures: {error_analysis['extraction_failures']} ({error_analysis['extraction_failure_rate']:.2f}%)")
    
    print(f"\nResults saved to: {final_path}")
    print(f"Live CSV results saved to: {csv_path}")
    
    return final_results


def create_agent_from_config(config: Dict):
    def client_factory():
        return create_llm_client(DictConfig(config["client"]))
    
    agent = MathAgent(
        client_factory=client_factory(),
        use_planner=config["agent"]["use_planner"],
        use_tool_selector=config["agent"]["use_tool_selector"],
        use_compute_selector=config["agent"]["use_compute_selector"],
        fixed_tool=config["agent"].get("fixed_tool"),
        fixed_compute=config["agent"].get("fixed_compute"),
        remember_cot=config["agent"]["remember_cot"],
        max_text_history=config["agent"]["max_text_history"],
    )
    
    return agent


def main():
    parser = argparse.ArgumentParser(description="Evaluate Math Agent on MATH dataset")
    parser.add_argument("--config", type=str, default="./config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
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