import yaml
import argparse
from typing import Dict, Any
from omegaconf import DictConfig

from BALROG.balrog.client import create_llm_client
from src.help_functions.prompts_templates import *

from src.Evaluator import get_evaluator
from src.universal_agent import UniversalAgent
from src.math_core import AMOCore, MATHCore, AIME24Core
from src.gsm8k_core import GSM8KCore
from src.help_functions.prm_selector import PRMSelector
import time
import json
import csv
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import numpy as np


def evaluate_reasoning_agent(
    agent_factory: Callable,
    dataset: List[Dict[str, Any]],
    output_dir: str,
    agent_config: Dict[str, Any],
    check_answer_fn: Callable[[str, str], bool],
    extract_prediction_fn: Callable[[Any, Dict], str],
    csv_filename_template: str,
    json_filename_template: str,
    pathway_filename_template: str,
    prompt_kwargs: Dict[str, Any],
    num_iterations: int = 20,
    problem_key: str = "problem",
    gold_answer_key: str = "gold_answer_extracted",
    problem_type_key: str = "problem_type",
    difficulty_key: str = "level",
    extra_problem_keys: Optional[List[str]] = None,
    use_prm_selection: bool = True,
    prm_selection_metric: str = "mean_reward",
    verbose: bool = True,
    dataset_name: str = "MATH"
):
    """
    Evaluate a reasoning agent on a dataset with PRM-based iteration selection.
    
    Args:
        agent_factory: Factory function to create fresh agent instances
        dataset: List of problem dictionaries
        output_dir: Directory to save results
        agent_config: Agent configuration dictionary
        check_answer_fn: Function to check if prediction matches gold answer
        extract_prediction_fn: Function to extract prediction from response
        csv_filename_template: Template for CSV filename
        json_filename_template: Template for JSON filename
        pathway_filename_template: Template for pathway filename
        prompt_kwargs: Keyword arguments for agent.solve()
        num_iterations: Number of iterations to run per problem
        problem_key: Key for problem text in dataset
        gold_answer_key: Key for gold answer in dataset
        problem_type_key: Key for problem type in dataset
        difficulty_key: Key for difficulty level in dataset
        extra_problem_keys: Additional keys to include in results
        use_prm_selection: Whether to use PRM for selecting best iteration
        prm_selection_metric: Metric to use for PRM selection ('mean_reward', 'min_reward', 'final_reward')
        verbose: Whether to print detailed progress
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = get_evaluator(dataset_name.upper())
    agent = agent_factory()
    
    # Initialize PRM selector if enabled
    prm_selector = None
    if use_prm_selection:
        print(f"[INFO] Initializing PRM selector with metric: {prm_selection_metric}")
        prm_selector = PRMSelector()
    
    results = []
    correct = 0
    total = 0
    
    stats_by_type = {}
    stats_by_level = {}
    
    tool_usage_stats = {}
    compute_strategy_stats = {}
    reasoning_length_stats = []
    verification_stats = {"attempted": 0, "passed": 0, "failed": 0}
    plan_usage_stats = {"with_plan": 0, "without_plan": 0}
    compute_param_distribution = {}
    error_analysis = {"extraction_failures": 0, "timeout_errors": 0}
    
    # PRM-specific statistics
    prm_stats = {
        "iterations_selected": [],  # Which iteration was selected for each problem
        "selection_improvements": 0,  # How many times PRM selected a different iteration than last
        "avg_reward_improvement": 0.0,
        "reward_scores": []  # All reward scores
    }
    
    pathway_results = []
    
    csv_path = output_dir / csv_filename_template.format(**agent_config)
    csv_headers = [
        "problem_id", problem_type_key, difficulty_key,
        "gold_answer", "iteration", "predicted_answer",
        "is_correct", "cumulative_accuracy", "time_seconds",
        "reasoning_length", "word_count", "tools_used", "compute_strategy",
        "selected_by_prm", "prm_mean_reward", "prm_min_reward", "prm_final_reward"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
    
    if verbose:
        print(f"[INFO] CSV results will be saved to: {csv_path}")
        print(f"[INFO] Evaluating {len(dataset)} problems with {num_iterations} iterations each...")
        if use_prm_selection:
            print(f"[INFO] Using PRM-based selection with metric: {prm_selection_metric}")
    
    start_time = time.time()
    
    for problem_data in dataset:
        problem_id = problem_data.get("problem_id", total)
        problem = problem_data[problem_key]
        gold_answer = problem_data[gold_answer_key]
        problem_type = problem_data.get(problem_type_key, "unknown")
        level = problem_data.get(difficulty_key, "unknown")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[INFO] Problem {problem_id + 1}/{len(dataset)}")
            print(f"{problem_type_key}: {problem_type}, {difficulty_key}: {level}")
            print(f"Problem: {problem[:150]}...")
        
        iteration_results = []
        pathway_info = {
            "problem_id": problem_id,
            "problem": problem,
            problem_type_key: problem_type,
            difficulty_key: level,
            "gold_answer": gold_answer,
            "iterations": []
        }
        
        # Run all iterations
        for iteration in range(num_iterations):
            agent.reset()
            
            iteration_start = time.time()
            response = agent.solve(problem, **prompt_kwargs)
            iteration_time = time.time() - iteration_start
            
            #TODO ADDED THE EVALUATOR FROM SATORI
            # predicted_answer = extract_prediction_fn(response, problem_data)
            # is_correct = check_answer_fn(predicted_answer, gold_answer)


            predicted_answer = evaluator._extract_answer_from_model_completion(response.answer_unstructured)
            # gold_answer = evaluator._extract_answer_from_gold_solution(gold_answer)
            is_correct = evaluator._check_answers_equiv(predicted_answer, gold_answer)

            print("••••••••••••••••••••••••••••••••••••••••••••••")
            print(f"Question: {problem}")
            print(f"Answer: {response.answer_unstructured}")
            print(f"Predicted (normalized): {predicted_answer}")
            print(f"Gold (normalized): {gold_answer}")
            print(f"Is Correct: {is_correct}")
            print("••••••••••••••••••••••••••••••••••••••••••••••")
            reasoning_length = len(response.reasoning) if response.reasoning else 0
            word_count = len(response.reasoning.split()) if response.reasoning else 0
            
            tools_used_list = response.metadata.get("tools_used", [])
            compute_configs_list = response.metadata.get("compute_configs_used", [])
            
            primary_compute_strategy = "none"
            if compute_configs_list:
                primary_compute_strategy = compute_configs_list[0].get("strategy", "none")
            
            iteration_pathway = {
                "iteration": iteration + 1,
                "tools_selected": tools_used_list,
                "compute_configs": compute_configs_list,
                "plan": response.plan,
                "time_seconds": iteration_time
            }
            pathway_info["iterations"].append(iteration_pathway)
            
            iteration_result = {
                "iteration": iteration + 1,
                "predicted_answer": predicted_answer,
                "answer_unstructured": response.answer_unstructured,
                "reasoning": response.reasoning,
                "reasoning_steps": response.reasoning_steps,  # Store individual steps
                "reasoning_length": reasoning_length,
                "word_count": word_count,
                "plan": response.plan,
                "is_correct": is_correct,
                "time_seconds": iteration_time,
                "tools_used": tools_used_list,
                "compute_configs_used": compute_configs_list,
                "metadata": response.metadata
            }
            iteration_results.append(iteration_result)
            
            if verbose and iteration == 0:
                print(f"\n--- ITERATION {iteration + 1} ---")
                print(f"Reasoning: {response.reasoning[:200]}..." if len(response.reasoning) > 200 else f"Reasoning: {response.reasoning}")
                print(f"Predicted: {predicted_answer}")
                print(f"Correct: {'✅' if is_correct else '❌'}")
        
        # Select best iteration using PRM or use last iteration
        if use_prm_selection and prm_selector:
            
            
            best_iteration_idx, best_iteration, all_prm_scores = prm_selector.select_best_iteration(
                problem=problem,
                iteration_results=iteration_results,
                selection_metric=prm_selection_metric,
                system_prompt="Please reason step by step, and put your final answer within \\boxed{}.",
                meta_data=response.metadata
            )
            
            # Store PRM scores in pathway info
            pathway_info["prm_scores"] = all_prm_scores
            pathway_info["selected_iteration_idx"] = best_iteration_idx + 1
            pathway_info["selection_metric"] = prm_selection_metric
            
            # Update statistics
            prm_stats["iterations_selected"].append(best_iteration_idx + 1)
            prm_stats["reward_scores"].extend(all_prm_scores)
            
            if best_iteration_idx != num_iterations - 1:
                prm_stats["selection_improvements"] += 1
            
            # Mark selected iteration in iteration_results
            for idx, iter_result in enumerate(iteration_results):
                iter_result["selected_by_prm"] = (idx == best_iteration_idx)
                iter_result["prm_scores"] = all_prm_scores[idx] if idx < len(all_prm_scores) else {}
            
            selected_iteration = best_iteration
            
            if verbose:
                print(f"\n[PRM SELECTION] Selected iteration {best_iteration_idx + 1}/{num_iterations}")
                print(f"  {prm_selection_metric}: {all_prm_scores[best_iteration_idx].get(prm_selection_metric, 0):.4f}")
        else:
            # Use last iteration (original behavior)
            selected_iteration = iteration_results[-1]
            best_iteration_idx = num_iterations - 1
            
            # Mark last iteration as selected
            for idx, iter_result in enumerate(iteration_results):
                iter_result["selected_by_prm"] = (idx == best_iteration_idx)
                iter_result["prm_scores"] = {}
        
        pathway_results.append(pathway_info)
        
        # Use selected iteration for accuracy calculation
        is_correct = selected_iteration["is_correct"]
        predicted_answer = selected_iteration["predicted_answer"]
        
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
        
        # Collect statistics from all iterations
        for iter_result in iteration_results:
            for tool in iter_result.get("tools_used", []):
                tool_usage_stats[tool] = tool_usage_stats.get(tool, 0) + 1
            
            for config in iter_result.get("compute_configs_used", []):
                strategy = config.get("strategy", "unknown")
                param = config.get("param", 0)
                
                compute_strategy_stats[strategy] = compute_strategy_stats.get(strategy, 0) + 1
                
                key = f"{strategy}_param_{param}"
                compute_param_distribution[key] = compute_param_distribution.get(key, 0) + 1
            
            reasoning_length_stats.append({
                "problem_id": problem_id,
                "iteration": iter_result["iteration"],
                "length": iter_result["reasoning_length"],
                "word_count": iter_result["word_count"],
                "is_correct": iter_result["is_correct"]
            })
            
            if iter_result["plan"]:
                plan_usage_stats["with_plan"] += 1
            else:
                plan_usage_stats["without_plan"] += 1
            
            compute_metadata = iter_result["metadata"].get("compute_metadata", [])
            for meta in compute_metadata:
                if meta.get("tool") in ["numeric_verifier", "verifier"]:
                    verification_stats["attempted"] += 1
                    result = meta.get("result", {})
                    if isinstance(result, dict) and result.get("is_valid"):
                        verification_stats["passed"] += 1
                    else:
                        verification_stats["failed"] += 1
        
        accuracy = correct / total * 100
        
        if verbose:
            print(f"\n--- FINAL RESULT (Selected Iteration {best_iteration_idx + 1}) ---")
            print(f"Gold: {gold_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {'✅' if is_correct else '❌'}")
            print(f"Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        result = {
            "problem_id": problem_id,
            "problem": problem,
            problem_type_key: problem_type,
            difficulty_key: level,
            "gold_answer": gold_answer,
            "iterations": iteration_results,
            "selected_iteration_idx": best_iteration_idx + 1,
            "final_predicted_answer": predicted_answer,
            "final_is_correct": is_correct,
            "total_time_seconds": sum(ir["time_seconds"] for ir in iteration_results)
        }
        
        if extra_problem_keys:
            for key in extra_problem_keys:
                if key in problem_data:
                    result[key] = problem_data[key]
        
        results.append(result)
        
        # Write all iterations to CSV
        for iter_result in iteration_results:
            primary_compute_strategy = "none"
            if iter_result.get("compute_configs_used"):
                primary_compute_strategy = iter_result["compute_configs_used"][0].get("strategy", "none")
            
            prm_scores = iter_result.get("prm_scores", {})
            
            csv_row = {
                "problem_id": problem_id,
                problem_type_key: problem_type,
                difficulty_key: level,
                "gold_answer": gold_answer,
                "iteration": iter_result["iteration"],
                "predicted_answer": iter_result["predicted_answer"],
                "is_correct": iter_result["is_correct"],
                "cumulative_accuracy": f"{accuracy:.2f}",
                "time_seconds": f"{iter_result['time_seconds']:.2f}",
                "reasoning_length": iter_result["reasoning_length"],
                "word_count": iter_result["word_count"],
                "tools_used": ",".join(iter_result.get("tools_used", [])),
                "compute_strategy": primary_compute_strategy,
                "selected_by_prm": iter_result.get("selected_by_prm", False),
                "prm_mean_reward": f"{prm_scores.get('mean_reward', 0):.4f}" if prm_scores else "",
                "prm_min_reward": f"{prm_scores.get('min_reward', 0):.4f}" if prm_scores else "",
                "prm_final_reward": f"{prm_scores.get('final_reward', 0):.4f}" if prm_scores else ""
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
    
    total_tool_calls = sum(tool_usage_stats.values())
    tool_usage_percentages = {
        tool: (count / total_tool_calls * 100) if total_tool_calls > 0 else 0
        for tool, count in tool_usage_stats.items()
    }
    
    accuracy_by_tool = {}
    for result in results:
        for iter_result in result["iterations"]:
            for tool in iter_result.get("tools_used", []):
                if tool not in accuracy_by_tool:
                    accuracy_by_tool[tool] = {"correct": 0, "total": 0}
                accuracy_by_tool[tool]["total"] += 1
                if iter_result["is_correct"]:
                    accuracy_by_tool[tool]["correct"] += 1
    
    for tool in accuracy_by_tool:
        stats = accuracy_by_tool[tool]
        stats["accuracy"] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    total_compute_calls = sum(compute_strategy_stats.values())
    compute_strategy_percentages = {
        strategy: (count / total_compute_calls * 100) if total_compute_calls > 0 else 0
        for strategy, count in compute_strategy_stats.items()
    }
    
    accuracy_by_compute_strategy = {}
    for result in results:
        for iter_result in result["iterations"]:
            for config in iter_result.get("compute_configs_used", []):
                strategy = config.get("strategy", "unknown")
                if strategy not in accuracy_by_compute_strategy:
                    accuracy_by_compute_strategy[strategy] = {"correct": 0, "total": 0}
                accuracy_by_compute_strategy[strategy]["total"] += 1
                if iter_result["is_correct"]:
                    accuracy_by_compute_strategy[strategy]["correct"] += 1
    
    for strategy in accuracy_by_compute_strategy:
        stats = accuracy_by_compute_strategy[strategy]
        stats["accuracy"] = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
    
    avg_reasoning_length = sum(r["length"] for r in reasoning_length_stats) / len(reasoning_length_stats) if reasoning_length_stats else 0
    avg_word_count = sum(r["word_count"] for r in reasoning_length_stats) / len(reasoning_length_stats) if reasoning_length_stats else 0
    
    correct_reasoning_lengths = [r["length"] for r in reasoning_length_stats if r["is_correct"]]
    incorrect_reasoning_lengths = [r["length"] for r in reasoning_length_stats if not r["is_correct"]]
    
    avg_correct_reasoning_length = sum(correct_reasoning_lengths) / len(correct_reasoning_lengths) if correct_reasoning_lengths else 0
    avg_incorrect_reasoning_length = sum(incorrect_reasoning_lengths) / len(incorrect_reasoning_lengths) if incorrect_reasoning_lengths else 0
    
    total_compute_params = sum(
        config.get("param", 0)
        for result in results
        for iter_result in result["iterations"]
        for config in iter_result.get("compute_configs_used", [])
    )
    total_iterations = total * num_iterations
    avg_compute_params_per_problem = total_compute_params / total_iterations if total_iterations > 0 else 0
    
    accuracy_with_plan = sum(1 for r in results for ir in r["iterations"] if ir["is_correct"] and ir["plan"]) / plan_usage_stats["with_plan"] * 100 if plan_usage_stats["with_plan"] > 0 else 0
    accuracy_without_plan = sum(1 for r in results for ir in r["iterations"] if ir["is_correct"] and not ir["plan"]) / plan_usage_stats["without_plan"] * 100 if plan_usage_stats["without_plan"] > 0 else 0
    
    problem_times = [ir["time_seconds"] for r in results for ir in r["iterations"]]
    correct_times = [ir["time_seconds"] for r in results for ir in r["iterations"] if ir["is_correct"]]
    incorrect_times = [ir["time_seconds"] for r in results for ir in r["iterations"] if not ir["is_correct"]]
    
    time_analysis = {
        "avg_time": sum(problem_times) / len(problem_times) if problem_times else 0,
        "min_time": min(problem_times) if problem_times else 0,
        "max_time": max(problem_times) if problem_times else 0,
        "median_time": sorted(problem_times)[len(problem_times)//2] if problem_times else 0,
        "total_time": total_time,
        "avg_time_correct": sum(correct_times) / len(correct_times) if correct_times else 0,
        "avg_time_incorrect": sum(incorrect_times) / len(incorrect_times) if incorrect_times else 0
    }
    
    # Calculate PRM-specific statistics
    if use_prm_selection and prm_stats["iterations_selected"]:
        prm_stats["avg_selected_iteration"] = np.mean(prm_stats["iterations_selected"])
        prm_stats["median_selected_iteration"] = np.median(prm_stats["iterations_selected"])
        prm_stats["selection_improvement_rate"] = (prm_stats["selection_improvements"] / total * 100) if total > 0 else 0
        
        # Calculate average rewards
        all_mean_rewards = [score.get("mean_reward", 0) for score in prm_stats["reward_scores"]]
        if all_mean_rewards:
            prm_stats["avg_mean_reward"] = np.mean(all_mean_rewards)
            prm_stats["std_mean_reward"] = np.std(all_mean_rewards)
    
    agent_stats = agent.get_stats()
    agent_stats.update({
        "num_iterations": num_iterations,
        "total_problems": total,
        "correct": correct,
        "accuracy": final_accuracy,
        "total_time_seconds": total_time,
        "avg_time_per_problem": total_time / total if total > 0 else 0,
        "stats_by_type": stats_by_type,
        "stats_by_level": stats_by_level,
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_level": accuracy_by_level,
        
        "tool_usage": {
            "counts": tool_usage_stats,
            "percentages": tool_usage_percentages,
            "accuracy_by_tool": accuracy_by_tool
        },
        
        "compute_strategies": {
            "counts": compute_strategy_stats,
            "percentages": compute_strategy_percentages,
            "param_distribution": compute_param_distribution,
            "accuracy_by_strategy": accuracy_by_compute_strategy,
            "total_compute_params": total_compute_params,
            "avg_params_per_problem": avg_compute_params_per_problem
        },
        
        "reasoning_analysis": {
            "avg_length_chars": avg_reasoning_length,
            "avg_word_count": avg_word_count,
            "avg_length_correct": avg_correct_reasoning_length,
            "avg_length_incorrect": avg_incorrect_reasoning_length,
            "length_correlation_with_accuracy": avg_correct_reasoning_length / avg_incorrect_reasoning_length if avg_incorrect_reasoning_length > 0 else 0,
            "detailed_stats": reasoning_length_stats
        },
        
        "planning_analysis": {
            "usage": plan_usage_stats,
            "accuracy_with_plan": accuracy_with_plan,
            "accuracy_without_plan": accuracy_without_plan,
            "plan_effectiveness_gain": accuracy_with_plan - accuracy_without_plan
        },
        
        "verification": {
            "total_problems_with_verification": verification_stats["attempted"],
            "verification_pass_rate": (verification_stats["passed"] / verification_stats["attempted"] * 100) if verification_stats["attempted"] > 0 else 0,
            **verification_stats
        },
        
        "time_analysis": {
            **time_analysis,
            "time_efficiency": {
                "seconds_per_correct_answer": total_time / correct if correct > 0 else 0,
                "time_overhead_for_incorrect": time_analysis["avg_time_incorrect"] - time_analysis["avg_time_correct"]
            }
        },
        
        "compute_efficiency": {
            "accuracy_per_compute_param": final_accuracy / avg_compute_params_per_problem if avg_compute_params_per_problem > 0 else 0,
            "correct_per_second": correct / total_time if total_time > 0 else 0,
            "total_compute_params_used": total_compute_params,
            "avg_compute_params_per_problem": avg_compute_params_per_problem
        },
        
        "prm_selection": {
            "enabled": use_prm_selection,
            "selection_metric": prm_selection_metric if use_prm_selection else None,
            **prm_stats
        } if use_prm_selection else {"enabled": False},
        
        "error_analysis": error_analysis,
        
        "configuration": agent_config
    })
    
    final_results = {
        "statistics": agent_stats,
        "results": results
    }
    
    final_path = output_dir / json_filename_template.format(**agent_config)
    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=2)
    
    pathway_path = output_dir / pathway_filename_template.format(**agent_config)
    with open(pathway_path, "w") as f:
        json.dump(pathway_results, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total problems: {total}")
        print(f"Correct: {correct}")
        print(f"Overall Accuracy: {final_accuracy:.2f}%")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time per problem: {total_time / total:.2f}s")
        
        if use_prm_selection:
            print(f"\n--- PRM SELECTION STATISTICS ---")
            print(f"  Selection metric: {prm_selection_metric}")
            print(f"  Avg selected iteration: {prm_stats.get('avg_selected_iteration', 0):.2f}")
            print(f"  Median selected iteration: {prm_stats.get('median_selected_iteration', 0):.2f}")
            print(f"  Times PRM selected non-last iteration: {prm_stats['selection_improvements']}/{total} ({prm_stats.get('selection_improvement_rate', 0):.2f}%)")
            if "avg_mean_reward" in prm_stats:
                print(f"  Avg mean reward: {prm_stats['avg_mean_reward']:.4f} ± {prm_stats['std_mean_reward']:.4f}")
        
        print(f"\n--- ACCURACY BY {problem_type_key.upper()} ---")
        for ptype, acc in accuracy_by_type.items():
            print(f"  {ptype}: {acc:.2f}% ({stats_by_type[ptype]['correct']}/{stats_by_type[ptype]['total']})")
        
        print(f"\n--- ACCURACY BY {difficulty_key.upper()} ---")
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
        print(f"  With plan: {plan_usage_stats['with_plan']} iterations, Accuracy: {accuracy_with_plan:.2f}%")
        print(f"  Without plan: {plan_usage_stats['without_plan']} iterations, Accuracy: {accuracy_without_plan:.2f}%")
        print(f"  Plan effectiveness gain: {accuracy_with_plan - accuracy_without_plan:+.2f}%")
        
        print(f"\nResults saved to: {final_path}")
        print(f"Pathway results saved to: {pathway_path}")
        print(f"Live CSV results saved to: {csv_path}")
    
    return final_results

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_universal_agent(config: Dict) -> UniversalAgent:
    client = create_llm_client(DictConfig(config["client"]))
    
    agent = UniversalAgent(
        client_factory=client,
        use_planner=config["agent"]["use_planner"],
        use_tool_selector=config["agent"]["use_tool_selector"],
        use_compute_selector=config["agent"]["use_compute_selector"],
        fixed_tool=config["agent"].get("fixed_tool"),
        fixed_compute=config["agent"].get("fixed_compute"),
        remember_cot=config["agent"]["remember_cot"],
        max_text_history=config["agent"]["max_text_history"],
    )
    
    return agent


def get_prompts() -> Dict[str, str]:
    return {
        "planning_prompt_template": PLANNING_PROMPT_TEMPLATE,
        "system_prompt": MATH_SYSTEM_PROMPT,
        "tool_selector_prompt": TOOL_SELECTOR_PROMPT,
        "tool_selector_system_prompt": TOOL_SELECTOR_SYSTEM_PROMPT,
        "compute_selector_prompt": COMPUTE_SELECTOR_PROMPT,
        "compute_selector_system_prompt": COMPUTE_SELECTOR_SYSTEM_PROMPT,
        "self_reflection_instruction_prompt": SELF_REFLECTION_INSTRUCTION_PROMPT,
        "cot_instruction_prompt": COT_INSTRUCTION_PROMPT,
        "prm_scoring_prompt": PRM_SCORING_PROMPT,
        "final_answer_system_prompt": FINAL_ANSWER_SYSTEM_PROMPT,
        "final_answer_user_prompt": FINAL_ANSWER_USER_PROMPT,
        "direct_solve_prompt": DIRECT_SOLVE_PROMPT,
        "direct_solve_system_prompt": DIRECT_SOLVE_SYSTEM_PROMPT,
        "unstructured_final_answer_system_prompt": UNSTRUCTURED_FINAL_ANSWER_SYSTEM_PROMPT,
        "unstructured_final_answer_user_prompt": UNSTRUCTURED_FINAL_ANSWER_USER_PROMPT,
        "direct_unstructured_final_answer_system_prompt": DIRECT_UNSTRUCTURED_FINAL_ANSWER_SYSTEM_PROMPT,
        "direct_unstructured_final_answer_user_prompt": DIRECT_UNSTRUCTURED_FINAL_ANSWER_USER_PROMPT,
    }


def evaluate_dataset(
    dataset_name: str,
    config: Dict,
    verbose: bool = True
) -> Dict[str, Any]:
    agent = create_universal_agent(config)
    
    prompts = get_prompts()
    
    num_iterations = config["eval"].get("num_iterations", 20)
    use_prm_selection = config["eval"].get("use_prm_selection", False)
    # print(f"🚧🚧🚧🚧🚧🚧🚧use_prm_selection: {use_prm_selection}")
    prm_selection_metric = config["eval"].get("prm_selection_metric", "mean_reward")
    
    if dataset_name.lower() == "math" or dataset_name.lower() == "aime24" or dataset_name.lower() == "amo":
        if dataset_name.lower() == "aime24":
            core = AIME24Core(agent, prompts)
        if dataset_name.lower() == "amo":
            core = AMOCore(agent, prompts)
        else:
            core = MATHCore(agent, prompts)
        
        dataset = core.load_dataset(
            split=config["eval"]["split"],
            problem_types=config["eval"].get("problem_types"),
            difficulty_levels=config["eval"].get("difficulty_levels"),
            max_problems=config["eval"].get("max_problems"),
        )
        
        csv_template = (
            "math_mode-{mode}_planner-{use_planner}_"
            "toolsel-{use_tool_selector}_computesel-{use_compute_selector}_"
            "results_live.csv"
        )
        json_template = (
            "math_mode-{mode}_planner-{use_planner}_"
            "toolsel-{use_tool_selector}_computesel-{use_compute_selector}_"
            "results_final.json"
        )
        pathway_template = (
            "math_mode-{mode}_planner-{use_planner}_"
            "toolsel-{use_tool_selector}_computesel-{use_compute_selector}_"
            "pathway.json"
        )
        
    elif dataset_name.lower() == "gsm8k":
        core = GSM8KCore(agent, prompts)
        
        dataset = core.load_dataset(
            split=config["eval"].get("split", "test"),
            max_problems=config["eval"].get("max_problems"),
        )
        
        csv_template = (
            "gsm8k_mode-{mode}_planner-{use_planner}_"
            "toolsel-{use_tool_selector}_computesel-{use_compute_selector}_"
            "results_live.csv"
        )
        json_template = (
            "gsm8k_mode-{mode}_planner-{use_planner}_"
            "toolsel-{use_tool_selector}_computesel-{use_compute_selector}_"
            "results_final.json"
        )
        pathway_template = (
            "gsm8k_mode-{mode}_planner-{use_planner}_"
            "toolsel-{use_tool_selector}_computesel-{use_compute_selector}_"
            "pathway.json"
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'math' or 'gsm8k'")
    
    def agent_factory():
        fresh_agent = create_universal_agent(config)
        if dataset_name.lower() == "math":
            return MATHCore(fresh_agent, prompts)
        elif dataset_name.lower() == "aime24":
            return AIME24Core(fresh_agent, prompts)
        else:
            return GSM8KCore(fresh_agent, prompts)
    
    results = evaluate_reasoning_agent(
        agent_factory=agent_factory,
        dataset=dataset,
        output_dir=config["eval"]["output_dir"],
        agent_config=config["agent"],
        check_answer_fn=core.check_answer,
        extract_prediction_fn=core.extract_prediction,
        csv_filename_template=csv_template,
        json_filename_template=json_template,
        pathway_filename_template=pathway_template,
        prompt_kwargs={},
        num_iterations=num_iterations,
        use_prm_selection=use_prm_selection,
        prm_selection_metric=prm_selection_metric,
        problem_key="problem",
        gold_answer_key="gold_answer_extracted",
        problem_type_key="problem_type",
        difficulty_key="level",
        extra_problem_keys=["gold_answer_raw", "gold_answer_normalized", "solution"] if dataset_name.lower() == "math" else ["gold_answer_raw", "gold_answer_normalized"],
        verbose=verbose,
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Universal Agent Evaluation with PRM Selection")
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config.yaml", 
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",
        choices=["math", "gsm8k", "aime24", "amo"],
        help="Dataset to evaluate on (math or gsm8k)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    parser.add_argument(
        "--no-prm",
        action="store_true",
        help="Disable PRM-based iteration selection"
    )
    parser.add_argument(
        "--prm-metric",
        type=str,
        default="mean_reward",
        choices=["mean_reward", "min_reward", "final_reward"],
        help="Metric to use for PRM selection"
    )
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Override config with command line arguments
    # if args.no_prm:
    #     config["eval"]["use_prm_selection"] = False
    # else:
    #     config["eval"]["use_prm_selection"] = True
    
    if config["eval"]["num_iterations"] > 1:
        config["eval"]["use_prm_selection"] = True
    else:
        config["eval"]["use_prm_selection"] = False
        
    config["eval"]["prm_selection_metric"] = args.prm_metric
    
    print(f"\n{'='*70}")
    print(f"UNIVERSAL AGENT EVALUATION - {args.dataset.upper()} DATASET")
    if config["eval"].get("use_prm_selection", True):
        print(f"PRM Selection: ENABLED (metric: {args.prm_metric})")
    else:
        print(f"PRM Selection: DISABLED")
    print(f"{'='*70}\n")
    
    results = evaluate_dataset(
        dataset_name=args.dataset,
        config=config,
        verbose=args.verbose
    )
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    main()