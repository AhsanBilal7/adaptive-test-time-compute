"""Compute-cost metrics for finished runs: theoretical FLOPs (F_theo) and compute intensity score (S_CI)."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List


def compute_average_compute_cost(
    results_json: Dict[str, Any],
    model_config: Optional[Dict[str, Any]] = None,
    *,
    K: float = 1e6,
    token_multiplier: float = 1.5,
) -> Dict[str, Any]:
    """Compute accuracy, S_CI, and per-problem F_theo from a results_final.json dict.

    S_CI = G * T * (1 + 0.1 * C) / K, where G is generations, T reasoning tokens,
    and C tool calls per problem. F_theo = 2 * M * min(1.5 * T, L_ctx) * G.
    """
    statistics = results_json.get("statistics", {})
    results = results_json.get("results", [])

    total_problems = statistics.get("total_problems", len(results))
    num_iterations = statistics.get("num_iterations", 1)
    prm_enabled = statistics.get("prm_selection", {}).get("enabled", False)

    # Accuracy comes from results_final.json (this file)
    total_accuracy = statistics.get("accuracy", None)

    if total_problems == 0:
        return {
            "accuracy": total_accuracy,
            "compute_intensity_score": 0.0,
            "theoretical_flops_per_problem": "0.0000e+00",
            "prm_selection_used": prm_enabled,
        }

    total_generations = 0
    total_reasoning_tokens = 0
    total_tool_calls = 0

    print(f"\n[DEBUG] Processing {total_problems} problems...")
    print(f"[DEBUG] PRM Selection: {prm_enabled}")
    print(f"[DEBUG] Num iterations per problem: {num_iterations}")

    for idx, result in enumerate(results):
        iterations = result.get("iterations", [])

        if prm_enabled:
            selected_idx = result.get("selected_iteration_idx", len(iterations)) - 1
            iterations_to_process = iterations[: selected_idx + 1]
        else:
            iterations_to_process = iterations

        if idx == 0:
            print(f"\n[DEBUG] First problem sample:")
            print(f"  Total iterations available: {len(iterations)}")
            print(f"  Iterations to process: {len(iterations_to_process)}")

        for iter_idx, iteration in enumerate(iterations_to_process):
            compute_configs = iteration.get("compute_configs_used", [])
            tools_used = iteration.get("tools_used", [])
            word_count = iteration.get("word_count", 0)
            reasoning_length = iteration.get("reasoning_length", 0)
            reasoning = iteration.get("reasoning", "")

            if word_count == 0 and reasoning_length > 0:
                word_count = reasoning_length // 5
            if word_count == 0 and reasoning:
                word_count = len(reasoning.split())
            if word_count == 0:
                predicted_answer = iteration.get("predicted_answer", "")
                if predicted_answer:
                    word_count = len(predicted_answer.split())
            if word_count == 0:
                raise ValueError("Failed to compute word count for iteration")

            if idx == 0 and iter_idx == 0:
                print(f"\n[DEBUG] First iteration details:")
                print(f"  compute_configs_used: {compute_configs}")
                print(f"  tools_used: {tools_used}")
                print(f"  word_count (original): {iteration.get('word_count', 0)}")
                print(f"  reasoning_length: {reasoning_length}")
                print(f"  reasoning sample: {reasoning[:100]}")
                print(f"  word_count (computed): {word_count}")

            generations_this_iter = 0
            if not compute_configs:
                generations_this_iter = 1
            else:
                for config in compute_configs:
                    param = config.get("param", 1)
                    strategy = config.get("strategy", "none")

                    if strategy in ["best_of_n", "beam_search", "lookahead"]:
                        generations_this_iter += param
                    elif strategy in ["none", "direct"]:
                        generations_this_iter += 1
                    else:
                        generations_this_iter += 1

            total_generations += generations_this_iter
            total_reasoning_tokens += word_count
            total_tool_calls += len(tools_used)

            if idx == 0 and iter_idx == 0:
                print(f"  generations_this_iter: {generations_this_iter}")

    avg_generations = total_generations / total_problems
    avg_reasoning_tokens = total_reasoning_tokens / total_problems
    avg_total_tokens = avg_reasoning_tokens * token_multiplier
    avg_tool_calls = total_tool_calls / total_problems

    compute_intensity_score = (avg_generations * avg_reasoning_tokens * (1 + avg_tool_calls * 0.1)) / K
    compute_intensity_score = float(f"{compute_intensity_score:.4e}")

    theoretical_flops_per_problem = 0.0
    if model_config:
        model_params = model_config.get("model_parameters", 7e9)
        context_length = model_config.get("context_length", 2048)

        tokens_per_forward = min(avg_total_tokens, context_length)
        flops_per_forward = 2 * model_params * tokens_per_forward
        theoretical_flops_per_problem = flops_per_forward * avg_generations
        theoretical_flops_per_problem = float(f"{theoretical_flops_per_problem:.4e}")

    return {
        "accuracy": total_accuracy,
        "compute_intensity_score": compute_intensity_score,
        "theoretical_flops_per_problem": f"{theoretical_flops_per_problem:.4e}",
        "prm_selection_used": prm_enabled,
    }


def compute_and_save_metrics(results_json_path: str, model_config: Optional[Dict[str, Any]] = None):
    """Find the *_results_final.json in a run directory, compute cost metrics, and save them next to it."""
    results_path = Path(results_json_path)

    # Always read accuracy from *_results_final.json (not pathway)
    if results_path.is_dir():
        json_files = list(results_path.glob("*_results_final.json"))
        if not json_files:
            print(f"No results_final.json files found in {results_path}")
            return None, None, None
        results_path = json_files[0]
        print(f"Found results file: {results_path}")

    with open(results_path, "r") as f:
        results_json = json.load(f)

    compute_metrics = compute_average_compute_cost(results_json, model_config)

    output_path = results_path.parent / f"{results_path.stem}_compute_metrics.json"
    with open(output_path, "w") as f:
        json.dump(compute_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL METRICS (ONLY)")
    print(f"{'='*60}")
    print(f"  Accuracy (from results_final): {compute_metrics.get('accuracy')}")
    print(f"  Compute Intensity Score (S_CI): {compute_metrics['compute_intensity_score']:.4e}")
    print(f"  Theoretical FLOPs / problem: {compute_metrics['theoretical_flops_per_problem']}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}\n")

    return compute_metrics, output_path, results_path.name


def _print_summary_table(rows: List[Dict[str, Any]]) -> None:
    """Print a table of accuracy, compute intensity, and FLOPs per run."""
    if not rows:
        print("\nNo runs produced metrics.\n")
        return

    headers = ["FILE", "ACCURACY", "COMPUTE_INTENSITY", "FLOPS_PER_PROBLEM"]
    widths = {h: len(h) for h in headers}
    for r in rows:
        widths["FILE"] = max(widths["FILE"], len(str(r["file"])))
        widths["ACCURACY"] = max(widths["ACCURACY"], len(str(r["accuracy"])))
        widths["COMPUTE_INTENSITY"] = max(widths["COMPUTE_INTENSITY"], len(str(r["compute_intensity"])))
        widths["FLOPS_PER_PROBLEM"] = max(widths["FLOPS_PER_PROBLEM"], len(str(r["flops"])))

    def line(sep: str = "+", fill: str = "-") -> str:
        return (
            f"{sep}{fill * (widths['FILE'] + 2)}"
            f"{sep}{fill * (widths['ACCURACY'] + 2)}"
            f"{sep}{fill * (widths['COMPUTE_INTENSITY'] + 2)}"
            f"{sep}{fill * (widths['FLOPS_PER_PROBLEM'] + 2)}{sep}"
        )

    def row(vals: List[str]) -> str:
        return (
            f"| {vals[0]:<{widths['FILE']}} "
            f"| {vals[1]:<{widths['ACCURACY']}} "
            f"| {vals[2]:<{widths['COMPUTE_INTENSITY']}} "
            f"| {vals[3]:<{widths['FLOPS_PER_PROBLEM']}} |"
        )

    print("\nSUMMARY TABLE")
    print(line())
    print(row(headers))
    print(line())
    for r in rows:
        print(
            row(
                [
                    str(r["file"]),
                    str(r["accuracy"]),
                    str(r["compute_intensity"]),
                    str(r["flops"]),
                ]
            )
        )
    print(line())
    print()


if __name__ == "__main__":
    parent_dir = Path("./amo_results")

    model_config = {
        "model_parameters": 7e9,
        "context_length": 1024,
    }

    summary_rows: List[Dict[str, Any]] = []

    for run_dir in sorted(parent_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        print(f"\n[RUN] Processing: {run_dir}")
        metrics, out_path, results_file = compute_and_save_metrics(str(run_dir), model_config=model_config)

        if metrics is None:
            print(f"[SKIP] No *_results_final.json found in {run_dir}")
            continue

        summary_rows.append(
            {
                "file": str(run_dir),  # this is *_results_final.json filename
                "accuracy": metrics.get("accuracy"),
                "compute_intensity": f"{metrics['compute_intensity_score']:.4e}",
                "flops": metrics["theoretical_flops_per_problem"],
            }
        )

        print(f"[OK] Saved: {out_path}")

    _print_summary_table(summary_rows)


