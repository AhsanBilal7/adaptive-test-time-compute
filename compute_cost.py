import json
from pathlib import Path
from typing import Dict, Any

def compute_average_compute_cost(results_json: Dict[str, Any], model_config: Dict[str, Any] = None) -> Dict[str, float]:
    statistics = results_json.get("statistics", {})
    results = results_json.get("results", [])
    
    total_problems = statistics.get("total_problems", len(results))
    num_iterations = statistics.get("num_iterations", 1)
    prm_enabled = statistics.get("prm_selection", {}).get("enabled", False)
    
    if total_problems == 0:
        return {
            "avg_generations_per_problem": 0.0,
            "avg_reasoning_tokens_per_problem": 0.0,
            "avg_total_tokens_per_problem": 0.0,
            "avg_compute_params_per_problem": 0.0,
            "avg_tool_calls_per_problem": 0.0,
            "avg_iterations_per_problem": 0.0,
            "total_generations": 0,
            "total_reasoning_tokens": 0,
            "total_tool_calls": 0,
            "total_iterations": 0,
            "compute_intensity_score": 0.0,
            "model_normalized_compute": 0.0,
            "theoretical_flops_per_problem": 0.0,
        }
    
    total_generations = 0
    total_reasoning_tokens = 0
    total_tool_calls = 0
    total_iterations_computed = 0
    
    print(f"\n[DEBUG] Processing {total_problems} problems...")
    print(f"[DEBUG] PRM Selection: {prm_enabled}")
    print(f"[DEBUG] Num iterations per problem: {num_iterations}")
    
    for idx, result in enumerate(results):
        iterations = result.get("iterations", [])
        
        if prm_enabled:
            selected_idx = result.get("selected_iteration_idx", len(iterations)) - 1
            iterations_to_process = iterations[:selected_idx + 1]
        else:
            iterations_to_process = iterations
        
        total_iterations_computed += len(iterations_to_process)
        
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
                # word_count = 50
            
            if idx == 0 and iter_idx == 0:
                print(f"\n[DEBUG] First iteration details:")
                print(f"  compute_configs_used: {compute_configs}")
                print(f"  tools_used: {tools_used}")
                print(f"  word_count (original): {iteration.get('word_count', 0)}")
                print(f"  reasoning_length: {reasoning_length}")
                print(f"  reasoning sample: {reasoning[:100]}")
                print(f"  word_count (computed): {word_count}")
            
            generations_this_iter = 0
            if not compute_configs or len(compute_configs) == 0:
                generations_this_iter = 1
            else:
                for config in compute_configs:
                    param = config.get("param", 1)
                    strategy = config.get("strategy", "none")
                    
                    if strategy in ["best_of_n", "beam_search", "lookahead"]:
                        generations_this_iter += param
                    elif strategy == "none" or strategy == "direct":
                        generations_this_iter += 1
                    else:
                        generations_this_iter += 1
            
            total_generations += generations_this_iter
            total_reasoning_tokens += word_count
            total_tool_calls += len(tools_used)
            
            if idx == 0 and iter_idx == 0:
                print(f"  generations_this_iter: {generations_this_iter}")
    
    print(f"\n[DEBUG] Totals:")
    print(f"  total_generations: {total_generations}")
    print(f"  total_reasoning_tokens: {total_reasoning_tokens}")
    print(f"  total_tool_calls: {total_tool_calls}")
    print(f"  total_iterations_computed: {total_iterations_computed}")
    
    avg_generations = total_generations / total_problems if total_problems > 0 else 0
    avg_reasoning_tokens = total_reasoning_tokens / total_problems if total_problems > 0 else 0
    avg_total_tokens = avg_reasoning_tokens * 1.5
    
    compute_strategies = statistics.get("compute_strategies", {})
    total_compute_params = compute_strategies.get("total_compute_params", 0)
    total_all_iterations = total_problems * num_iterations
    avg_compute_params = total_compute_params / total_all_iterations if total_all_iterations > 0 else 0
    
    avg_tool_calls = total_tool_calls / total_problems if total_problems > 0 else 0
    avg_iterations = total_iterations_computed / total_problems if total_problems > 0 else 0
    
    # compute_intensity_score = (avg_generations * avg_reasoning_tokens * (1 + avg_tool_calls * 0.1)) / 1000
    compute_intensity_score = (avg_generations * avg_reasoning_tokens * (1 + avg_tool_calls * 0.1)) / total_problems**2
    compute_intensity_score = float(f"{compute_intensity_score:.4e}")

    print(f"\n[DEBUG] Averages:")
    print(f"  avg_generations: {avg_generations}")
    print(f"  avg_reasoning_tokens: {avg_reasoning_tokens}")
    print(f"  avg_tool_calls: {avg_tool_calls}")
    print(f"  compute_intensity_score: {compute_intensity_score}")
    
    model_normalized_compute = 0.0
    theoretical_flops_per_problem = 0.0
    
    if model_config:
        model_params = model_config.get("model_parameters", 7e9)
        context_length = model_config.get("context_length", 2048)
        
        tokens_per_forward = min(avg_total_tokens, context_length)
        flops_per_forward = 2 * model_params * tokens_per_forward
        theoretical_flops_per_problem = flops_per_forward * avg_generations
        theoretical_flops_per_problem = float(f"{theoretical_flops_per_problem:.4e}")

        base_model_params = 7e9
        model_normalized_compute = (model_params / base_model_params) * avg_generations * avg_total_tokens
    
    return {
        "avg_generations_per_problem": avg_generations,
        "avg_reasoning_tokens_per_problem": avg_reasoning_tokens,
        "avg_total_tokens_per_problem": avg_total_tokens,
        "avg_compute_params_per_problem": avg_compute_params,
        "avg_tool_calls_per_problem": avg_tool_calls,
        "avg_iterations_per_problem": avg_iterations,
        "total_generations": total_generations,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_tool_calls": total_tool_calls,
        "total_iterations": total_iterations_computed,
        "compute_intensity_score": compute_intensity_score,
        "model_normalized_compute": model_normalized_compute,
        "theoretical_flops_per_problem": f"{theoretical_flops_per_problem:.4e}",
        "prm_selection_used": prm_enabled,
    }

def compute_and_save_metrics(results_json_path: str, model_config: Dict[str, Any] = None):
    results_path = Path(results_json_path)
    
    if results_path.is_dir():
        json_files = list(results_path.glob("*_results_final.json"))
        if not json_files:
            print(f"No results_final.json files found in {results_path}")
            return None, None
        results_path = json_files[0]
        print(f"Found results file: {results_path}")
    
    with open(results_path, 'r') as f:
        results_json = json.load(f)
    
    compute_metrics = compute_average_compute_cost(results_json, model_config)
    
    output_path = results_path.parent / f"{results_path.stem}_compute_metrics.json"
    
    with open(output_path, 'w') as f:
        json.dump(compute_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL COMPUTE METRICS")
    print(f"{'='*60}")
    print(f"  PRM Selection: {'Enabled' if compute_metrics.get('prm_selection_used') else 'Disabled'}")
    print(f"  Compute Intensity Score (S_CI): {compute_metrics['compute_intensity_score']:.4f}")
    print(f"  Avg Generations: {compute_metrics['avg_generations_per_problem']:.2f}")
    print(f"  Avg Reasoning Tokens: {compute_metrics['avg_reasoning_tokens_per_problem']:.2f}")
    print(f"  Avg Total Tokens: {compute_metrics['avg_total_tokens_per_problem']:.2f}")
    print(f"  Avg Tool Calls: {compute_metrics['avg_tool_calls_per_problem']:.2f}")
    print(f"  Avg Iterations Used: {compute_metrics['avg_iterations_per_problem']:.2f}")
    print(f"  Avg Compute Params: {compute_metrics['avg_compute_params_per_problem']:.2f}")
    if model_config:
        print(f"  Theoretical FLOPs: {compute_metrics['theoretical_flops_per_problem']}")
        print(f"  Model Normalized Compute: {compute_metrics['model_normalized_compute']:.2e}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}\n")
    
    return compute_metrics, output_path

if __name__ == "__main__":
    compute_and_save_metrics(
        # "./all_saved_results/qwen_2.5_7b/direct_results_test_with_prm/",
        "./dynamic_results_testing_with_prm_qwen2.5_10_iterations",
        model_config={
            "model_parameters": 8e9,
            "context_length": 1024
        }
    )

# if __name__ == "__main__":
#     base = Path("./all_saved_results/gsm8k")

#     # TEMP: enumerate all subfolders and runs
#     for model_dir in base.iterdir():
#         if not model_dir.is_dir():
#             continue

#         model_name = model_dir.name.lower()

#         # 🔥 Infer model parameters from folder name
#         if "8b" in model_name or "8" in model_name:
#             model_params = 8e9
#         else:
#             model_params = 7e9

#         model_config = {
#             "model_parameters": model_params,
#             "context_length": 1024,
#         }

#         print(f"\n====================================")
#         print(f"[MODEL] {model_dir.name}")
#         print(f"  Using model_parameters = {model_params:.2e}")
#         print(f"====================================")

#         # Loop over each run inside the model
#         for run_dir in model_dir.iterdir():
#             if not run_dir.is_dir():
#                 continue

#             print(f"\n[RUN] Processing {run_dir}")

#             metrics, output_path = compute_and_save_metrics(
#                 str(run_dir),
#                 model_config=model_config
#             )

#             if metrics is None:
#                 print(f"[SKIP] No results_final.json found in {run_dir}")
#             else:
#                 print(f"[OK] Saved metrics to: {output_path}")
