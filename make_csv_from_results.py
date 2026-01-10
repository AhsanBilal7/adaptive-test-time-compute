import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import seaborn as sns


def load_pathway_data(pathway_file: str) -> List[Dict[str, Any]]:
    """
    Load pathway data from either pathway.json or results_final.json format.
    Automatically detects and converts if needed.
    """
    with open(pathway_file, 'r') as f:
        data = json.load(f)
    
    # Check if this is results_final.json format (has 'results' or 'statistics' keys)
    if isinstance(data, dict) and ('results' in data or 'statistics' in data):
        print("[INFO] Detected results_final.json format, converting to pathway format...")
        results = data.get('results', [])
        
        pathway_data = []
        for idx, result in enumerate(results):
            metadata = result.get('metadata', {})
            pathway_info = metadata.get('pathway', [])
            
            if not pathway_info:
                # Create minimal pathway entry from available data
                pathway_info = [{
                    'iteration': 0,
                    'tools_selected': metadata.get('tools_used', []),
                    'compute_configs': metadata.get('compute_configs_used', []),
                    'plan': result.get('plan', ''),
                    'reasoning': result.get('reasoning', ''),
                    'time_seconds': result.get('time_seconds', 0)
                }]
            
            problem_pathway = {
                'problem_id': result.get('problem_id', idx),
                'problem_type': result.get('problem_type', 'unknown'),
                'level': result.get('level', 'unknown'),
                'is_correct': result.get('is_correct', False),
                'iterations': pathway_info
            }
            
            pathway_data.append(problem_pathway)
        
        print(f"[INFO] Converted {len(pathway_data)} problems from results format")
        return pathway_data
    
    # Assume it's already in pathway format
    return data


def plot_tool_selection_distribution(pathway_data: List[Dict], output_dir: Path):
    all_tools = []
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            all_tools.extend(iteration["tools_selected"])
    
    tool_counts = Counter(all_tools)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    tools = list(tool_counts.keys())
    counts = list(tool_counts.values())
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(tools)))
    ax.bar(tools, counts, color=colors)
    ax.set_xlabel('Tool Name', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Tool Selection Distribution Across All Iterations', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'tool_distribution.png', dpi=300)
    plt.close()


def plot_compute_strategy_distribution(pathway_data: List[Dict], output_dir: Path):
    all_strategies = []
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            for config in iteration["compute_configs"]:
                all_strategies.append(config.get("strategy", "unknown"))
    
    strategy_counts = Counter(all_strategies)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies = list(strategy_counts.keys())
    counts = list(strategy_counts.values())
    
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(strategies)))
    ax.bar(strategies, counts, color=colors)
    ax.set_xlabel('Compute Strategy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Compute Strategy Distribution Across All Iterations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'compute_strategy_distribution.png', dpi=300)
    plt.close()


def plot_compute_param_distribution(pathway_data: List[Dict], output_dir: Path):
    strategy_params = defaultdict(list)
    
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            for config in iteration["compute_configs"]:
                strategy = config.get("strategy", "unknown")
                param = config.get("param", 0)
                strategy_params[strategy].append(param)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    positions = []
    data_to_plot = []
    labels = []
    
    for i, (strategy, params) in enumerate(strategy_params.items()):
        positions.append(i)
        data_to_plot.append(params)
        labels.append(strategy)
    
    bp = ax.boxplot(data_to_plot, positions=positions, labels=labels, patch_artist=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Compute Strategy', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Compute Parameter Distribution by Strategy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'compute_param_boxplot.png', dpi=300)
    plt.close()


def plot_iteration_time_trends(pathway_data: List[Dict], output_dir: Path):
    num_iterations = len(pathway_data[0]["iterations"]) if pathway_data else 0
    
    iteration_times = defaultdict(list)
    
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            iter_num = iteration["iteration"]
            time_sec = iteration["time_seconds"]
            iteration_times[iter_num].append(time_sec)
    
    iterations = sorted(iteration_times.keys())
    avg_times = [np.mean(iteration_times[i]) for i in iterations]
    std_times = [np.std(iteration_times[i]) for i in iterations]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iterations, avg_times, marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax.fill_between(iterations, 
                     np.array(avg_times) - np.array(std_times),
                     np.array(avg_times) + np.array(std_times),
                     alpha=0.3, color='#A23B72')
    ax.set_xlabel('Iteration Number', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Average Execution Time per Iteration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'iteration_time_trends.png', dpi=300)
    plt.close()


def plot_tool_sequence_heatmap(pathway_data: List[Dict], output_dir: Path):
    tool_sequences = []
    
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            tools = iteration["tools_selected"]
            if tools:
                tool_sequences.append(tuple(tools))
    
    sequence_counts = Counter(tool_sequences)
    top_sequences = sequence_counts.most_common(15)
    
    if not top_sequences:
        return
    
    sequences = [' -> '.join(seq) for seq, _ in top_sequences]
    counts = [count for _, count in top_sequences]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    y_pos = np.arange(len(sequences))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sequences)))
    ax.barh(y_pos, counts, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sequences, fontsize=9)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_title('Top 15 Tool Sequence Patterns', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'tool_sequence_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_average_trajectory(pathway_data: List[Dict], output_dir: Path):
    """Plot the average trajectory across all problems"""
    if not pathway_data:
        return
    
    # Collect data by iteration number across all problems
    iteration_tool_counts = defaultdict(lambda: defaultdict(int))
    iteration_strategy_counts = defaultdict(lambda: defaultdict(int))
    iteration_params = defaultdict(list)
    iteration_totals = defaultdict(int)
    
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            iter_num = iteration["iteration"]
            iteration_totals[iter_num] += 1
            
            # Count tools
            tools = iteration["tools_selected"]
            tools_str = ', '.join(sorted(tools)) if tools else 'none'
            iteration_tool_counts[iter_num][tools_str] += 1
            
            # Count strategies and params
            for config in iteration["compute_configs"]:
                strategy = config.get("strategy", "none")
                param = config.get("param", 0)
                iteration_strategy_counts[iter_num][strategy] += 1
                iteration_params[iter_num].append(param)
    
    iterations = sorted(iteration_totals.keys())
    
    # Get most common tool combination per iteration
    most_common_tools = []
    tool_percentages = []
    for iter_num in iterations:
        tool_counts = iteration_tool_counts[iter_num]
        if tool_counts:
            most_common = max(tool_counts.items(), key=lambda x: x[1])
            most_common_tools.append(most_common[0])
            tool_percentages.append((most_common[1] / iteration_totals[iter_num]) * 100)
        else:
            most_common_tools.append('none')
            tool_percentages.append(0)
    
    # Get most common strategy per iteration
    most_common_strategies = []
    strategy_percentages = []
    for iter_num in iterations:
        strat_counts = iteration_strategy_counts[iter_num]
        if strat_counts:
            most_common = max(strat_counts.items(), key=lambda x: x[1])
            most_common_strategies.append(most_common[0])
            strategy_percentages.append((most_common[1] / iteration_totals[iter_num]) * 100)
        else:
            most_common_strategies.append('none')
            strategy_percentages.append(0)
    
    # Average parameters per iteration
    avg_params = [np.mean(iteration_params[i]) if iteration_params[i] else 0 for i in iterations]
    std_params = [np.std(iteration_params[i]) if iteration_params[i] else 0 for i in iterations]
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 14))
    
    # Plot 1: Most common tool combination per iteration
    unique_tools = list(set(most_common_tools))
    tool_to_num = {tool: i for i, tool in enumerate(unique_tools)}
    tool_nums = [tool_to_num[t] for t in most_common_tools]
    
    ax1.plot(iterations, tool_nums, marker='o', linewidth=2.5, markersize=9, color='#E63946', label='Most Common')
    ax1.set_yticks(range(len(unique_tools)))
    ax1.set_yticklabels(unique_tools, fontsize=9)
    ax1.set_xlabel('Iteration Number', fontsize=12)
    ax1.set_ylabel('Tool Selection', fontsize=12)
    ax1.set_title('Average Tool Selection Trajectory (Most Common per Iteration)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add percentage annotations
    for i, (iter_num, pct) in enumerate(zip(iterations, tool_percentages)):
        ax1.annotate(f'{pct:.1f}%', (iter_num, tool_nums[i]), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=8, color='#E63946')
    
    # Plot 2: Most common strategy per iteration
    unique_strats = list(set(most_common_strategies))
    strat_to_num = {s: i for i, s in enumerate(unique_strats)}
    strat_nums = [strat_to_num[s] for s in most_common_strategies]
    
    ax2.plot(iterations, strat_nums, marker='s', linewidth=2.5, markersize=9, color='#457B9D', label='Most Common')
    ax2.set_yticks(range(len(unique_strats)))
    ax2.set_yticklabels(unique_strats, fontsize=9)
    ax2.set_xlabel('Iteration Number', fontsize=12)
    ax2.set_ylabel('Compute Strategy', fontsize=12)
    ax2.set_title('Average Compute Strategy Trajectory (Most Common per Iteration)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add percentage annotations
    for i, (iter_num, pct) in enumerate(zip(iterations, strategy_percentages)):
        ax2.annotate(f'{pct:.1f}%', (iter_num, strat_nums[i]), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=8, color='#457B9D')
    
    # Plot 3: Average compute parameters
    ax3.plot(iterations, avg_params, marker='^', linewidth=2.5, markersize=9, color='#F1A208', label='Mean')
    ax3.fill_between(iterations, 
                     np.array(avg_params) - np.array(std_params),
                     np.array(avg_params) + np.array(std_params),
                     alpha=0.3, color='#F1A208', label='±1 Std Dev')
    ax3.set_xlabel('Iteration Number', fontsize=12)
    ax3.set_ylabel('Compute Parameter Value', fontsize=12)
    ax3.set_title('Average Compute Parameter Trajectory', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'average_trajectory.png', dpi=300)
    plt.close()
    
    # Also create a comprehensive heatmap showing distribution across iterations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Tool distribution heatmap
    all_tool_combos = set()
    for iter_data in iteration_tool_counts.values():
        all_tool_combos.update(iter_data.keys())
    all_tool_combos = sorted(all_tool_combos)
    
    tool_matrix = np.zeros((len(iterations), len(all_tool_combos)))
    for i, iter_num in enumerate(iterations):
        for j, tool_combo in enumerate(all_tool_combos):
            count = iteration_tool_counts[iter_num].get(tool_combo, 0)
            tool_matrix[i, j] = (count / iteration_totals[iter_num]) * 100 if iteration_totals[iter_num] > 0 else 0
    
    im1 = ax1.imshow(tool_matrix.T, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax1.set_xticks(range(len(iterations)))
    ax1.set_xticklabels(iterations)
    ax1.set_yticks(range(len(all_tool_combos)))
    ax1.set_yticklabels(all_tool_combos, fontsize=8)
    ax1.set_xlabel('Iteration Number', fontsize=12)
    ax1.set_ylabel('Tool Combination', fontsize=12)
    ax1.set_title('Tool Selection Distribution Across Iterations (% of problems)', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Percentage (%)', fontsize=11)
    
    # Strategy distribution heatmap
    all_strategies = set()
    for iter_data in iteration_strategy_counts.values():
        all_strategies.update(iter_data.keys())
    all_strategies = sorted(all_strategies)
    
    strategy_matrix = np.zeros((len(iterations), len(all_strategies)))
    for i, iter_num in enumerate(iterations):
        for j, strategy in enumerate(all_strategies):
            count = iteration_strategy_counts[iter_num].get(strategy, 0)
            strategy_matrix[i, j] = (count / iteration_totals[iter_num]) * 100 if iteration_totals[iter_num] > 0 else 0
    
    im2 = ax2.imshow(strategy_matrix.T, cmap='Blues', aspect='auto', interpolation='nearest')
    ax2.set_xticks(range(len(iterations)))
    ax2.set_xticklabels(iterations)
    ax2.set_yticks(range(len(all_strategies)))
    ax2.set_yticklabels(all_strategies, fontsize=9)
    ax2.set_xlabel('Iteration Number', fontsize=12)
    ax2.set_ylabel('Compute Strategy', fontsize=12)
    ax2.set_title('Compute Strategy Distribution Across Iterations (% of problems)', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Percentage (%)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_distribution_heatmap.png', dpi=300)
    plt.close()


def plot_problem_trajectory(pathway_data: List[Dict], problem_idx: int, output_dir: Path):
    if problem_idx >= len(pathway_data):
        return
    
    problem = pathway_data[problem_idx]
    iterations = problem["iterations"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    iter_nums = [it["iteration"] for it in iterations]
    
    tools_by_iter = []
    for it in iterations:
        tools = it["tools_selected"]
        tools_str = ', '.join(tools) if tools else 'none'
        tools_by_iter.append(tools_str)
    
    unique_tool_combos = list(set(tools_by_iter))
    tool_to_num = {tool: i for i, tool in enumerate(unique_tool_combos)}
    tool_nums = [tool_to_num[t] for t in tools_by_iter]
    
    ax1.plot(iter_nums, tool_nums, marker='o', linewidth=2, markersize=8, color='#E63946')
    ax1.set_yticks(range(len(unique_tool_combos)))
    ax1.set_yticklabels(unique_tool_combos, fontsize=8)
    ax1.set_xlabel('Iteration Number', fontsize=11)
    ax1.set_ylabel('Tool Selection', fontsize=11)
    ax1.set_title(f'Problem {problem_idx + 1}: Tool Selection Trajectory', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    compute_strats = []
    compute_params = []
    for it in iterations:
        configs = it["compute_configs"]
        if configs:
            strat = configs[0].get("strategy", "none")
            param = configs[0].get("param", 0)
        else:
            strat = "none"
            param = 0
        compute_strats.append(strat)
        compute_params.append(param)
    
    unique_strats = list(set(compute_strats))
    strat_to_num = {s: i for i, s in enumerate(unique_strats)}
    strat_nums = [strat_to_num[s] for s in compute_strats]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(iter_nums, strat_nums, marker='s', linewidth=2, markersize=8, 
                     color='#457B9D', label='Compute Strategy')
    ax2.set_yticks(range(len(unique_strats)))
    ax2.set_yticklabels(unique_strats, fontsize=8)
    ax2.set_ylabel('Compute Strategy', fontsize=11, color='#457B9D')
    ax2.tick_params(axis='y', labelcolor='#457B9D')
    
    line2 = ax2_twin.plot(iter_nums, compute_params, marker='^', linewidth=2, markersize=8,
                          color='#F1A208', label='Compute Param')
    ax2_twin.set_ylabel('Compute Parameter', fontsize=11, color='#F1A208')
    ax2_twin.tick_params(axis='y', labelcolor='#F1A208')
    
    ax2.set_xlabel('Iteration Number', fontsize=11)
    ax2.set_title(f'Problem {problem_idx + 1}: Compute Configuration Trajectory', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'problem_{problem_idx + 1}_trajectory.png', dpi=300)
    plt.close()


def plot_plan_usage_distribution(pathway_data: List[Dict], output_dir: Path):
    with_plan = 0
    without_plan = 0
    
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            if iteration.get("plan"):
                with_plan += 1
            else:
                without_plan += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['With Plan', 'Without Plan']
    sizes = [with_plan, without_plan]
    colors = ['#06FFA5', '#FF6B6B']
    explode = (0.05, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12})
    ax.set_title('Plan Usage Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'plan_usage_distribution.png', dpi=300)
    plt.close()


def plot_strategy_param_heatmap(pathway_data: List[Dict], output_dir: Path):
    strategy_param_matrix = defaultdict(lambda: defaultdict(int))
    
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            for config in iteration["compute_configs"]:
                strategy = config.get("strategy", "unknown")
                param = config.get("param", 0)
                strategy_param_matrix[strategy][param] += 1
    
    strategies = sorted(strategy_param_matrix.keys())
    all_params = set()
    for params_dict in strategy_param_matrix.values():
        all_params.update(params_dict.keys())
    params = sorted(all_params)
    
    matrix = np.zeros((len(strategies), len(params)))
    for i, strategy in enumerate(strategies):
        for j, param in enumerate(params):
            matrix[i, j] = strategy_param_matrix[strategy].get(param, 0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(params)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels(params)
    ax.set_yticklabels(strategies)
    
    ax.set_xlabel('Parameter Value', fontsize=12)
    ax.set_ylabel('Compute Strategy', fontsize=12)
    ax.set_title('Strategy-Parameter Frequency Heatmap', fontsize=14, fontweight='bold')
    
    for i in range(len(strategies)):
        for j in range(len(params)):
            text = ax.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frequency', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'strategy_param_heatmap.png', dpi=300)
    plt.close()


def plot_problem_type_tool_correlation(pathway_data: List[Dict], output_dir: Path):
    problem_type_tools = defaultdict(lambda: defaultdict(int))
    
    for problem in pathway_data:
        problem_type = problem.get("problem_type", "unknown")
        for iteration in problem["iterations"]:
            for tool in iteration["tools_selected"]:
                problem_type_tools[problem_type][tool] += 1
    
    problem_types = sorted(problem_type_tools.keys())
    all_tools = set()
    for tools_dict in problem_type_tools.values():
        all_tools.update(tools_dict.keys())
    tools = sorted(all_tools)
    
    matrix = np.zeros((len(problem_types), len(tools)))
    for i, ptype in enumerate(problem_types):
        for j, tool in enumerate(tools):
            matrix[i, j] = problem_type_tools[ptype].get(tool, 0)
    
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_normalized = matrix / row_sums * 100
    
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(matrix_normalized, cmap='Blues', aspect='auto')
    
    ax.set_xticks(np.arange(len(tools)))
    ax.set_yticks(np.arange(len(problem_types)))
    ax.set_xticklabels(tools, rotation=45, ha='right')
    ax.set_yticklabels(problem_types)
    
    ax.set_xlabel('Tool', fontsize=12)
    ax.set_ylabel('Problem Type', fontsize=12)
    ax.set_title('Tool Usage by Problem Type (Percentage)', fontsize=14, fontweight='bold')
    
    for i in range(len(problem_types)):
        for j in range(len(tools)):
            text = ax.text(j, i, f'{matrix_normalized[i, j]:.1f}%',
                          ha="center", va="center", 
                          color="white" if matrix_normalized[i, j] > 50 else "black",
                          fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'problem_type_tool_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_trajectory_report(pathway_data: List[Dict], output_dir: Path):
    num_problems = len(pathway_data)
    num_iterations = len(pathway_data[0]["iterations"]) if pathway_data else 0
    total_iterations = num_problems * num_iterations
    
    all_tools = []
    all_strategies = []
    all_params = []
    all_times = []
    plan_count = 0
    
    for problem in pathway_data:
        for iteration in problem["iterations"]:
            all_tools.extend(iteration["tools_selected"])
            all_times.append(iteration["time_seconds"])
            if iteration.get("plan"):
                plan_count += 1
            
            for config in iteration["compute_configs"]:
                all_strategies.append(config.get("strategy", "unknown"))
                all_params.append(config.get("param", 0))
    
    tool_counts = Counter(all_tools)
    strategy_counts = Counter(all_strategies)
    
    report = []
    report.append("=" * 80)
    report.append("TRAJECTORY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nDataset Statistics:")
    report.append(f"  Total Problems: {num_problems}")
    report.append(f"  Iterations per Problem: {num_iterations}")
    report.append(f"  Total Iterations: {total_iterations}")
    
    report.append(f"\n{'-' * 80}")
    report.append("Tool Usage Statistics:")
    report.append(f"{'-' * 80}")
    for tool, count in tool_counts.most_common():
        percentage = (count / len(all_tools)) * 100 if all_tools else 0
        report.append(f"  {tool:25s}: {count:6d} times ({percentage:5.2f}%)")
    
    report.append(f"\n{'-' * 80}")
    report.append("Compute Strategy Statistics:")
    report.append(f"{'-' * 80}")
    for strategy, count in strategy_counts.most_common():
        percentage = (count / len(all_strategies)) * 100 if all_strategies else 0
        report.append(f"  {strategy:25s}: {count:6d} times ({percentage:5.2f}%)")
    
    report.append(f"\n{'-' * 80}")
    report.append("Compute Parameter Statistics:")
    report.append(f"{'-' * 80}")
    if all_params:
        report.append(f"  Mean Parameter Value: {np.mean(all_params):.2f}")
        report.append(f"  Median Parameter Value: {np.median(all_params):.2f}")
        report.append(f"  Min Parameter Value: {np.min(all_params):.2f}")
        report.append(f"  Max Parameter Value: {np.max(all_params):.2f}")
        report.append(f"  Std Dev Parameter Value: {np.std(all_params):.2f}")
    
    report.append(f"\n{'-' * 80}")
    report.append("Timing Statistics:")
    report.append(f"{'-' * 80}")
    if all_times:
        report.append(f"  Mean Iteration Time: {np.mean(all_times):.2f} seconds")
        report.append(f"  Median Iteration Time: {np.median(all_times):.2f} seconds")
        report.append(f"  Min Iteration Time: {np.min(all_times):.2f} seconds")
        report.append(f"  Max Iteration Time: {np.max(all_times):.2f} seconds")
        report.append(f"  Total Time: {np.sum(all_times):.2f} seconds")
    
    report.append(f"\n{'-' * 80}")
    report.append("Planning Statistics:")
    report.append(f"{'-' * 80}")
    plan_percentage = (plan_count / total_iterations) * 100 if total_iterations else 0
    report.append(f"  Iterations with Plan: {plan_count} ({plan_percentage:.2f}%)")
    report.append(f"  Iterations without Plan: {total_iterations - plan_count} ({100 - plan_percentage:.2f}%)")
    
    report.append(f"\n{'=' * 80}\n")
    
    report_text = '\n'.join(report)
    
    with open(output_dir / 'trajectory_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)


def visualize_trajectories(pathway_file: str, output_dir: str, num_sample_problems: int = 5):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading pathway data from: {pathway_file}")
    pathway_data = load_pathway_data(pathway_file)
    
    print(f"\nGenerating visualizations...")
    
    print("  - Tool selection distribution...")
    plot_tool_selection_distribution(pathway_data, output_path)
    
    print("  - Compute strategy distribution...")
    plot_compute_strategy_distribution(pathway_data, output_path)
    
    print("  - Compute parameter distribution...")
    plot_compute_param_distribution(pathway_data, output_path)
    
    print("  - Iteration time trends...")
    plot_iteration_time_trends(pathway_data, output_path)
    
    print("  - Tool sequence patterns...")
    plot_tool_sequence_heatmap(pathway_data, output_path)
    
    print("  - Plan usage distribution...")
    plot_plan_usage_distribution(pathway_data, output_path)
    
    print("  - Strategy-parameter heatmap...")
    plot_strategy_param_heatmap(pathway_data, output_path)
    
    print("  - Problem type-tool correlation...")
    plot_problem_type_tool_correlation(pathway_data, output_path)
    
    print("  - Average trajectory across all problems...")
    plot_average_trajectory(pathway_data, output_path)
    
    print(f"\n  - Individual problem trajectories (sampling {num_sample_problems} problems)...")
    num_problems = len(pathway_data)
    sample_indices = np.linspace(0, num_problems - 1, min(num_sample_problems, num_problems), dtype=int)
    
    for idx in sample_indices:
        plot_problem_trajectory(pathway_data, idx, output_path)
    
    print("\nGenerating trajectory report...")
    generate_trajectory_report(pathway_data, output_path)
    
    print(f"\n{'=' * 80}")
    print(f"All visualizations saved to: {output_path}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize Agent Trajectory Pathways")
    parser.add_argument(
        "--pathway_file",
        type=str,
        default="./all_saved_results/llama3.1_8b_instruct/dynamic_results/math_mode-dynamic_planner-True_toolsel-True_computesel-True_pathway.json",
        help="Path to the pathway JSON file or results_final.json file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualization plots"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample problems to visualize individual trajectories"
    )
    
    args = parser.parse_args()
    
    visualize_trajectories(
        pathway_file=args.pathway_file,
        output_dir=args.output_dir,
        num_sample_problems=args.num_samples
    )


if __name__ == "__main__":
    main()