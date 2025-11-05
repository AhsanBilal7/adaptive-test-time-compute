#!/usr/bin/env python3
"""
Visualization script for tool-chain frequency analysis.

Analyzes JSON outputs from evaluator to plot:
1. Tool selection frequency
2. Most common tool chains
3. Tool chain performance correlation
4. Token/latency breakdown by tool
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir):
    """Load all JSON result files from directory."""
    results = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if 'tools_used' in data:  # Only include runs with tool data
                            results.append(data)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    return results


def analyze_tool_frequency(results):
    """Analyze individual tool selection frequency."""
    tool_counts = Counter()
    for result in results:
        tools = result.get('tools_used', [])
        tool_counts.update(tools)
    return tool_counts


def analyze_tool_chains(results):
    """Analyze complete tool chains (sequences)."""
    chain_counts = Counter()
    chain_rewards = defaultdict(list)
    
    for result in results:
        tools = result.get('tools_used', [])
        if tools:
            chain = ' → '.join(tools)
            chain_counts[chain] += 1
            reward = result.get('episode_return', 0)
            chain_rewards[chain].append(reward)
    
    return chain_counts, chain_rewards


def analyze_tool_performance(results):
    """Analyze performance metrics per tool."""
    tool_tokens = defaultdict(list)
    tool_latency = defaultdict(list)
    
    for result in results:
        metadata = result.get('tool_metadata', [])
        for m in metadata:
            tool = m.get('tool', 'unknown')
            tool_tokens[tool].append(m.get('tokens_used', 0))
            tool_latency[tool].append(m.get('latency_ms', 0))
    
    return tool_tokens, tool_latency


def plot_tool_frequency(tool_counts, output_dir):
    """Plot tool selection frequency."""
    if not tool_counts:
        print("No tool frequency data to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    tools = list(tool_counts.keys())
    counts = list(tool_counts.values())
    
    plt.bar(tools, counts, color='steelblue', alpha=0.8)
    plt.xlabel('Tool', fontsize=12)
    plt.ylabel('Selection Frequency', fontsize=12)
    plt.title('Tool Selection Frequency', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'tool_frequency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_tool_chains(chain_counts, chain_rewards, output_dir, top_n=10):
    """Plot most common tool chains and their performance."""
    if not chain_counts:
        print("No tool chain data to plot.")
        return
    
    # Get top N chains
    top_chains = chain_counts.most_common(top_n)
    chains = [c[0] for c in top_chains]
    counts = [c[1] for c in top_chains]
    
    # Calculate average reward per chain
    avg_rewards = [sum(chain_rewards[c]) / len(chain_rewards[c]) if chain_rewards[c] else 0 for c in chains]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot chain frequency
    ax1.barh(range(len(chains)), counts, color='coral', alpha=0.8)
    ax1.set_yticks(range(len(chains)))
    ax1.set_yticklabels(chains, fontsize=9)
    ax1.set_xlabel('Frequency', fontsize=12)
    ax1.set_title(f'Top {top_n} Tool Chains by Frequency', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # Plot chain performance
    colors = ['green' if r > 0 else 'red' for r in avg_rewards]
    ax2.barh(range(len(chains)), avg_rewards, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(chains)))
    ax2.set_yticklabels(chains, fontsize=9)
    ax2.set_xlabel('Average Episode Return', fontsize=12)
    ax2.set_title(f'Average Performance by Tool Chain', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'tool_chains.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_tool_metrics(tool_tokens, tool_latency, output_dir):
    """Plot token usage and latency per tool."""
    if not tool_tokens or not tool_latency:
        print("No tool metrics data to plot.")
        return
    
    tools = list(tool_tokens.keys())
    avg_tokens = [sum(tool_tokens[t]) / len(tool_tokens[t]) if tool_tokens[t] else 0 for t in tools]
    avg_latency = [sum(tool_latency[t]) / len(tool_latency[t]) if tool_latency[t] else 0 for t in tools]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot average tokens
    ax1.bar(tools, avg_tokens, color='mediumseagreen', alpha=0.8)
    ax1.set_xlabel('Tool', fontsize=12)
    ax1.set_ylabel('Average Tokens Used', fontsize=12)
    ax1.set_title('Token Usage by Tool', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot average latency
    ax2.bar(tools, avg_latency, color='darkorange', alpha=0.8)
    ax2.set_xlabel('Tool', fontsize=12)
    ax2.set_ylabel('Average Latency (ms)', fontsize=12)
    ax2.set_title('Latency by Tool', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'tool_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_stats(results, tool_counts, chain_counts):
    """Print summary statistics to console."""
    print("\n" + "="*60)
    print("TOOL CHAIN ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nTotal episodes analyzed: {len(results)}")
    print(f"Total tool invocations: {sum(tool_counts.values())}")
    print(f"Unique tools used: {len(tool_counts)}")
    print(f"Unique tool chains: {len(chain_counts)}")
    
    print("\nMost common tools:")
    for tool, count in tool_counts.most_common(5):
        pct = (count / sum(tool_counts.values())) * 100
        print(f"  {tool:20s}: {count:4d} ({pct:5.1f}%)")
    
    print("\nMost common tool chains:")
    for chain, count in chain_counts.most_common(5):
        pct = (count / len(results)) * 100
        print(f"  {chain:50s}: {count:3d} ({pct:5.1f}%)")
    
    print("\n" + "="*60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_tool_chains.py <results_directory> [output_directory]")
        print("\nExample: python visualize_tool_chains.py ./results ./visualizations")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './visualizations'
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    if not results:
        print("No results with tool data found. Make sure episodes have 'tools_used' logged.")
        sys.exit(1)
    
    print(f"Loaded {len(results)} episodes with tool data.")
    
    # Analyze data
    tool_counts = analyze_tool_frequency(results)
    chain_counts, chain_rewards = analyze_tool_chains(results)
    tool_tokens, tool_latency = analyze_tool_performance(results)
    
    # Print summary
    print_summary_stats(results, tool_counts, chain_counts)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_tool_frequency(tool_counts, output_dir)
    plot_tool_chains(chain_counts, chain_rewards, output_dir)
    plot_tool_metrics(tool_tokens, tool_latency, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()