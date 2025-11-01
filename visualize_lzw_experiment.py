# visualize_lzw_experiment.py
"""
Visualization script for LZW compression ratio vs accuracy experiments.
Creates plots showing the relationship between selection strategies and performance.
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_combined_results(results_path: Path):
    """Load the combined results JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def plot_accuracy_by_strategy(results, output_dir: Path):
    """Bar plot of accuracy for each selection strategy."""
    strategies = [r["strategy"] for r in results["strategies"]]
    accuracies = [r["accuracy"] for r in results["strategies"]]
    
    # Color mapping
    colors = {
        'lowest': '#e74c3c',     # Red - most compressible
        'median': '#3498db',     # Blue - middle
        'highest': '#2ecc71',    # Green - least compressible  
        'sweet_spot': '#f39c12'  # Orange - sweet spot
    }
    bar_colors = [colors.get(s, '#95a5a6') for s in strategies]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, accuracies, color=bar_colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Selection Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Accuracy by LZW Selection Strategy\n({results["dataset"]}, n={results["limit"]})', 
              fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_strategy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_by_strategy.png'}")


def plot_lzw_ratio_by_strategy(results, output_dir: Path):
    """Bar plot of average LZW ratio for each selection strategy."""
    strategies = [r["strategy"] for r in results["strategies"]]
    avg_lzw = [r["avg_lzw_ratio"] for r in results["strategies"]]
    
    colors = {
        'lowest': '#e74c3c',
        'median': '#3498db',
        'highest': '#2ecc71',
        'sweet_spot': '#f39c12'
    }
    bar_colors = [colors.get(s, '#95a5a6') for s in strategies]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, avg_lzw, color=bar_colors, alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Selection Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Average LZW Compression Ratio', fontsize=12, fontweight='bold')
    plt.title(f'Average LZW Ratio by Selection Strategy\n({results["dataset"]}, n={results["limit"]})', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lzw_ratio_by_strategy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'lzw_ratio_by_strategy.png'}")


def plot_accuracy_vs_lzw(results, output_dir: Path):
    """Scatter plot of accuracy vs average LZW ratio."""
    strategies = [r["strategy"] for r in results["strategies"]]
    accuracies = [r["accuracy"] for r in results["strategies"]]
    avg_lzw = [r["avg_lzw_ratio"] for r in results["strategies"]]
    
    colors = {
        'lowest': '#e74c3c',
        'median': '#3498db',
        'highest': '#2ecc71',
        'sweet_spot': '#f39c12'
    }
    
    plt.figure(figsize=(10, 8))
    
    for strategy, acc, lzw in zip(strategies, accuracies, avg_lzw):
        color = colors.get(strategy, '#95a5a6')
        plt.scatter(lzw, acc, s=300, c=color, alpha=0.7, edgecolor='black', linewidth=2, label=strategy)
        plt.annotate(strategy, (lzw, acc), xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    # Fit and plot trend line
    z = np.polyfit(avg_lzw, accuracies, 2)  # Quadratic fit
    p = np.poly1d(z)
    lzw_range = np.linspace(min(avg_lzw) - 0.1, max(avg_lzw) + 0.1, 100)
    plt.plot(lzw_range, p(lzw_range), "k--", alpha=0.5, linewidth=2, label='Trend (quadratic)')
    
    plt.xlabel('Average LZW Compression Ratio', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title(f'Accuracy vs LZW Compression Ratio\n({results["dataset"]}, n={results["limit"]})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_vs_lzw.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_vs_lzw.png'}")


def plot_combined_comparison(results, output_dir: Path):
    """Combined plot showing both accuracy and LZW ratios side by side."""
    strategies = [r["strategy"] for r in results["strategies"]]
    accuracies = [r["accuracy"] for r in results["strategies"]]
    avg_lzw = [r["avg_lzw_ratio"] for r in results["strategies"]]
    
    colors = {
        'lowest': '#e74c3c',
        'median': '#3498db',
        'highest': '#2ecc71',
        'sweet_spot': '#f39c12'
    }
    bar_colors = [colors.get(s, '#95a5a6') for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy subplot
    bars1 = ax1.bar(strategies, accuracies, color=bar_colors, alpha=0.8, edgecolor='black')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    ax1.set_xlabel('Selection Strategy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy by Strategy', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # LZW ratio subplot
    bars2 = ax2.bar(strategies, avg_lzw, color=bar_colors, alpha=0.8, edgecolor='black')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    ax2.set_xlabel('Selection Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average LZW Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Average LZW Compression Ratio', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle(f'LZW Selection Strategy Comparison\n({results["dataset"]}, n={results["limit"]}, K={results["K"]})', 
                 fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'combined_comparison.png'}")


def plot_lzw_distribution_by_strategy(base_output_dir: Path):
    """Box plot showing distribution of LZW ratios for each strategy."""
    strategies = []
    all_ratios = []
    
    # Load traces for each strategy
    for strategy_dir in base_output_dir.iterdir():
        if strategy_dir.is_dir() and strategy_dir.name.startswith("strategy_"):
            strategy = strategy_dir.name.replace("strategy_", "")
            traces_file = strategy_dir / "traces.jsonl"
            
            if traces_file.exists():
                ratios = []
                with open(traces_file, "r") as f:
                    for line in f:
                        trace = json.loads(line)
                        if "lzw_ratios" in trace:
                            ratios.extend(trace["lzw_ratios"])
                
                if ratios:
                    strategies.append(strategy)
                    all_ratios.append(ratios)
    
    if not strategies:
        print("Warning: No trace files found for distribution plot")
        return
    
    colors = {
        'lowest': '#e74c3c',
        'median': '#3498db',
        'highest': '#2ecc71',
        'sweet_spot': '#f39c12'
    }
    box_colors = [colors.get(s, '#95a5a6') for s in strategies]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bp = ax.boxplot(all_ratios, labels=strategies, patch_artist=True,
                     boxprops=dict(linewidth=2),
                     medianprops=dict(linewidth=2, color='darkred'),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Selection Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('LZW Compression Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of LZW Ratios by Selection Strategy', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(base_output_dir / 'lzw_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {base_output_dir / 'lzw_distribution.png'}")


def create_summary_table(results, output_dir: Path):
    """Create a formatted summary table and save as image."""
    strategies = [r["strategy"] for r in results["strategies"]]
    accuracies = [r["accuracy"] for r in results["strategies"]]
    avg_lzw = [r["avg_lzw_ratio"] for r in results["strategies"]]
    min_lzw = [r["min_lzw_ratio"] for r in results["strategies"]]
    max_lzw = [r["max_lzw_ratio"] for r in results["strategies"]]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Strategy', 'Accuracy', 'Avg LZW', 'Min LZW', 'Max LZW'])
    
    for i in range(len(strategies)):
        table_data.append([
            strategies[i].upper(),
            f'{accuracies[i]:.4f}',
            f'{avg_lzw[i]:.4f}',
            f'{min_lzw[i]:.4f}',
            f'{max_lzw[i]:.4f}'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows based on strategy
    colors = {
        'LOWEST': '#e74c3c',
        'MEDIAN': '#3498db',
        'HIGHEST': '#2ecc71',
        'SWEET_SPOT': '#f39c12'
    }
    
    for i in range(1, len(table_data)):
        strategy = table_data[i][0]
        color = colors.get(strategy, '#95a5a6')
        for j in range(5):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_alpha(0.3)
    
    plt.title(f'Summary: LZW Selection Strategy Performance\n{results["dataset"]}, n={results["limit"]}, K={results["K"]}',
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'summary_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize LZW experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Path to the output directory containing combined_results.json")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_file = results_dir / "combined_results.json"
    
    if not results_file.exists():
        print(f"Error: {results_file} not found!")
        return
    
    print(f"Loading results from: {results_file}")
    results = load_combined_results(results_file)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create all plots
    print("\nGenerating visualizations...")
    plot_accuracy_by_strategy(results, results_dir)
    plot_lzw_ratio_by_strategy(results, results_dir)
    plot_accuracy_vs_lzw(results, results_dir)
    plot_combined_comparison(results, results_dir)
    plot_lzw_distribution_by_strategy(results_dir)
    create_summary_table(results, results_dir)
    
    print(f"\nAll visualizations saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - accuracy_by_strategy.png")
    print("  - lzw_ratio_by_strategy.png")
    print("  - accuracy_vs_lzw.png")
    print("  - combined_comparison.png")
    print("  - lzw_distribution.png")
    print("  - summary_table.png")


if __name__ == "__main__":
    main()
