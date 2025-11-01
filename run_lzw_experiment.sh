#!/bin/bash
# run_lzw_experiment.sh
# Convenience script to run the full LZW compression experiment pipeline

set -e  # Exit on error

echo "========================================"
echo "LZW Compression Strategy Experiment"
echo "========================================"
echo ""

# Check if config file exists
CONFIG_FILE="${1:-config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"
echo ""

# Parse output directory from config
OUTPUT_DIR=$(grep "out_dir:" "$CONFIG_FILE" | awk '{print $2}')
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run the experiment
echo "Step 1: Running experiment with all strategies..."
echo "This may take several minutes depending on your hardware..."
echo ""

python experiment_lzw_strategies.py \
    --config "$CONFIG_FILE" \
    --strategies lowest median highest sweet_spot

echo ""
echo "========================================"
echo "Experiment completed!"
echo "========================================"
echo ""

# Check if results exist
if [ ! -f "$OUTPUT_DIR/combined_results.json" ]; then
    echo "Error: Results file not found at $OUTPUT_DIR/combined_results.json"
    exit 1
fi

# Display quick summary
echo "Quick Results Summary:"
echo "----------------------"
python -c "
import json
with open('$OUTPUT_DIR/combined_results.json') as f:
    results = json.load(f)
print(f\"Dataset: {results['dataset']}\")
print(f\"Sample size: {results['limit']}\")
print(f\"K (candidates): {results['K']}\")
print(f\"Max steps: {results['max_steps']}\")
print()
print('Strategy Results:')
print('-' * 50)
for r in results['strategies']:
    print(f\"{r['strategy']:15s} | Acc: {r['accuracy']:.4f} | LZW: {r['avg_lzw_ratio']:.4f}\")
"
echo ""

# Generate visualizations
echo "Step 2: Generating visualizations..."
echo ""

python visualize_lzw_experiment.py --results_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "All done!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - combined_results.json"
echo "  - accuracy_by_strategy.png"
echo "  - lzw_ratio_by_strategy.png"
echo "  - accuracy_vs_lzw.png"
echo "  - combined_comparison.png"
echo "  - lzw_distribution.png"
echo "  - summary_table.png"
echo ""
echo "View the visualizations to see the relationship between"
echo "LZW compression ratio and accuracy!"
