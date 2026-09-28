#!/bin/bash

################################################################################
# Fixed-configuration ablation: tool x compute strategy x parameter x iterations (Table 2).
################################################################################
# nohup ./run_ablation_qwen.sh > ablation_main_qwen.log 2>&1 &


MAX_PARALLEL=1

NUM_ITERATIONS=(5 10)
FIXED_TOOLS=("cot" "self_reflection")
STRATEGIES=("best_of_n" "beam_search" "lookahead")
PARAMS=(1 5 10)

DATASET="math"

FILTER_TOOL=""
FILTER_STRATEGY=""
FILTER_PARAM=""

################################################################################

CONFIG_DIR="./configs_ablation_qwen_math"
LOG_DIR="./logs_ablation_qwen_math"
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

generate_config() {
    local num_iter=$1
    local tool=$2
    local strategy=$3
    local param=$4
    local config_file=$5
    
    cat > "$config_file" <<EOF
agent:
  mode: fixed_tool_compute
  use_planner: false
  use_tool_selector: false
  use_compute_selector: false
  remember_cot: true
  max_text_history: 16
  max_image_history: 0
  fixed_tool: $tool
  fixed_compute:
    strategy: $strategy
    param: $param
  planning_frequency: null
  domain: math

eval:
  num_workers: 16
  output_dir: fixed_tool_compute_qwen_${tool}_${strategy}_${param}_iteration_${num_iter}
  max_problems: 100
  num_iterations: $num_iter
  use_prm_selection: true
  prm_selection_metric: "mean_reward"
  split: test
  problem_types: null
  difficulty_levels: null

client:
  client_name: transformer
  base_url: http://localhost4:11434/v1
  model_id: Qwen/Qwen2.5-7B-Instruct
  generate_kwargs:
    temperature: 0.3
    top_k: 20
    max_tokens: 1024
  timeout: 60
  max_retries: 5
  delay: 2
  alternate_roles: false
EOF
}

run_experiment() {
    local config_file=$1
    local log_file=$2
    local exp_name=$3
    local output_dir=$4
    
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] Starting: $exp_name${NC}"
    
    # Create the output directory before running
    mkdir -p "$output_dir"
    
    python main.py --config "$config_file" --dataset "$DATASET" > "$log_file" 2>&1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ Completed: $exp_name${NC}"
    else
        echo -e "${RED}[$(date '+%H:%M:%S')] ✗ Failed: $exp_name (exit code: $exit_code)${NC}"
        echo -e "${RED}    Check log: $log_file${NC}"
    fi
    
    return $exit_code
}

# Generate experiments
experiments=()
for num_iter in "${NUM_ITERATIONS[@]}"; do
    for tool in "${FIXED_TOOLS[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            for param in "${PARAMS[@]}"; do
                [[ -n "$FILTER_TOOL" && "$tool" != "$FILTER_TOOL" ]] && continue
                [[ -n "$FILTER_STRATEGY" && "$strategy" != "$FILTER_STRATEGY" ]] && continue
                [[ -n "$FILTER_PARAM" && "$param" != "$FILTER_PARAM" ]] && continue
                
                exp_name="${tool}_${strategy}_${param}_iter${num_iter}"
                config_file="${CONFIG_DIR}/config_${exp_name}.yaml"
                log_file="${LOG_DIR}/log_${exp_name}.txt"
                output_dir="fixed_tool_compute_qwen_${tool}_${strategy}_${param}_iteration_${num_iter}"
                
                generate_config "$num_iter" "$tool" "$strategy" "$param" "$config_file"
                experiments+=("$config_file|$log_file|$exp_name|$output_dir")
            done
        done
    done
done

total=${#experiments[@]}
echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  ABLATION STUDY: $total experiments on $DATASET dataset${NC}"
echo -e "${YELLOW}  Running with $MAX_PARALLEL parallel processes${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}"
echo ""

start_time=$(date +%s)
current_idx=0

wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
}

# Launch experiments
for exp in "${experiments[@]}"; do
    IFS='|' read -r config_file log_file exp_name output_dir <<< "$exp"
    
    wait_for_slot
    run_experiment "$config_file" "$log_file" "$exp_name" "$output_dir" &
    
    current_idx=$((current_idx + 1))
    echo "[$current_idx/$total] Launched: $exp_name"
done

echo ""
echo "Waiting for all experiments to complete..."
wait

end_time=$(date +%s)
duration=$((end_time - start_time))

# Count results
completed=0
failed=0
for exp in "${experiments[@]}"; do
    IFS='|' read -r _ log_file _ _ <<< "$exp"
    if tail -n 50 "$log_file" 2>/dev/null | grep -q "EVALUATION COMPLETE"; then
        completed=$((completed + 1))
    else
        failed=$((failed + 1))
    fi
done

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ABLATION STUDY COMPLETE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo "Total: $total | Completed: $completed | Failed: $failed"
echo "Duration: $(($duration / 60))m $(($duration % 60))s"
echo "Logs: $LOG_DIR"
echo ""

# Show failed experiments if any
if [ $failed -gt 0 ]; then
    echo -e "${RED}Failed Experiments:${NC}"
    for exp in "${experiments[@]}"; do
        IFS='|' read -r _ log_file exp_name _ <<< "$exp"
        if ! tail -n 50 "$log_file" 2>/dev/null | grep -q "EVALUATION COMPLETE"; then
            echo -e "${RED}  - $exp_name${NC}"
            echo "    Last error: $(tail -n 5 "$log_file" 2>/dev/null | grep -i error | head -n 1)"
        fi
    done
    echo ""
fi