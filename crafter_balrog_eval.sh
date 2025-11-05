

# cd BALROG

# echo ">>> Launching vLLM server for meta-llama/Llama-3.2-3B-Instruct on port 8080"
# vllm serve "meta-llama/Llama-3.2-3B-Instruct" --port 8080 --kv-cache-dtype fp8_e5m2   --gpu-memory-utilization 0.70   --max-num-batched-tokens 1024
# vllm serve "meta-llama/Llama-3.1-70B-Instruct" --port 8080 --kv-cache-dtype fp8_e5m2   --gpu-memory-utilization 0.70   --max-num-batched-tokens 1024


#####################################################################
# nohup vllm serve "meta-llama/Llama-3.1-8B-Instruct" \
#   --port 8080 \
#   --kv-cache-dtype fp8_e5m2 \
#   --gpu-memory-utilization 0.90 \
#   --max-num-batched-tokens 1024 > ./vllm_3b.out 2>&1 < /dev/null & echo $! > ./vllm_3b.pid
# nohup vllm serve "meta-llama/Llama-3.2-3B-Instruct" \
#   --port 8080 \
#   --kv-cache-dtype fp8_e5m2 \
#   --max-num-batched-tokens 1024 > ./vllm_3b.out 2>&1 < /dev/null & echo $! > ./vllm_3b.pid
# nohup vllm serve "meta-llama/Llama-3.2-1B-Instruct" \
#   --port 8080 \
#   --kv-cache-dtype fp8_e5m2 \
#   --gpu-memory-utilization 0.7 \
#   --max-num-batched-tokens 1024 > ./vllm_3b.out 2>&1 < /dev/null & echo $! > ./vllm_3b.pid
# nohup vllm serve "meta-llama/Llama-3.2-3B-Instruct" \
#   --port 8080 \
#   --kv-cache-dtype fp8_e5m2 \
#   --gpu-memory-utilization 0.90 \
#   --max-num-batched-tokens 1024 > ./vllm_3b.out 2>&1 < /dev/null & echo $! > ./vllm_3b.pid
# nohup vllm serve "Qwen/Qwen2-7B-Instruct" \
#   --host 0.0.0.0 --port 8000 \
#   --kv-cache-dtype fp8_e5m2 \
#   --gpu-memory-utilization 0.85 \
#   --max-model-len 32768   > ./vllm_7b.out 2>&1 < /dev/null & echo $! > ./vllm_7b.pid


# Follow logs: tail -f ./vllm_3b.out

# Stop later: kill $(cat ./vllm_3b.pid)
#####################################################################



# Naive Original
# echo ">>> Starting BALROG evaluation on Crafter with Llama-3.1-8B-Instruct (vLLM)"
# python eval.py \
#   envs.names=crafter \
#   agent.type=naive \
#   agent.remember_cot=true \
#   agent.max_text_history=16 \
#   agent.max_image_history=0 \
#   eval.num_workers=16 \
#   client.client_name=vllm \
#   client.model_id="meta-llama/Llama-3.2-3B-Instruct" \
#   client.base_url="http://0.0.0.0:8080/v1" 

# echo ">>> Done. Results saved under results/<DATE>_..._naive_Llama-3.1-8B-Instruct"
# echo "    vLLM logs: vllm_llama31_8b.log"




# Example 1: Dynamic Planning
# python eval.py \
#   agent.type=custom \
#   agent.mode=dynamic \
#   agent.remember_cot=true \
#   agent.max_text_history=16 \
#   agent.max_image_history=0 \
#   eval.num_workers=16 \
#   client.client_name=vllm \
#   client.model_id="meta-llama/Llama-3.2-3B-Instruct" \
#   client.base_url="http://0.0.0.0:8080/v1" 


# Example 2: Always Planning Baseline
# python eval.py \
#   envs.names=crafter\   
#   agent.type=custom \
#   agent.mode=fixed \
#   agent.remember_cot=true \
#   agent.max_text_history=16 \
#   agent.max_image_history=0 \
#   eval.num_workers=16 \
#   client.client_name=vllm \
#   client.model_id="meta-llama/Llama-3.2-3B-Instruct" \
#   client.base_url="http://0.0.0.0:8080/v1" 


# Example 3: Plan Every 16 Steps
# python eval.py \
#     envs.names=crafter \
#     agent.type=custom \
#     agent.mode=fixed \
#     agent.planning_frequency=16 \
#     agent.remember_cot=true \
#     agent.max_text_history=16 \
#     agent.max_image_history=0 \
#     eval.num_workers=16 \
#     client.client_name=vllm \
#     client.model_id="meta-llama/Llama-3.2-3B-Instruct" \
#     client.base_url="http://0.0.0.0:8080/v1" 


# Example 4: Plan Every K Steps
cd BALROG

# for K in 1 2 4 8 16; do
#     python eval.py \
#         envs.names=crafter \
#         agent.type=custom \
#         agent.mode=fixed \
#         agent.planning_frequency=$K \
#         agent.remember_cot=true \
#         agent.max_text_history=16 \
#         agent.max_image_history=0 \
#         eval.num_workers=16 \
#         eval.output_dir="results" \
#         client.client_name=vllm \
#         client.model_id="meta-llama/Llama-3.2-1B-Instruct" \
#         client.base_url="http://0.0.0.0:8080/v1" 
# done

# # Example 1: Dynamic Planning
# python eval.py \
#   envs.names=crafter \
#   agent.type=custom \
#   agent.mode=dynamic \
#   agent.remember_cot=true \
#   agent.max_text_history=16 \
#   agent.max_image_history=0 \
#   eval.num_workers=16 \
#   eval.output_dir="results" \
#   client.client_name=vllm \
#   client.model_id="meta-llama/Llama-3.2-1B-Instruct" \
#   client.base_url="http://0.0.0.0:8080/v1" 


# nohup ./crafter_balrog_eval.sh \
#   >> ./crafter_eval.out 2>&1 < /dev/null & echo $! > ./crafter_eval.pid



# vllm serve "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" --host 0.0.0.0 --port 8000   --enforce-eager --gpu-memory-utilization 0.98 --max-model-len 8128 --quantization awq
# vllm serve  "meta-llama/Meta-Llama-3.1-70B-Instruct" --host 0.0.0.0 --port 8000  --dtype bfloat16 --enforce-eager --gpu-memory-utilization 0.95 --max-model-len 8128 --tensor-parallel-size 2 --quantization bitsandbytes --load-format bitsandbytes


# vllm serve "TheBloke/Meta-Llama-3.1-70B-Instruct-AWQ" \
#   --host 0.0.0.0 \
#   --port 8000 \
#   --dtype float16 \
#   --gpu-memory-utilization 0.90 \
#   --max-model-len 8192 \
#   --quantization awq






# python eval.py envs.names=crafter agent.type=custom agent.mode=fixed agent.planning_frequency=16 agent.remember_cot=true agent.max_text_history=16 agent.max_image_history=0 eval.num_workers=16 client.client_name=vllm client.model_id="Qwen/Qwen2-7B-Instruct" client.base_url="http://0.0.0.0:8000/v1"



# python eval.py \
#     envs.names=crafter \
#     agent.type=custom \
#     agent.mode=fixed \
#     agent.planning_frequency=16 \
#     agent.fixed_tool=cot \
#     agent.fixed_compute.strategy=best_of_n \
#     agent.fixed_compute.param=5 \
#     agent.remember_cot=true \
#     agent.max_text_history=16 \
#     eval.output_dir=  "our_results" \
#     agent.max_image_history=0 \
#     eval.num_workers=16 \
#     client.client_name=vllm \
#     client.model_id="meta-llama/Llama-3.2-1B-Instruct" \
#     client.base_url="http://0.0.0.0:8080/v1"

# set -eu

# TOOLS="best_of_n beam_search lookahead"
# STRATEGIES="heuristic_script cot reactive_actor"

# for tool in $TOOLS; do
#   for strategy in $STRATEGIES; do
#     python eval.py \
#       envs.names=crafter \
#       agent.type=custom \
#       agent.mode=fixed \
#       agent.planning_frequency=16 \
#       agent.fixed_tool="$tool" \
#       agent.fixed_compute.strategy="$strategy" \
#       agent.fixed_compute.param=5 \
#       agent.remember_cot=true \
#       agent.max_text_history=16 \
#       agent.max_image_history=0 \
#       eval.num_workers=16 \
#       eval.output_dir="our_results" \
#       client.client_name=vllm \
#       client.model_id="meta-llama/Llama-3.2-1B-Instruct" \
#       client.base_url="http://0.0.0.0:8080/v1"
#   done
# done
# for tool in $TOOLS; do
#   for strategy in $STRATEGIES; do
#     python eval.py \
#       envs.names=crafter \
#       agent.type=custom \
#       agent.mode=dynamic \
#       agent.fixed_tool="$tool" \
#       agent.fixed_compute.strategy="$strategy" \
#       agent.fixed_compute.param=5 \
#       agent.remember_cot=true \
#       agent.max_text_history=16 \
#       agent.max_image_history=0 \
#       eval.num_workers=16 \
#       eval.output_dir="our_results" \
#       client.client_name=vllm \
#       client.model_id="meta-llama/Llama-3.2-1B-Instruct" \
#       client.base_url="http://0.0.0.0:8080/v1"
#   done
# done


#####################################################################
# MULTI-TOOL CHAINING EXPERIMENTS WITH MISTRAL 7B
# Tests all combinations of compute strategies with multi-tool selection
# Start vLLM server first:
# nohup vllm serve "mistralai/Mistral-7B-Instruct-v0.2" \
#   --host 0.0.0.0 --port 8080 \
#   --kv-cache-dtype fp8_e5m2 \
#   --gpu-memory-utilization 0.85 \
#   --max-model-len 8192 \
#   > ./vllm_mistral7b.out 2>&1 < /dev/null & echo $! > ./vllm_mistral7b.pid
#####################################################################

set -eu

# Run with multi-tool chaining (use_tool_selector=true, use_compute_selector=true)
# This enables the full hierarchical gate system with all 8 tools
python eval.py \
  envs.names=crafter \
  agent._target_=balrog.agents.custom.CustomAgent \
  agent.mode=dynamic \
  agent.use_planner=true \
  agent.use_tool_selector=true \
  agent.use_compute_selector=true \
  agent.remember_cot=true \
  agent.max_text_history=16 \
  agent.max_image_history=0 \
  agent.dataset=crafter \
  eval.num_workers=16 \
  eval.output_dir="multi_tool_results" \
  client.client_name=vllm \
  client.model_id="mistralai/Mistral-7B-Instruct-v0.2" \
  client.base_url="http://0.0.0.0:8080/v1"

echo "✅ Multi-tool chaining experiment complete!"
echo "📊 Analyze results: python visualize_tool_chains.py ./multi_tool_results ./visualizations"