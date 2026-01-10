import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np
import warnings

warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")


class PRMSelector:
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-PRM-7B", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        
        print(f"[INFO] Loading PRM model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        print(f"[INFO] PRM model loaded successfully")
    
    def make_step_rewards(
        self, 
        logits: torch.Tensor, 
        token_masks: torch.Tensor
    ) -> List[List[float]]:
        
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        
        return all_scores_res
    
    def score_reasoning(
        self, 
        problem: str, 
        reasoning_steps: List[str],
        system_prompt: str
    ) -> Dict[str, Any]:
        
        if len(reasoning_steps) == 1 and isinstance(reasoning_steps[0], str):
            steps = [reasoning_steps[0]]
        else:
            steps = reasoning_steps
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
        ]
        
        try:
            conversation_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            input_ids = self.tokenizer.encode(
                conversation_str,
                return_tensors="pt",
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, use_cache=False)
            
            step_sep_id = self.tokenizer.encode("<extra_0>")[0]
            token_masks = (input_ids == step_sep_id)
            step_rewards = self.make_step_rewards(outputs[0], token_masks)
            
            rewards = step_rewards[0] if step_rewards else []
            
        except Exception as e:
            print(f"[WARNING] Error during PRM scoring: {e}")
            rewards = []
        
        if rewards:
            mean_reward = np.mean(rewards)
            min_reward = np.min(rewards)
            final_reward = rewards[-1]
        else:
            mean_reward = 0.0
            min_reward = 0.0
            final_reward = 0.0
        
        return {
            "step_rewards": rewards,
            "mean_reward": float(mean_reward),
            "min_reward": float(min_reward),
            "final_reward": float(final_reward),
            "num_steps": len(rewards)
        }
    
    def select_best_iteration(
        self,
        problem: str,
        iteration_results: List[Dict[str, Any]],
        selection_metric: str,
        system_prompt: str, 
        meta_data=None
    ) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
        
        all_scores = []
        
        print(f"[INFO] Scoring {len(iteration_results)} iterations with PRM...")
        
        for idx, iter_result in enumerate(iteration_results):
            reasoning_steps = []
            predicted_answer = iter_result.get("predicted_answer", "")
            
            if meta_data and (meta_data.get("use_tool_selector") or meta_data.get("fixed_tool")):
                # Case 1: Tool selector or fixed tool - add reasoning steps AND final answer
                reasoning_steps = iter_result.get("reasoning_steps", [])
                if not reasoning_steps:
                    reasoning = iter_result.get("reasoning", "")
                    reasoning_steps = [reasoning] if reasoning else [""]
                
                # Add final prediction at the end of reasoning steps
                if predicted_answer:
                    reasoning_steps.append(f"Final Answer: {predicted_answer}")
            else:
                # Case 2: No tool selector/fixed tool - only add final answer
                if predicted_answer:
                    reasoning_steps = [f"Final Answer of this problem is {predicted_answer}"]
            scores = self.score_reasoning(problem, reasoning_steps, system_prompt)
            
            print("=" * 60)
            print(f"Problem: {problem}")
            print(f"Reasoning Steps: {reasoning_steps}")
            print(f" Score: {scores}")
            print("=" * 60)

            scores["iteration"] = idx + 1
            all_scores.append(scores)
            
            if (idx + 1) % 5 == 0:
                print(f"[INFO] Scored {idx + 1}/{len(iteration_results)} iterations")
        
        if not all_scores:
            return 0, iteration_results[0], all_scores
        
        all_metric_values = [s.get(selection_metric, 0.0) for s in all_scores]
        if all(v == 0.0 for v in all_metric_values):
            print("[WARNING] All PRM scores are zero. Falling back to last iteration.")
            best_idx = len(iteration_results) - 1
        else:
            best_idx = max(
                range(len(all_scores)),
                key=lambda i: all_scores[i].get(selection_metric, 0.0)
            )
        
        best_iteration = iteration_results[best_idx]
        
        print(
            f"[INFO] Selected iteration {best_idx + 1} "
            f"with {selection_metric}={all_scores[best_idx][selection_metric]:.4f}"
        )
        
        return best_idx, best_iteration, all_scores
    
    def batch_score_iterations(
        self,
        problems: List[str],
        iteration_results_list: List[List[Dict[str, Any]]],
        selection_metric: str,
        system_prompt: str
    ) -> List[Tuple[int, Dict[str, Any], List[Dict[str, Any]]]]:
        
        results = []
        
        for idx, (problem, iterations) in enumerate(zip(problems, iteration_results_list)):
            print(f"\n[INFO] Processing problem {idx + 1}/{len(problems)}")
            best_idx, best_iteration, all_scores = self.select_best_iteration(
                problem, iterations, selection_metric, system_prompt
            )
            results.append((best_idx, best_iteration, all_scores))
        
        return results

