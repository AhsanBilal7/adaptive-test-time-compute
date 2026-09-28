"""
Evaluation Script for Trained Controller Policy

Evaluates the controller on:
1. Tool selection accuracy
2. Compute efficiency
3. Solution correctness
4. Confidence calibration
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from src.trainer_files.controller_inference import TrainedControllerPolicy
from src.math_core import MATHCore, check_math_answer_equivalence
from src.universal_agent import UniversalAgent
from BALROG.balrog.client import create_llm_client
from omegaconf import DictConfig
from src.help_functions.prompts_templates import *


class ControllerEvaluator:
    """
    Comprehensive evaluator for trained controller policy.
    """
    
    def __init__(
        self,
        controller: TrainedControllerPolicy,
        client_factory,
        prompts: Dict[str, str],
        ground_truth_file: Optional[str] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            controller: Trained controller policy
            client_factory: LLM client factory for running solutions
            prompts: Prompts dictionary for agent
            ground_truth_file: Optional file with ground truth tool/strategy selections
        """
        self.controller = controller
        self.client_factory = client_factory
        self.prompts = prompts
        
        # Load ground truth if provided
        self.ground_truth = {}
        if ground_truth_file:
            self.ground_truth = self._load_ground_truth(ground_truth_file)
    
    def _load_ground_truth(self, filepath: str) -> Dict:
        """Load ground truth tool/strategy selections."""
        ground_truth = {}
        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                if data.get("correct", False):  # Only use correct trajectories
                    qid = data["qid"]
                    if qid not in ground_truth:
                        ground_truth[qid] = {
                            "tool": data["selected_tools"][0] if data.get("selected_tools") else "cot",
                            "strategy": data["compute_configs"][0].get("strategy") if data.get("compute_configs") else "best_of_n",
                            "param": data["compute_configs"][0].get("param", 1) if data.get("compute_configs") else 1,
                        }
        return ground_truth
    
    def evaluate_tool_selection(
        self,
        test_problems: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate tool selection accuracy against ground truth.
        """
        if not self.ground_truth:
            return {"error": "No ground truth available for tool selection evaluation"}
        
        results = {
            "total": 0,
            "correct_tool": 0,
            "correct_strategy": 0,
            "correct_param": 0,
            "exact_match": 0,
            "tool_distribution": defaultdict(int),
            "strategy_distribution": defaultdict(int),
        }
        
        for problem_data in tqdm(test_problems, desc="Evaluating tool selection"):
            qid = problem_data.get("problem_id", problem_data.get("qid"))
            
            if qid not in self.ground_truth:
                continue
            
            problem = problem_data["problem"]
            decision = self.controller.predict(problem, return_reasoning=False)
            
            gt = self.ground_truth[qid]
            
            results["total"] += 1
            results["tool_distribution"][decision.tool] += 1
            results["strategy_distribution"][decision.compute_strategy] += 1
            
            if decision.tool == gt["tool"]:
                results["correct_tool"] += 1
            
            if decision.compute_strategy == gt["strategy"]:
                results["correct_strategy"] += 1
            
            if decision.compute_param == gt["param"]:
                results["correct_param"] += 1
            
            if (decision.tool == gt["tool"] and
                decision.compute_strategy == gt["strategy"] and
                decision.compute_param == gt["param"]):
                results["exact_match"] += 1
        
        # Compute accuracies
        if results["total"] > 0:
            results["tool_accuracy"] = results["correct_tool"] / results["total"]
            results["strategy_accuracy"] = results["correct_strategy"] / results["total"]
            results["param_accuracy"] = results["correct_param"] / results["total"]
            results["exact_match_accuracy"] = results["exact_match"] / results["total"]
        
        return results
    
    def evaluate_solution_quality(
        self,
        test_problems: List[Dict[str, Any]],
        max_problems: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end solution quality using controller decisions.
        """
        results = {
            "total": 0,
            "correct": 0,
            "total_compute_cost": 0,
            "avg_compute_cost": 0,
            "solutions": [],
        }
        
        # Sample problems if needed
        if len(test_problems) > max_problems:
            test_problems = np.random.choice(test_problems, max_problems, replace=False).tolist()
        
        for problem_data in tqdm(test_problems, desc="Evaluating solutions"):
            problem = problem_data["problem"]
            gold_answer = problem_data.get("gold_answer_normalized") or problem_data.get("gold_answer")
            
            # Get controller decision
            decision = self.controller.predict(problem, return_reasoning=False)
            
            # Create agent with controller's decision
            agent = UniversalAgent(
                client_factory=self.client_factory,
                use_planner=False,
                use_tool_selector=False,
                use_compute_selector=False,
                fixed_tool=decision.tool,
                fixed_compute={
                    "strategy": decision.compute_strategy,
                    "param": decision.compute_param
                },
            )
            
            # Solve problem
            try:
                response = agent.solve(problem, self.prompts)
                predicted_answer = response.answer_unstructured
                
                # Check correctness
                is_correct = check_math_answer_equivalence(predicted_answer, gold_answer)
                
                # Compute cost
                compute_cost = decision.compute_param
                
                results["total"] += 1
                if is_correct:
                    results["correct"] += 1
                results["total_compute_cost"] += compute_cost
                
                results["solutions"].append({
                    "problem": problem,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "is_correct": is_correct,
                    "tool": decision.tool,
                    "strategy": decision.compute_strategy,
                    "compute_cost": compute_cost,
                    "confidence": decision.confidence,
                })
                
            except Exception as e:
                print(f"Error solving problem: {e}")
                continue
            
            agent.reset()
        
        # Compute metrics
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]
            results["avg_compute_cost"] = results["total_compute_cost"] / results["total"]
        
        return results
    
    def evaluate_confidence_calibration(
        self,
        test_problems: List[Dict[str, Any]],
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate how well the controller's confidence correlates with correctness.
        """
        confidences = []
        correctness = []
        
        for problem_data in tqdm(test_problems, desc="Evaluating confidence"):
            problem = problem_data["problem"]
            gold_answer = problem_data.get("gold_answer_normalized") or problem_data.get("gold_answer")
            
            decision = self.controller.predict(problem, return_reasoning=False)
            
            # Solve and check
            agent = UniversalAgent(
                client_factory=self.client_factory,
                use_planner=False,
                fixed_tool=decision.tool,
                fixed_compute={
                    "strategy": decision.compute_strategy,
                    "param": decision.compute_param
                },
            )
            
            try:
                response = agent.solve(problem, self.prompts)
                is_correct = check_math_answer_equivalence(
                    response.answer_unstructured,
                    gold_answer
                )
                
                confidences.append(decision.confidence)
                correctness.append(1 if is_correct else 0)
                
            except Exception:
                continue
            
            agent.reset()
        
        if not confidences:
            return {"error": "No valid predictions"}
        
        # Bin analysis
        bins = np.linspace(0, 1, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_mask = (np.array(confidences) >= bins[i]) & (np.array(confidences) < bins[i+1])
            bin_count = bin_mask.sum()
            
            if bin_count > 0:
                bin_accuracy = np.array(correctness)[bin_mask].mean()
                bin_confidence = np.array(confidences)[bin_mask].mean()
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(int(bin_count))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bins[i] + bins[i+1]) / 2)
                bin_counts.append(0)
        
        # Expected Calibration Error (ECE)
        ece = sum(
            (count / len(confidences)) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
        )
        
        return {
            "ece": ece,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "overall_accuracy": np.mean(correctness),
            "avg_confidence": np.mean(confidences),
        }
    
    def run_full_evaluation(
        self,
        test_problems: List[Dict[str, Any]],
        output_file: str,
    ):
        """
        Run complete evaluation and save results.
        """
        print("\n" + "="*70)
        print("CONTROLLER EVALUATION")
        print("="*70 + "\n")
        
        results = {}
        
        # 1. Tool Selection Accuracy
        if self.ground_truth:
            print("[1/4] Evaluating tool selection accuracy...")
            results["tool_selection"] = self.evaluate_tool_selection(test_problems)
            print(f"  Tool Accuracy: {results['tool_selection'].get('tool_accuracy', 0):.2%}")
            print(f"  Exact Match: {results['tool_selection'].get('exact_match_accuracy', 0):.2%}")
        
        # 2. Solution Quality
        print("\n[2/4] Evaluating solution quality...")
        results["solution_quality"] = self.evaluate_solution_quality(test_problems, max_problems=50)
        print(f"  Accuracy: {results['solution_quality'].get('accuracy', 0):.2%}")
        print(f"  Avg Compute Cost: {results['solution_quality'].get('avg_compute_cost', 0):.2f}")
        
        # 3. Confidence Calibration
        print("\n[3/4] Evaluating confidence calibration...")
        results["confidence"] = self.evaluate_confidence_calibration(test_problems[:50])
        print(f"  ECE: {results['confidence'].get('ece', 0):.4f}")
        print(f"  Avg Confidence: {results['confidence'].get('avg_confidence', 0):.3f}")
        
        # Save results
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[SUCCESS] Evaluation complete. Results saved to: {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained controller policy")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained controller model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test problems JSONL"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Path to ground truth trajectories JSONL"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Config file for LLM client"
    )
    
    args = parser.parse_args()
    
    # Load controller
    controller = TrainedControllerPolicy(
        model_path=args.model_path,
        temperature=0.7,
    )
    
    # Load config and create client
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    client_factory = create_llm_client(DictConfig(config["client"]))
    
    # Load prompts
    prompts = {
        "system_prompt": MATH_SYSTEM_PROMPT,
        "final_answer_system_prompt": FINAL_ANSWER_SYSTEM_PROMPT,
        "final_answer_user_prompt": FINAL_ANSWER_USER_PROMPT,
        "unstructured_final_answer_system_prompt": UNSTRUCTURED_FINAL_ANSWER_SYSTEM_PROMPT,
        "unstructured_final_answer_user_prompt": UNSTRUCTURED_FINAL_ANSWER_USER_PROMPT,
    }
    
    # Load test data
    test_problems = []
    with open(args.test_data) as f:
        for line in f:
            test_problems.append(json.loads(line))
    
    # Create evaluator
    evaluator = ControllerEvaluator(
        controller=controller,
        client_factory=client_factory,
        prompts=prompts,
        ground_truth_file=args.ground_truth,
    )
    
    # Run evaluation
    evaluator.run_full_evaluation(
        test_problems=test_problems,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
