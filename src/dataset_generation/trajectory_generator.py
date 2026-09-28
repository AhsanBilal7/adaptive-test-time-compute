"""Collects multi-configuration agent rollouts and builds preference pairs for the optional learned controller."""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datasets import load_dataset
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from src.universal_agent import UniversalAgent
from src.math_core import MATHCore


@dataclass
class TrajectoryStep:
    tool_name: str
    tool_input: str
    tool_output: str
    reasoning_steps: List[str]
    # Detailed per-step outputs
    reasoner_output: Optional[Dict[str, Any]] = None
    compute_strategy_output: Optional[Dict[str, Any]] = None


@dataclass
class Trajectory:
    qid: str
    problem: str
    plan: Optional[str]
    selected_tools: List[str]
    compute_configs: List[Dict[str, Any]]
    steps: List[TrajectoryStep]
    final_answer_structured: str
    final_answer_unstructured: str
    correct: bool
    cost: Dict[str, float]
    run_id: str
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    # Complete output tracking
    all_compute_strategy_outputs: List[Dict[str, Any]] = None
    all_reasoner_outputs: List[Dict[str, Any]] = None
    # Answer tracking
    predicted_answer: str = None
    gold_answer: str = None


class TrajectoryGenerator:
    def __init__(
        self,
        client_factory,
        prompts: Dict[str, str],
        output_dir: str,
        train_split_ratio: float = 0.7,
        dev_split_ratio: float = 0.15,
        seed: int = 42
    ):
        self.client_factory = client_factory
        self.prompts = prompts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Load and split dataset
        self.dataset = load_dataset("qwedsacf/competition_math")
        self.train_data, self.dev_data, self.test_data = self._split_dataset(
            train_split_ratio, dev_split_ratio
        )
        
        print(f"[INFO] Dataset splits: Train={len(self.train_data)}, "
              f"Dev={len(self.dev_data)}, Test={len(self.test_data)}")
    
    def _split_dataset(self, train_ratio, dev_ratio):
        """Split dataset into train/dev/test with fixed seed."""
        full_data = self.dataset['train']
        n_total = len(full_data)
        
        np.random.seed(self.seed)
        indices = np.random.permutation(n_total)
        
        n_train = int(n_total * train_ratio)
        n_dev = int(n_total * dev_ratio)
        
        train_indices = indices[:n_train]
        dev_indices = indices[n_train:n_train + n_dev]
        test_indices = indices[n_train + n_dev:]
        
        return (
            full_data.select(train_indices.tolist()),
            full_data.select(dev_indices.tolist()),
            full_data.select(test_indices.tolist())
        )
    
    def _get_rollout_configs(self) -> List[Dict[str, Any]]:
        """Generate K rollout configurations per question."""
        configs = []
        
        # 10× Full system with compute selector
        configs.extend([
            {
                "name": f"full_system_n{i}",
                "use_planner": True,
                "use_tool_selector": True,
                "use_compute_selector": True,
                "fixed_tool": None,
                "fixed_compute": None,
            }
            for i in range(1,  10)
        ])    
        return configs


    def _compute_trajectory_cost(
        self,
        metadata: Dict[str, Any],
        reasoning_steps: List[str]
    ) -> Dict[str, float]:
        """Compute trajectory cost metrics."""
        tool_calls = len(metadata.get("tools_used", []))
        
        compute_param_sum = sum(
            config.get("param", 0)
            for config in metadata.get("compute_configs_used", [])
        )
        
        reasoning_step_count = len(reasoning_steps)
        
        return {
            "tool_calls": tool_calls,
            "compute_param_sum": compute_param_sum,
            "reasoning_step_count": reasoning_step_count,
        }
    
    def _compute_trajectory_score(
        self,
        correct: bool,
        cost: Dict[str, float],
        lambda_tool: float = 0.01,
        mu_compute: float = 0.005,
        nu_steps: float = 0.001
    ) -> float:
        """Compute trajectory utility score."""
        score = float(correct)
        score -= lambda_tool * cost["tool_calls"]
        score -= mu_compute * cost["compute_param_sum"]
        score -= nu_steps * cost["reasoning_step_count"]
        return score
    
    def _create_trajectory_from_response(
        self,
        qid: str,
        problem: str,
        response,
        config: Dict[str, Any],
        run_id: str,
        is_correct: bool,
        predicted_answer: str,
        gold_answer: str
    ) -> Trajectory:
        """Create trajectory object from UniversalAgent response."""
        # Extract steps from metadata with enhanced output tracking
        steps = []
        compute_metadata = response.metadata.get("compute_metadata", [])
        
        # Get compute strategy and reasoner outputs
        compute_strategy_outputs = getattr(response, 'compute_strategy_outputs', [])
        reasoner_outputs = getattr(response, 'reasoner_outputs', [])
        
        # Match steps with their outputs
        for i, meta in enumerate(compute_metadata):
            tool_name = meta.get("tool", "unknown")
            
            # Get reasoning for this step
            step_reasoning = meta.get("chosen_reasoning", "")
            if not step_reasoning:
                step_reasoning = meta.get("reasoning", "")
            
            # Split reasoning into individual steps
            reasoning_steps = [s.strip() for s in step_reasoning.split('\n') if s.strip()]
            
            # Get corresponding compute strategy output
            compute_output = None
            if i < len(compute_strategy_outputs):
                compute_output = compute_strategy_outputs[i]
            
            # Get corresponding reasoner output
            reasoner_output = None
            if i < len(reasoner_outputs):
                reasoner_output = reasoner_outputs[i]
            
            step = TrajectoryStep(
                tool_name=tool_name,
                tool_input=problem,  # Simplified
                tool_output=str(meta.get("action", "")),
                reasoning_steps=reasoning_steps,
                reasoner_output=reasoner_output,
                compute_strategy_output=compute_output
            )
            steps.append(step)
        
        # If no steps in metadata, use reasoning_steps from response
        if not steps and response.reasoning_steps:
            # Still try to get outputs if available
            reasoner_output = reasoner_outputs[0] if reasoner_outputs else None
            compute_output = compute_strategy_outputs[0] if compute_strategy_outputs else None
            
            steps.append(TrajectoryStep(
                tool_name="direct",
                tool_input=problem,
                tool_output=response.answer,
                reasoning_steps=response.reasoning_steps,
                reasoner_output=reasoner_output,
                compute_strategy_output=compute_output
            ))
        
        cost = self._compute_trajectory_cost(response.metadata, response.reasoning_steps)
        
        return Trajectory(
            qid=qid,
            problem=problem,
            plan=response.plan,
            selected_tools=response.metadata.get("tools_used", []),
            compute_configs=response.metadata.get("compute_configs_used", []),
            steps=steps,
            final_answer_structured=response.answer,
            final_answer_unstructured=response.answer_unstructured,
            correct=is_correct,
            cost=cost,
            run_id=run_id,
            config=config,
            metadata=response.metadata,
            all_compute_strategy_outputs=compute_strategy_outputs,
            all_reasoner_outputs=reasoner_outputs,
            predicted_answer=predicted_answer,
            gold_answer=gold_answer
        )
    
    def generate_trajectories(
        self,
        split: str = "train",
        max_problems: Optional[int] = None,
        k_rollouts: int = 10
    ):
        """Generate K rollouts for each problem in the specified split."""
        if split == "train":
            data = self.train_data
        elif split == "dev":
            data = self.dev_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if max_problems:
            data = data.select(range(min(max_problems, len(data))))
        
        output_file = self.output_dir / f"rollouts_{split}.jsonl"
        
        configs = self._get_rollout_configs()[:k_rollouts]
        
        print(f"[INFO] Generating {len(configs)} rollouts per problem for {len(data)} problems")
        print(f"[INFO] Output: {output_file}")
        
        with open(output_file, "w") as f:
            for idx, problem_data in enumerate(data):
                problem = problem_data["problem"]
                solution = problem_data["solution"]
                qid = f"{split}_{idx}"
                
                print(f"\n[{idx+1}/{len(data)}] Processing: {qid}")
                print(f"Problem: {problem[:100]}...")
                
                for config_idx, config in enumerate(configs):
                    print(f"  Config {config_idx+1}/{len(configs)}: {config['name']}")
                    
                    # Create agent with this config
                    agent = UniversalAgent(
                        client_factory=self.client_factory,
                        use_planner=config["use_planner"],
                        use_tool_selector=config["use_tool_selector"],
                        use_compute_selector=config["use_compute_selector"],
                        fixed_tool=config.get("fixed_tool"),
                        fixed_compute=config.get("fixed_compute"),
                    )
                    
                    math_core = MATHCore(agent, self.prompts)
                    
                    try:
                        start_time = time.time()
                        response = math_core.solve(problem)
                        elapsed = time.time() - start_time
                        
                        # Evaluate correctness
                        predicted = MATHCore.extract_prediction(response, problem_data)
                        gold = MATHCore.get_last_dollar_normalize_final_answer(problem_data["solution"]) 
                        is_correct = MATHCore.check_answer(predicted, gold)
                        
                        # Create trajectory
                        run_id = f"{qid}_run{config_idx}"
                        trajectory = self._create_trajectory_from_response(
                            qid=qid,
                            problem=problem,
                            response=response,
                            config=config,
                            run_id=run_id,
                            is_correct=is_correct,
                            predicted_answer=predicted,
                            gold_answer=gold
                        )
                        
                        # Add score
                        score = self._compute_trajectory_score(
                            trajectory.correct,
                            trajectory.cost
                        )
                        
                        # Write to file
                        output_dict = asdict(trajectory)
                        output_dict["score"] = score
                        output_dict["elapsed_time"] = elapsed
                        
                        f.write(json.dumps(output_dict) + "\n")
                        f.flush()
                        
                        print(f"    Correct: {is_correct}, Score: {score:.4f}, Time: {elapsed:.2f}s")
                        
                    except Exception as e:
                        print(f"    ERROR: {str(e)}")
                        continue
        
        print(f"\n[INFO] Trajectory generation complete: {output_file}")
    
    def create_preference_pairs(
        self,
        split: str = "train"
    ):
        """Create contrastive preference pairs from trajectories."""
        rollouts_file = self.output_dir / f"rollouts_{split}.jsonl"
        output_file = self.output_dir / f"prefs_{split}.jsonl"
        
        if not rollouts_file.exists():
            raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")
        
        # Load all trajectories
        trajectories = []
        with open(rollouts_file, "r") as f:
            for line in f:
                trajectories.append(json.loads(line))
        
        # Group by qid
        qid_to_trajectories = {}
        for traj in trajectories:
            qid = traj["qid"]
            if qid not in qid_to_trajectories:
                qid_to_trajectories[qid] = []
            qid_to_trajectories[qid].append(traj)
        
        print(f"[INFO] Creating preference pairs from {len(qid_to_trajectories)} problems")
        
        pairs = []
        for qid, trajs in qid_to_trajectories.items():
            # Separate correct and incorrect
            correct_trajs = [t for t in trajs if t["correct"]]
            incorrect_trajs = [t for t in trajs if not t["correct"]]
            
            if not correct_trajs:
                continue  # Skip if no correct trajectories
            
            # Find best correct trajectory (highest score)
            best_correct = max(correct_trajs, key=lambda t: t["score"])
            
            # Create pairs
            # 1. Hard negative: best incorrect
            if incorrect_trajs:
                best_incorrect = max(incorrect_trajs, key=lambda t: t["score"])
                pairs.append({
                    "qid": qid,
                    "preferred_run_id": best_correct["run_id"],
                    "rejected_run_id": best_incorrect["run_id"],
                    "preference_type": "hard_negative",
                    "score_diff": best_correct["score"] - best_incorrect["score"]
                })
            
            # 2. Efficiency negative: correct but high cost
            if len(correct_trajs) > 1:
                # Sort correct trajectories by score (descending)
                sorted_correct = sorted(correct_trajs, key=lambda t: t["score"], reverse=True)
                
                # Take best vs worst efficient
                efficiency_negative = sorted_correct[-1]
                pairs.append({
                    "qid": qid,
                    "preferred_run_id": best_correct["run_id"],
                    "rejected_run_id": efficiency_negative["run_id"],
                    "preference_type": "efficiency_negative",
                    "score_diff": best_correct["score"] - efficiency_negative["score"]
                })
        
        # Write pairs
        with open(output_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        
        print(f"[INFO] Created {len(pairs)} preference pairs: {output_file}")
        print(f"  Hard negatives: {sum(1 for p in pairs if p['preference_type'] == 'hard_negative')}")
        print(f"  Efficiency negatives: {sum(1 for p in pairs if p['preference_type'] == 'efficiency_negative')}")
    
    def analyze_trajectory_outputs(
        self,
        split: str = "train",
        output_analysis_file: Optional[str] = None
    ):
        """Analyze compute strategy and reasoner outputs across trajectories."""
        rollouts_file = self.output_dir / f"rollouts_{split}.jsonl"
        
        if not rollouts_file.exists():
            raise FileNotFoundError(f"Rollouts file not found: {rollouts_file}")
        
        if output_analysis_file is None:
            output_analysis_file = self.output_dir / f"analysis_{split}.json"
        
        # Load all trajectories
        trajectories = []
        with open(rollouts_file, "r") as f:
            for line in f:
                trajectories.append(json.loads(line))
        
        # Analysis metrics
        analysis = {
            "total_trajectories": len(trajectories),
            "correct_trajectories": sum(1 for t in trajectories if t["correct"]),
            "compute_strategy_stats": {},
            "reasoner_stats": {},
            "candidate_diversity": [],
            "score_distributions": {},
        }
        
        # Analyze compute strategies
        for traj in trajectories:
            if "all_compute_strategy_outputs" in traj and traj["all_compute_strategy_outputs"]:
                for cs_output in traj["all_compute_strategy_outputs"]:
                    strategy = cs_output.get("strategy", "unknown")
                    
                    if strategy not in analysis["compute_strategy_stats"]:
                        analysis["compute_strategy_stats"][strategy] = {
                            "count": 0,
                            "avg_candidates": [],
                            "avg_scores": []
                        }
                    
                    analysis["compute_strategy_stats"][strategy]["count"] += 1
                    
                    output_data = cs_output.get("output", {})
                    if "all_candidates" in output_data:
                        num_candidates = len(output_data["all_candidates"])
                        analysis["compute_strategy_stats"][strategy]["avg_candidates"].append(num_candidates)
                    
                    if "scores" in output_data:
                        scores = output_data["scores"]
                        analysis["compute_strategy_stats"][strategy]["avg_scores"].extend(scores)
        
        # Compute averages
        for strategy, stats in analysis["compute_strategy_stats"].items():
            if stats["avg_candidates"]:
                stats["avg_candidates"] = np.mean(stats["avg_candidates"])
            if stats["avg_scores"]:
                stats["mean_score"] = np.mean(stats["avg_scores"])
                stats["std_score"] = np.std(stats["avg_scores"])
                del stats["avg_scores"]  # Remove raw scores to save space
        
        # Analyze reasoners
        for traj in trajectories:
            if "all_reasoner_outputs" in traj and traj["all_reasoner_outputs"]:
                for reasoner_output in traj["all_reasoner_outputs"]:
                    reasoner = reasoner_output.get("reasoner", "unknown")
                    
                    if reasoner not in analysis["reasoner_stats"]:
                        analysis["reasoner_stats"][reasoner] = {
                            "count": 0,
                            "compute_enhanced": 0
                        }
                    
                    analysis["reasoner_stats"][reasoner]["count"] += 1
                    
                    if reasoner_output.get("compute_enhanced", False):
                        analysis["reasoner_stats"][reasoner]["compute_enhanced"] += 1
        
        # Save analysis
        with open(output_analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n[INFO] Trajectory analysis complete: {output_analysis_file}")
        print(f"  Total trajectories: {analysis['total_trajectories']}")
        print(f"  Correct: {analysis['correct_trajectories']}")
        print(f"  Compute strategies analyzed: {len(analysis['compute_strategy_stats'])}")
        print(f"  Reasoners analyzed: {len(analysis['reasoner_stats'])}")
        
        return analysis
    

def run_full_pipeline(
    client_factory,
    prompts: Dict[str, str],
    output_dir: str,
    max_train_problems: int = 1000,
    max_dev_problems: int = 200,
    k_rollouts: int = 10,
    num_epochs: int = 10,
    batch_size: int = 32
):
    """Run full dataset generation and training pipeline."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== STAGE 1: Dataset Generation =====
    print("\n" + "="*60)
    print("STAGE 1: Dataset Generation")
    print("="*60)
    
    generator = TrajectoryGenerator(
        client_factory=client_factory,
        prompts=prompts,
        output_dir=output_dir,
        train_split_ratio=0.7,
        dev_split_ratio=0.15,
        seed=42
    )
    
    # Generate train trajectories
    print("\n[1/4] Generating TRAIN trajectories...")
    generator.generate_trajectories(
        split="train",
        max_problems=max_train_problems,
        k_rollouts=k_rollouts
    )
    
    # Generate dev trajectories
    print("\n[2/4] Generating DEV trajectories...")
    generator.generate_trajectories(
        split="dev",
        max_problems=max_dev_problems,
        k_rollouts=k_rollouts
    )
    
    # Create preference pairs
    print("\n[3/4] Creating TRAIN preference pairs...")
    generator.create_preference_pairs(split="train")
    
    print("\n[4/4] Creating DEV preference pairs...")
    generator.create_preference_pairs(split="dev")
    