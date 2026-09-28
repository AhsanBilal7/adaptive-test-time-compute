"""
Preference Pair Generator
Creates multiple (correct, incorrect) pairs per problem for GRPO training
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np


class PreferencePairGenerator:
    """
    Generate preference pairs from trajectory rollouts.
    
    For each problem, creates multiple pairs:
    - Best correct vs each incorrect trajectory
    - Best correct vs efficiency negatives (correct but suboptimal)
    - Margin-based pairs (correct vs incorrect with varying score differences)
    """
    
    def __init__(
        self,
        rollouts_file: str,
        output_file: str,
        min_score_margin: float = 0.1,
        max_pairs_per_problem: int = None
    ):
        self.rollouts_file = Path(rollouts_file)
        self.output_file = Path(output_file)
        self.min_score_margin = min_score_margin
        self.max_pairs_per_problem = max_pairs_per_problem
        
    def load_trajectories(self) -> Dict[str, List[Dict]]:
        """Load and group trajectories by problem ID."""
        if not self.rollouts_file.exists():
            raise FileNotFoundError(f"Rollouts file not found: {self.rollouts_file}")
        
        trajectories = []
        with open(self.rollouts_file, "r") as f:
            for line in f:
                trajectories.append(json.loads(line))
        
        # Group by question ID
        qid_to_trajectories = defaultdict(list)
        for traj in trajectories:
            qid = traj["qid"]
            qid_to_trajectories[qid].append(traj)
        
        return dict(qid_to_trajectories)
    
    def compute_trajectory_quality_score(self, traj: Dict) -> float:
        """
        Compute comprehensive quality score for a trajectory.
        
        Score = correctness_bonus - cost_penalty
        where cost includes compute budget, tool usage, and reasoning length
        """
        # Base score
        correctness_bonus = 100.0 if traj["correct"] else 0.0
        
        # Cost penalties
        cost = traj.get("cost", {})
        compute_penalty = cost.get("compute_param_sum", 0) * 2.0
        tool_penalty = cost.get("tool_calls", 0) * 1.0
        length_penalty = cost.get("reasoning_step_count", 0) * 0.01
        
        # Efficiency bonus for correct trajectories
        efficiency_bonus = 0.0
        if traj["correct"]:
            # Reward shorter, more efficient solutions
            if cost.get("compute_param_sum", float('inf')) <= 4:
                efficiency_bonus += 10.0
            if cost.get("tool_calls", float('inf')) <= 3:
                efficiency_bonus += 5.0
        
        quality_score = (
            correctness_bonus 
            + efficiency_bonus
            - compute_penalty 
            - tool_penalty 
            - length_penalty
        )
        
        return quality_score
    
    def create_preference_pairs(self) -> List[Dict]:
        """
        Create comprehensive preference pairs for GRPO training.
        
        For each problem:
        1. Find best correct trajectory (highest quality score)
        2. Create pairs with ALL incorrect trajectories
        3. Create efficiency pairs (best vs suboptimal correct)
        4. Create margin-based pairs (varied difficulty)
        
        Returns:
            List of preference pairs, each containing:
            - qid: problem ID
            - preferred_trajectory: full correct trajectory data
            - rejected_trajectory: full incorrect trajectory data
            - score_diff: quality score difference
            - pair_type: type of preference pair
            - metadata: additional info for training
        """
        qid_to_trajectories = self.load_trajectories()
        all_pairs = []
        
        stats = {
            "total_problems": len(qid_to_trajectories),
            "problems_with_pairs": 0,
            "correctness_pairs": 0,
            "efficiency_pairs": 0,
            "margin_pairs": 0,
            "skipped_no_correct": 0,
            "skipped_no_incorrect": 0
        }
        
        for qid, trajectories in qid_to_trajectories.items():
            # Compute quality scores for all trajectories
            for traj in trajectories:
                traj["quality_score"] = self.compute_trajectory_quality_score(traj)
            
            # Separate correct and incorrect
            correct_trajs = [t for t in trajectories if t["correct"]]
            incorrect_trajs = [t for t in trajectories if not t["correct"]]
            
            if not correct_trajs:
                stats["skipped_no_correct"] += 1
                continue
            
            # Find best correct trajectory
            best_correct = max(correct_trajs, key=lambda t: t["quality_score"])
            
            problem_pairs = []
            
            # ===== TYPE 1: Correctness Pairs =====
            # Best correct vs EACH incorrect trajectory
            if incorrect_trajs:
                for incorrect_traj in incorrect_trajs:
                    score_diff = best_correct["quality_score"] - incorrect_traj["quality_score"]
                    
                    # Only create pair if margin is significant
                    if score_diff >= self.min_score_margin:
                        pair = {
                            "qid": qid,
                            "problem": best_correct["problem"],
                            "preferred_trajectory": self._extract_trajectory_data(best_correct),
                            "rejected_trajectory": self._extract_trajectory_data(incorrect_traj),
                            "score_diff": float(score_diff),
                            "pair_type": "correctness",
                            "metadata": {
                                "preferred_run_id": best_correct["run_id"],
                                "rejected_run_id": incorrect_traj["run_id"],
                                "preferred_config": best_correct.get("config", {}),
                                "rejected_config": incorrect_traj.get("config", {}),
                            }
                        }
                        problem_pairs.append(pair)
                        stats["correctness_pairs"] += 1
            else:
                stats["skipped_no_incorrect"] += 1

            # ===== TYPE 3: Margin-Based Pairs =====
            # Create pairs with varied score differences for curriculum learning
            if incorrect_trajs:
                # Sort incorrect by quality (best incorrect first)
                sorted_incorrect = sorted(
                    incorrect_trajs,
                    key=lambda t: t["quality_score"],
                    reverse=True
                )
                
                # Create pairs with varied margins (easy, medium, hard)
                for incorrect_traj in sorted_incorrect[:3]:  # Top 3 incorrect
                    score_diff = best_correct["quality_score"] - incorrect_traj["quality_score"]
                    
                    if score_diff >= self.min_score_margin:
                        # Determine difficulty based on score difference
                        if score_diff < 20:
                            difficulty = "hard"  # Close scores
                        elif score_diff < 50:
                            difficulty = "medium"
                        else:
                            difficulty = "easy"  # Large score gap
                        
                        pair = {
                            "qid": qid,
                            "problem": best_correct["problem"],
                            "preferred_trajectory": self._extract_trajectory_data(best_correct),
                            "rejected_trajectory": self._extract_trajectory_data(incorrect_traj),
                            "score_diff": float(score_diff),
                            "pair_type": f"margin_{difficulty}",
                            "metadata": {
                                "preferred_run_id": best_correct["run_id"],
                                "rejected_run_id": incorrect_traj["run_id"],
                                "difficulty": difficulty,
                                "preferred_config": best_correct.get("config", {}),
                                "rejected_config": incorrect_traj.get("config", {}),
                            }
                        }
                        problem_pairs.append(pair)
                        stats["margin_pairs"] += 1
            
            # Limit pairs per problem if specified
            if self.max_pairs_per_problem and len(problem_pairs) > self.max_pairs_per_problem:
                # Sample diverse pairs
                problem_pairs = self._sample_diverse_pairs(
                    problem_pairs, 
                    self.max_pairs_per_problem
                )
            
            if problem_pairs:
                stats["problems_with_pairs"] += 1
                all_pairs.extend(problem_pairs)
        
        return all_pairs, stats
    
    def _extract_trajectory_data(self, traj: Dict) -> Dict:
        """Extract relevant trajectory data for training."""
        return {
            "run_id": traj["run_id"],
            "plan": traj.get("plan"),
            "selected_tools": traj.get("selected_tools", []),
            "compute_configs": traj.get("compute_configs", []),
            "steps": traj.get("steps", []),
            "final_answer": traj.get("final_answer_unstructured", ""),
            "correct": traj.get("correct", False),
            "cost": traj.get("cost", {}),
            "quality_score": traj.get("quality_score", 0.0),
            "config_name": traj.get("config", {}).get("name", "unknown"),
            # Include detailed outputs for analysis
            "compute_strategy_outputs": traj.get("all_compute_strategy_outputs", []),
            "reasoner_outputs": traj.get("all_reasoner_outputs", []),
        }
    
    def _sample_diverse_pairs(
        self, 
        pairs: List[Dict], 
        max_pairs: int
    ) -> List[Dict]:
        """Sample diverse pairs maintaining type distribution."""
        # Group by pair type
        type_to_pairs = defaultdict(list)
        for pair in pairs:
            type_to_pairs[pair["pair_type"]].append(pair)
        
        # Sample proportionally from each type
        sampled = []
        types = list(type_to_pairs.keys())
        pairs_per_type = max_pairs // len(types)
        
        for pair_type in types:
            type_pairs = type_to_pairs[pair_type]
            n_sample = min(pairs_per_type, len(type_pairs))
            sampled.extend(np.random.choice(type_pairs, n_sample, replace=False).tolist())
        
        # Fill remaining slots if any
        remaining = max_pairs - len(sampled)
        if remaining > 0:
            all_remaining = [p for p in pairs if p not in sampled]
            if all_remaining:
                n_sample = min(remaining, len(all_remaining))
                sampled.extend(np.random.choice(all_remaining, n_sample, replace=False).tolist())
        
        return sampled
    
    def save_pairs(self, pairs: List[Dict]):
        """Save preference pairs to JSONL file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        
        print(f"[INFO] Saved {len(pairs)} preference pairs to: {self.output_file}")
    
    def generate_and_save(self) -> Tuple[List[Dict], Dict]:
        """Main method: generate and save preference pairs."""
        print(f"\n[INFO] Generating preference pairs from: {self.rollouts_file}")
        
        pairs, stats = self.create_preference_pairs()
        
        print(f"\n[STATS] Preference Pair Generation:")
        print(f"  Total problems: {stats['total_problems']}")
        print(f"  Problems with pairs: {stats['problems_with_pairs']}")
        print(f"  Skipped (no correct): {stats['skipped_no_correct']}")
        print(f"  Skipped (no incorrect): {stats['skipped_no_incorrect']}")
        print(f"\n[PAIR TYPES]:")
        print(f"  Correctness pairs: {stats['correctness_pairs']}")
        print(f"  Efficiency pairs: {stats['efficiency_pairs']}")
        print(f"  Margin pairs: {stats['margin_pairs']}")
        print(f"  Total pairs: {len(pairs)}")
        
        self.save_pairs(pairs)
        
        return pairs, stats


def create_preference_pairs_for_splits(
    data_dir: str,
    splits: List[str] = ["train", "dev"],
    **kwargs
):
    """
    Create preference pairs for multiple data splits.
    
    Args:
        data_dir: Directory containing rollout files
        splits: List of split names (e.g., ["train", "dev", "test"])
        **kwargs: Additional arguments for PreferencePairGenerator
    """
    data_dir = Path(data_dir)
    
    for split in splits:
        rollouts_file = data_dir / f"rollouts_{split}.jsonl"
        output_file = data_dir / f"preferences_{split}.jsonl"
        
        if not rollouts_file.exists():
            print(f"[WARNING] Rollouts file not found for split '{split}': {rollouts_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print(f"{'='*60}")
        
        generator = PreferencePairGenerator(
            rollouts_file=str(rollouts_file),
            output_file=str(output_file),
            **kwargs
        )
        
        generator.generate_and_save()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate preference pairs from trajectories")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing rollout JSONL files"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "dev"],
        help="Data splits to process"
    )
    parser.add_argument(
        "--min_score_margin",
        type=float,
        default=0.1,
        help="Minimum score difference for pair creation"
    )
    parser.add_argument(
        "--max_pairs_per_problem",
        type=int,
        default=None,
        help="Maximum number of pairs per problem (None = unlimited)"
    )
    
    args = parser.parse_args()
    
    create_preference_pairs_for_splits(
        data_dir=args.data_dir,
        splits=args.splits,
        min_score_margin=args.min_score_margin,
        max_pairs_per_problem=args.max_pairs_per_problem
    )
