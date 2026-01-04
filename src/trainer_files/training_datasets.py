"""
Dataset classes for Controller Policy Training
Supports both SFT and GRPO training formats
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class ControllerTrainingExample:
    """
    Single training example for controller policy.
    
    The controller learns to map:
    (problem, context) -> (tool_selection, compute_strategy)
    """
    problem: str
    plan: Optional[str]
    selected_tool: str
    compute_strategy: str
    compute_param: int
    reasoning: str
    final_answer: str
    is_correct: bool
    trajectory_id: str
    metadata: Dict[str, Any]


class ControllerSFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning (SFT) of controller policy.
    
    Loads trajectories and creates training examples that teach the model
    to predict tool and compute strategy selections.
    """
    
    def __init__(
        self,
        trajectories_file: str,
        tokenizer,
        max_length: int = 2048,
        use_correct_only: bool = True,
        include_plan: bool = True,
        format_style: str = "conversation"  # "conversation" or "instruction"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_correct_only = use_correct_only
        self.include_plan = include_plan
        self.format_style = format_style
        
        self.examples = self._load_trajectories(trajectories_file)
        
        print(f"[INFO] Loaded {len(self.examples)} SFT examples")
        if use_correct_only:
            print(f"  Using correct trajectories only")
        else:
            print(f"  Using both correct and incorrect trajectories")
    
    def _load_trajectories(self, filepath: str) -> List[ControllerTrainingExample]:
        """Load trajectories and convert to training examples."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Trajectories file not found: {filepath}")
        
        examples = []
        formatted_data = []  # For saving formatted dataset
        
        with open(filepath, "r") as f:
            for line in f:
                traj = json.loads(line)
                
                # Filter by correctness if specified
                if self.use_correct_only and not traj.get("correct", False):
                    continue
                
                # Extract controller decisions
                selected_tools = traj.get("selected_tools", [])
                compute_configs = traj.get("compute_configs", [])
                
                # Handle case where multiple tools/strategies are used
                # For now, use the first one (primary decision)
                selected_tool = selected_tools[0] if selected_tools else "cot"
                
                compute_config = compute_configs[0] if compute_configs else {
                    "strategy": "best_of_n",
                    "param": 1
                }
                
                # Extract detailed reasoning including tool outputs
                reasoning, metadata = self._extract_reasoning_with_details(traj)
                
                example = ControllerTrainingExample(
                    problem=traj.get("problem", ""),
                    plan=traj.get("plan"),
                    selected_tool=selected_tool,
                    compute_strategy=compute_config.get("strategy", "best_of_n"),
                    compute_param=compute_config.get("param", 1),
                    reasoning=reasoning,
                    final_answer=traj.get("final_answer", traj.get("final_answer_unstructured", "")),
                    is_correct=traj.get("correct", False),
                    trajectory_id=traj.get("run_id", ""),
                    metadata=metadata
                )
                
                examples.append(example)
                
                # Store formatted version for saving
                formatted_data.append({
                    "problem": example.problem,
                    "plan": example.plan,
                    "selected_tool": example.selected_tool,
                    "compute_strategy": example.compute_strategy,
                    "compute_param": example.compute_param,
                    "reasoning": example.reasoning,
                    "final_answer": example.final_answer,
                    "is_correct": example.is_correct,
                    "trajectory_id": example.trajectory_id,
                    "metadata": {
                        "tool_results": metadata.get("tool_results", []),
                        "compute_results": metadata.get("compute_results", []),
                        "num_steps": len(traj.get("steps", []))
                    }
                })
        
        # Save formatted dataset
        self._save_formatted_dataset(filepath, formatted_data)
        
        return examples
    
    def _extract_reasoning_with_details(self, traj: Dict) -> Tuple[str, Dict[str, Any]]:
        """
        Extract reasoning text AND metadata from trajectory steps.
        
        Returns:
            (reasoning_text, metadata_dict)
        """
        steps = traj.get("steps", [])
        reasoning_parts = []
        tool_results = []
        compute_results = []
        
        for step in steps:
            # Extract tool information
            tool_name = step.get("tool_name", "unknown")
            tool_output = step.get("tool_output", "")
            
            # Extract reasoning steps
            reasoning_steps = step.get("reasoning_steps", [])
            if reasoning_steps:
                reasoning_parts.extend(reasoning_steps)
            
            # Extract reasoner output
            reasoner_output = step.get("reasoner_output", {})
            if reasoner_output:
                reasoner_type = reasoner_output.get("reasoner", "")
                reasoner_reasoning = reasoner_output.get("reasoning", "")
                reasoner_action = reasoner_output.get("action", "")
                
                if reasoner_reasoning:
                    reasoning_parts.append(f"[{reasoner_type}] {reasoner_reasoning}")
                if reasoner_action:
                    reasoning_parts.append(f"Action: {reasoner_action}")
            
            # Store tool results
            if tool_name != "unknown":
                tool_results.append({
                    "tool": tool_name,
                    "output": tool_output[:200] if tool_output else ""  # Truncate long outputs
                })
            
            # Extract compute strategy information
            compute_output = step.get("compute_strategy_output", {})
            if compute_output:
                strategy = compute_output.get("strategy", "")
                param = compute_output.get("param", 0)
                output_data = compute_output.get("output", {})
                
                # Extract candidate information
                candidates = output_data.get("candidates", [])
                scores = output_data.get("scores", [])
                
                compute_results.append({
                    "strategy": strategy,
                    "param": param,
                    "num_candidates": len(candidates),
                    "best_score": max(scores) if scores else 0.0,
                    "candidates": candidates[:3]  # Store first 3 candidates
                })
        
        # Combine reasoning
        full_reasoning = "\n".join(reasoning_parts) if reasoning_parts else ""
        
        # Build metadata
        metadata = {
            "tool_results": tool_results,
            "compute_results": compute_results,
            "num_tools_used": len(traj.get("selected_tools", [])),
            "total_compute_budget": sum(c.get("param", 0) for c in traj.get("compute_configs", []))
        }
        
        return full_reasoning, metadata
    
    def _save_formatted_dataset(self, original_filepath: Path, formatted_data: List[Dict]):
        """Save formatted SFT dataset for inspection."""
        output_dir = original_filepath.parent
        split_name = original_filepath.stem.replace("rollouts_", "")
        
        # Save as JSONL
        output_file = output_dir / f"sft_dataset_{split_name}.jsonl"
        with open(output_file, 'w') as f:
            for example in formatted_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"[INFO] Saved SFT dataset to: {output_file}")
        
        # Save human-readable sample
        sample_file = output_dir / f"sft_dataset_{split_name}_sample.txt"
        with open(sample_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SFT DATASET SAMPLES\n")
            f.write("="*70 + "\n\n")
            
            # Save first 3 examples
            for i in range(min(3, len(formatted_data))):
                example = formatted_data[i]
                f.write(f"--- Example {i+1} ---\n\n")
                f.write(f"Problem: {example['problem']}\n\n")
                
                if example['plan']:
                    f.write(f"Plan: {example['plan']}\n\n")
                
                f.write(f"Selected Tool: {example['selected_tool']}\n")
                f.write(f"Compute Strategy: {example['compute_strategy']}(n={example['compute_param']})\n\n")
                
                if example['reasoning']:
                    reasoning_preview = example['reasoning'][:200]
                    if len(example['reasoning']) > 200:
                        reasoning_preview += "..."
                    f.write(f"Reasoning:\n{reasoning_preview}\n\n")
                
                f.write(f"Final Answer: {example['final_answer']}\n")
                f.write(f"Is Correct: {example['is_correct']}\n")
                f.write(f"Trajectory ID: {example['trajectory_id']}\n")
                f.write("\n" + "="*70 + "\n\n")
        
        print(f"[INFO] Saved sample to: {sample_file}")
    
    def _extract_reasoning(self, traj: Dict) -> str:
        """Extract reasoning text from trajectory steps."""
        steps = traj.get("steps", [])
        reasoning_parts = []
        
        for step in steps:
            reasoning_steps = step.get("reasoning_steps", [])
            reasoning_parts.extend(reasoning_steps)
        
        return "\n".join(reasoning_parts) if reasoning_parts else ""
    
    def _format_example_conversation(self, example: ControllerTrainingExample) -> str:
        """
        Format example in conversation style for training.
        
        Includes complete information:
        - Problem and plan
        - Tool selection with rationale
        - Compute strategy with results
        - Full reasoning chain
        - Tool outputs
        - Final answer
        """
        # User message
        user_msg = f"Problem: {example.problem}\n"
        
        if self.include_plan and example.plan:
            user_msg += f"\nPlan: {example.plan}\n"
        
        user_msg += "\nSelect the best reasoning tool and compute strategy for this problem, then solve it step by step."
        
        # Assistant message (target) - ENHANCED with full details
        assistant_msg = (
            f"I will use the {example.selected_tool} reasoning tool "
            f"with {example.compute_strategy}(n={example.compute_param}) strategy.\n\n"
        )
        
        # Add tool execution details if available in metadata
        if example.metadata:
            # Include tool outputs if available
            tool_results = example.metadata.get('tool_results', [])
            if tool_results:
                assistant_msg += "**Tool Execution:**\n"
                for i, result in enumerate(tool_results[:3], 1):  # Limit to first 3
                    tool_name = result.get('tool', 'unknown')
                    output = result.get('output', '')
                    if output:
                        output_preview = output[:150] + "..." if len(output) > 150 else output
                        assistant_msg += f"{i}. {tool_name}: {output_preview}\n"
                assistant_msg += "\n"
            
            # Include compute strategy results if available
            compute_results = example.metadata.get('compute_results', [])
            if compute_results:
                assistant_msg += "**Compute Strategy Results:**\n"
                for i, result in enumerate(compute_results[:2], 1):  # Limit to first 2
                    strategy = result.get('strategy', 'unknown')
                    num_candidates = result.get('num_candidates', 1)
                    best_score = result.get('best_score', 0.0)
                    assistant_msg += f"{i}. {strategy}: Generated {num_candidates} candidates, best score: {best_score:.2f}\n"
                assistant_msg += "\n"
        
        # Main reasoning
        if example.reasoning:
            assistant_msg += f"**Reasoning:**\n{example.reasoning}\n\n"
        
        # Final answer
        assistant_msg += f"**Final Answer:** {example.final_answer}"
        
        # Combine
        formatted = f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
        
        return formatted
    
    def _format_example_instruction(self, example: ControllerTrainingExample) -> str:
        """
        Format example in instruction style.
        
        Format:
        ### Instruction:
        Given the following problem, select optimal tool and compute strategy, then solve.
        
        ### Problem:
        {problem}
        
        ### Response:
        Tool: {tool}
        Strategy: {strategy}({param})
        
        {reasoning}
        
        Answer: {answer}
        """
        instruction = (
            "### Instruction:\n"
            "Given the following mathematical problem, select the optimal reasoning tool "
            "and compute strategy, then solve the problem.\n\n"
        )
        
        problem_section = f"### Problem:\n{example.problem}\n"
        
        if self.include_plan and example.plan:
            problem_section += f"\n### Plan:\n{example.plan}\n"
        
        response_section = (
            f"\n### Response:\n"
            f"**Tool Selection:** {example.selected_tool}\n"
            f"**Compute Strategy:** {example.compute_strategy}(n={example.compute_param})\n\n"
            f"**Reasoning:**\n{example.reasoning}\n\n"
            f"**Final Answer:** {example.final_answer}"
        )
        
        formatted = instruction + problem_section + response_section
        
        return formatted
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Format based on style
        if self.format_style == "conversation":
            text = self._format_example_conversation(example)
        else:
            text = self._format_example_instruction(example)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Only return tensors that the collator can handle
        # Store metadata separately if needed for analysis
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),  # For causal LM
        }


class ControllerGRPODataset(Dataset):
    """
    Dataset for Group Relative Policy Optimization (GRPO) training.
    
    Loads preference pairs where each pair consists of:
    - Preferred trajectory (correct/efficient)
    - Rejected trajectory (incorrect/inefficient)
    
    The model learns to assign higher probability to preferred trajectories.
    """
    
    def __init__(
        self,
        preferences_file: str,
        tokenizer,
        max_length: int = 2048,
        include_plan: bool = True,
        include_reasoning: bool = False,  # Whether to include full reasoning
        format_style: str = "conversation"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_plan = include_plan
        self.include_reasoning = include_reasoning
        self.format_style = format_style
        
        self.pairs = self._load_preferences(preferences_file)
        
        print(f"[INFO] Loaded {len(self.pairs)} GRPO preference pairs")
    
    def _load_preferences(self, filepath: str) -> List[Dict]:
        """Load preference pairs from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preferences file not found: {filepath}")
        
        pairs = []
        with open(filepath, "r") as f:
            for line in f:
                pair = json.loads(line)
                pairs.append(pair)
        
        return pairs
    
    def _format_trajectory_for_policy(
        self, 
        problem: str,
        plan: Optional[str],
        trajectory: Dict
    ) -> str:
        """
        Format a trajectory for policy learning.
        
        Focus on the controller's decisions (tool and compute strategy)
        rather than the full reasoning trace.
        """
        # User query
        user_msg = f"Problem: {problem}\n"
        
        if self.include_plan and plan:
            user_msg += f"\nPlan: {plan}\n"
        
        user_msg += "\nSelect the best reasoning tool and compute strategy for this problem."
        
        # Assistant decision
        selected_tools = trajectory.get("selected_tools", [])
        compute_configs = trajectory.get("compute_configs", [])
        
        tool = selected_tools[0] if selected_tools else "cot"
        config = compute_configs[0] if compute_configs else {"strategy": "best_of_n", "param": 1}
        
        assistant_msg = (
            f"I will use the {tool} reasoning tool "
            f"with {config.get('strategy', 'best_of_n')}(n={config.get('param', 1)}) strategy."
        )
        
        # Optionally include reasoning
        if self.include_reasoning:
            reasoning = self._extract_trajectory_reasoning(trajectory)
            if reasoning:
                assistant_msg += f"\n\nReasoning:\n{reasoning}"
        
        # Include final answer
        final_answer = trajectory.get("final_answer", "")
        if final_answer:
            assistant_msg += f"\n\nFinal Answer: {final_answer}"
        
        # Format based on style
        if self.format_style == "conversation":
            formatted = f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
        else:
            formatted = (
                f"### Problem:\n{user_msg}\n\n"
                f"### Response:\n{assistant_msg}"
            )
        
        return formatted
    
    def _extract_trajectory_reasoning(self, trajectory: Dict) -> str:
        """Extract reasoning from trajectory steps."""
        steps = trajectory.get("steps", [])
        reasoning_parts = []
        
        for step in steps:
            reasoning_steps = step.get("reasoning_steps", [])
            reasoning_parts.extend(reasoning_steps)
        
        # Limit length
        full_reasoning = "\n".join(reasoning_parts)
        if len(full_reasoning) > 1000:
            full_reasoning = full_reasoning[:1000] + "..."
        
        return full_reasoning
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        
        problem = pair["problem"]
        preferred_traj = pair["preferred_trajectory"]
        rejected_traj = pair["rejected_trajectory"]
        
        # Format preferred trajectory
        preferred_text = self._format_trajectory_for_policy(
            problem=problem,
            plan=preferred_traj.get("plan"),
            trajectory=preferred_traj
        )
        
        # Format rejected trajectory
        rejected_text = self._format_trajectory_for_policy(
            problem=problem,
            plan=rejected_traj.get("plan"),
            trajectory=rejected_traj
        )
        
        # Tokenize preferred
        preferred_encoding = self.tokenizer(
            preferred_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize rejected
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "preferred_input_ids": preferred_encoding["input_ids"].squeeze(0),
            "preferred_attention_mask": preferred_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
            "score_diff": pair.get("score_diff", 0.0),
            "pair_type": pair.get("pair_type", "unknown"),
            "qid": pair.get("qid", "")
        }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle_train: bool = True
):
    """Create data loaders for training and validation."""
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader