"""
Complete Training Pipeline for Controller Policy

Runs the full pipeline:
1. Generate trajectories (already done via trajectory_generator.py)
2. Create preference pairs
3. Train SFT model
4. Train GRPO model
5. Evaluate trained controller
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from src.trainer_files.preference_pair_generator import create_preference_pairs_for_splits
from src.trainer_files.train_sft import SFTTrainer
from src.trainer_files.train_grpo import GRPOTrainer
from src.trainer_files.controller_inference import TrainedControllerPolicy


class ControllerTrainingPipeline:
    """
    End-to-end training pipeline for controller policy.
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        base_model: str = "meta-llama/Llama-3.2-3B-Instruct",
        use_wandb: bool = False,
    ):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Directory containing trajectory rollouts
            output_dir: Root output directory for models
            base_model: Base model to fine-tune
            use_wandb: Whether to use W&B logging
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        self.use_wandb = use_wandb
        
        # Create output directories
        self.sft_output_dir = self.output_dir / "controller_sft"
        self.grpo_output_dir = self.output_dir / "controller_grpo"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("CONTROLLER POLICY TRAINING PIPELINE")
        print(f"{'='*70}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Base model: {self.base_model}")
        print(f"{'='*70}\n")
    
    def step1_create_preference_pairs(
        self,
        splits=["train", "dev"],
        min_score_margin=0.1,
        max_pairs_per_problem=None,
    ):
        """
        Step 1: Create preference pairs from trajectories.
        """
        print(f"\n{'='*70}")
        print("STEP 1: Creating Preference Pairs")
        print(f"{'='*70}\n")
        
        create_preference_pairs_for_splits(
            data_dir=str(self.data_dir),
            splits=splits,
            min_score_margin=min_score_margin,
            max_pairs_per_problem=max_pairs_per_problem,
        )
        
        print(f"\n[SUCCESS] Preference pairs created")
    
    def step2_train_sft(
        self,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=3,
        learning_rate=2e-4,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        load_in_4bit=True,
    ):
        """
        Step 2: Train SFT model on correct trajectories.
        """
        print(f"\n{'='*70}")
        print("STEP 2: Supervised Fine-Tuning (SFT)")
        print(f"{'='*70}\n")
        
        train_trajectories = self.data_dir / "rollouts_train.jsonl"
        val_trajectories = self.data_dir / "rollouts_dev.jsonl"
        
        if not train_trajectories.exists():
            raise FileNotFoundError(f"Training trajectories not found: {train_trajectories}")
        
        sft_trainer = SFTTrainer(
            model_name=self.base_model,
            output_dir=str(self.sft_output_dir),
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            load_in_4bit=load_in_4bit,
        )
        
        sft_trainer.train(
            train_trajectories=str(train_trajectories),
            val_trajectories=str(val_trajectories) if val_trajectories.exists() else None,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            use_wandb=self.use_wandb,
            wandb_project="controller-sft",
        )
        
        print(f"\n[SUCCESS] SFT training complete")
        return self.sft_output_dir / "final"
    
    def step3_train_grpo(
        self,
        sft_model_path: Optional[str] = None,
        batch_size=4,
        gradient_accumulation_steps=4,
        num_epochs=3,
        learning_rate=5e-5,
        beta=0.1,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        load_in_4bit=True,
    ):
        """
        Step 3: Train GRPO model on preference pairs.
        """
        print(f"\n{'='*70}")
        print("STEP 3: Group Relative Policy Optimization (GRPO)")
        print(f"{'='*70}\n")
        
        # Use SFT model if available, otherwise base model
        if sft_model_path is None:
            sft_model_path = self.sft_output_dir / "final"
        
        if not Path(sft_model_path).exists():
            print(f"[WARNING] SFT model not found at {sft_model_path}, using base model")
            sft_model_path = self.base_model
        
        train_preferences = self.data_dir / "preferences_train.jsonl"
        val_preferences = self.data_dir / "preferences_dev.jsonl"
        
        if not train_preferences.exists():
            raise FileNotFoundError(f"Training preferences not found: {train_preferences}")
        
        grpo_trainer = GRPOTrainer(
            model_name_or_path=str(sft_model_path),
            ref_model_name_or_path=str(sft_model_path),  # Use SFT as reference
            output_dir=str(self.grpo_output_dir),
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            load_in_4bit=load_in_4bit,
            beta=beta,
        )
        
        grpo_trainer.train(
            train_preferences=str(train_preferences),
            val_preferences=str(val_preferences) if val_preferences.exists() else None,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            use_wandb=self.use_wandb,
            wandb_project="controller-grpo",
        )
        
        print(f"\n[SUCCESS] GRPO training complete")
        return self.grpo_output_dir / "final"
    
    def step4_evaluate_controller(
        self,
        model_path: Optional[str] = None,
        test_problems: Optional[list] = None,
    ):
        """
        Step 4: Evaluate trained controller.
        """
        print(f"\n{'='*70}")
        print("STEP 4: Evaluating Trained Controller")
        print(f"{'='*70}\n")
        
        if model_path is None:
            model_path = self.grpo_output_dir / "final"
        
        if not Path(model_path).exists():
            print(f"[ERROR] Model not found at {model_path}")
            return
        
        # Load controller
        controller = TrainedControllerPolicy(
            model_path=str(model_path),
            temperature=0.7,
        )
        
        # Test problems
        if test_problems is None:
            test_problems = [
                "What is the sum of the first 100 positive integers?",
                "Solve the quadratic equation x^2 - 5x + 6 = 0",
                "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
                "A rectangle has length 12 and width 8. What is its area?",
                "If 3x + 5 = 20, what is the value of x?",
            ]
        
        print("[INFO] Testing controller on sample problems...\n")
        
        results = []
        for i, problem in enumerate(test_problems):
            print(f"\n--- Problem {i+1} ---")
            print(f"Problem: {problem}")
            
            decision = controller.predict(problem, return_reasoning=False)
            
            print(f"Tool: {decision.tool}")
            print(f"Strategy: {decision.compute_strategy}(n={decision.compute_param})")
            print(f"Confidence: {decision.confidence:.3f}")
            
            results.append({
                "problem": problem,
                "tool": decision.tool,
                "strategy": decision.compute_strategy,
                "param": decision.compute_param,
                "confidence": decision.confidence,
            })
        
        # Save results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[SUCCESS] Evaluation complete. Results saved to: {results_file}")
    
    def run_full_pipeline(
        self,
        skip_preference_pairs=False,
        skip_sft=False,
        skip_grpo=False,
        skip_evaluation=False,
        sft_config: Optional[dict] = None,
        grpo_config: Optional[dict] = None,
    ):
        """
        Run the complete training pipeline.
        
        Args:
            skip_preference_pairs: Skip preference pair generation
            skip_sft: Skip SFT training
            skip_grpo: Skip GRPO training
            skip_evaluation: Skip evaluation
            sft_config: Configuration for SFT training
            grpo_config: Configuration for GRPO training
        """
        print(f"\n{'='*70}")
        print("RUNNING FULL TRAINING PIPELINE")
        print(f"{'='*70}\n")
        
        sft_config = sft_config or {}
        grpo_config = grpo_config or {}
        
        # Step 1: Create preference pairs
        if not skip_preference_pairs:
            self.step1_create_preference_pairs()
        else:
            print("[INFO] Skipping preference pair generation")
        
        # Step 2: SFT training
        sft_model_path = None
        if not skip_sft:
            sft_model_path = self.step2_train_sft(**sft_config)
        else:
            print("[INFO] Skipping SFT training")
            sft_model_path = self.sft_output_dir / "final"
        
        # Step 3: GRPO training
        grpo_model_path = None
        if not skip_grpo:
            grpo_model_path = self.step3_train_grpo(
                sft_model_path=str(sft_model_path) if sft_model_path else None,
                **grpo_config
            )
        else:
            print("[INFO] Skipping GRPO training")
            grpo_model_path = self.grpo_output_dir / "final"
        
        # Step 4: Evaluation
        if not skip_evaluation:
            self.step4_evaluate_controller(model_path=str(grpo_model_path))
        else:
            print("[INFO] Skipping evaluation")
        
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"SFT Model: {self.sft_output_dir / 'final'}")
        print(f"GRPO Model: {self.grpo_output_dir / 'final'}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Complete Controller Training Pipeline")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing trajectory rollouts"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Root output directory for trained models"
    )
    
    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model to fine-tune"
    )
    
    # Pipeline control
    parser.add_argument("--skip_preference_pairs", action="store_true")
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--skip_grpo", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    
    # Training configurations
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--sft_batch_size", type=int, default=4)
    parser.add_argument("--sft_lr", type=float, default=2e-4)
    
    parser.add_argument("--grpo_epochs", type=int, default=3)
    parser.add_argument("--grpo_batch_size", type=int, default=4)
    parser.add_argument("--grpo_lr", type=float, default=5e-5)
    parser.add_argument("--grpo_beta", type=float, default=0.1)
    
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ControllerTrainingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        use_wandb=args.use_wandb,
    )
    
    # SFT configuration
    sft_config = {
        "num_epochs": args.sft_epochs,
        "batch_size": args.sft_batch_size,
        "learning_rate": args.sft_lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    
    # GRPO configuration
    grpo_config = {
        "num_epochs": args.grpo_epochs,
        "batch_size": args.grpo_batch_size,
        "learning_rate": args.grpo_lr,
        "beta": args.grpo_beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    
    # Run pipeline
    pipeline.run_full_pipeline(
        skip_preference_pairs=args.skip_preference_pairs,
        skip_sft=args.skip_sft,
        skip_grpo=args.skip_grpo,
        skip_evaluation=args.skip_evaluation,
        sft_config=sft_config,
        grpo_config=grpo_config,
    )


if __name__ == "__main__":
    main()
