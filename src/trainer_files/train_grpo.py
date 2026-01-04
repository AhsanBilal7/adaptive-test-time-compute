"""
Group Relative Policy Optimization (GRPO) Training for Controller Policy

Uses preference pairs to train the model to prefer correct/efficient
trajectories over incorrect/inefficient ones.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

from src.trainer_files.training_datasets import ControllerGRPODataset


class GRPOTrainer:
    """
    GRPO Trainer for Controller Policy using Direct Preference Optimization (DPO).
    
    DPO is a simpler alternative to traditional RLHF that directly optimizes
    the policy to prefer chosen responses over rejected ones without needing
    a separate reward model.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        ref_model_name_or_path: Optional[str] = None,
        output_dir: str = "./models/controller_grpo",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        beta: float = 0.1,  # DPO temperature parameter
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model_name_or_path: Path to SFT model or base model
            ref_model_name_or_path: Reference model (if None, uses copy of model)
            output_dir: Output directory
            use_lora: Whether to use LoRA
            beta: DPO temperature parameter (higher = more conservative)
        """
        self.model_name = model_name_or_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.beta = beta
        
        # Initialize tokenizer
        print(f"[INFO] Loading tokenizer from: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Setup quantization
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load model
        print(f"[INFO] Loading model from: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization_config else torch.float32,
        )
        
        # Setup LoRA if enabled
        self.use_lora = use_lora
        if use_lora:
            print(f"[INFO] Setting up LoRA (r={lora_r}, alpha={lora_alpha})")
            
            if quantization_config:
                self.model = prepare_model_for_kbit_training(self.model)
            
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        # Load reference model (for DPO)
        # Reference model stays frozen and provides baseline probabilities
        if ref_model_name_or_path is None:
            ref_model_name_or_path = model_name_or_path
        
        print(f"[INFO] Loading reference model from: {ref_model_name_or_path}")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization_config else torch.float32,
        )
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Enable gradient checkpointing
        self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
    
    def prepare_dpo_dataset(
        self,
        preferences_file: str,
        max_length: int = 2048,
        include_plan: bool = True,
        include_reasoning: bool = False,
    ) -> Dataset:
        """
        Prepare dataset in DPO format.
        
        DPO expects dataset with columns:
        - prompt: The input prompt
        - chosen: Preferred completion
        - rejected: Rejected completion
        """
        # Load preference pairs
        print(f"[INFO] Loading preferences from: {preferences_file}")
        pairs = []
        with open(preferences_file, "r") as f:
            for line in f:
                pairs.append(json.loads(line))
        
        print(f"[INFO] Loaded {len(pairs)} preference pairs")
        
        # Convert to DPO format
        dpo_data = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
            "score_diff": [],
            "pair_type": []
        }
        
        for pair in pairs:
            problem = pair["problem"]
            preferred_traj = pair["preferred_trajectory"]
            rejected_traj = pair["rejected_trajectory"]
            
            # Create prompt
            prompt = f"Problem: {problem}\n"
            
            if include_plan:
                plan = preferred_traj.get("plan") or rejected_traj.get("plan")
                if plan:
                    prompt += f"\nPlan: {plan}\n"
            
            prompt += "\nSelect the best reasoning tool and compute strategy for this problem."
            
            # Create chosen response (preferred trajectory)
            chosen_tools = preferred_traj.get("selected_tools", [])
            chosen_config = preferred_traj.get("compute_configs", [{}])[0]
            
            chosen = (
                f"I will use the {chosen_tools[0] if chosen_tools else 'cot'} reasoning tool "
                f"with {chosen_config.get('strategy', 'best_of_n')}(n={chosen_config.get('param', 1)}) strategy."
            )
            
            if include_reasoning:
                chosen_answer = preferred_traj.get("final_answer", "")
                if chosen_answer:
                    chosen += f"\n\nFinal Answer: {chosen_answer}"
            
            # Create rejected response
            rejected_tools = rejected_traj.get("selected_tools", [])
            rejected_config = rejected_traj.get("compute_configs", [{}])[0]
            
            rejected = (
                f"I will use the {rejected_tools[0] if rejected_tools else 'cot'} reasoning tool "
                f"with {rejected_config.get('strategy', 'best_of_n')}(n={rejected_config.get('param', 1)}) strategy."
            )
            
            if include_reasoning:
                rejected_answer = rejected_traj.get("final_answer", "")
                if rejected_answer:
                    rejected += f"\n\nFinal Answer: {rejected_answer}"
            
            # Add to dataset
            dpo_data["prompt"].append(prompt)
            dpo_data["chosen"].append(chosen)
            dpo_data["rejected"].append(rejected)
            dpo_data["score_diff"].append(pair.get("score_diff", 0.0))
            dpo_data["pair_type"].append(pair.get("pair_type", "unknown"))
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict(dpo_data)
        
        print(f"[INFO] Created DPO dataset with {len(dataset)} examples")
        return dataset
    
    def train(
        self,
        train_preferences: str,
        val_preferences: Optional[str] = None,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        include_plan: bool = True,
        include_reasoning: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "controller-grpo",
    ):
        """
        Train using DPO (Direct Preference Optimization).
        
        Args:
            train_preferences: Path to training preference pairs
            val_preferences: Path to validation preference pairs
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
            eval_steps: Evaluation frequency
            save_steps: Save frequency
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length
            include_plan: Include planning context
            include_reasoning: Include full reasoning in responses
            use_wandb: Use Weights & Biases logging
            wandb_project: W&B project name
        """
        # Initialize W&B
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=f"grpo_{self.model_name.split('/')[-1]}",
                config={
                    "model": self.model_name,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "beta": self.beta,
                    "use_lora": self.use_lora,
                }
            )
        
        # Prepare datasets
        print("\n[INFO] Preparing training dataset...")
        train_dataset = self.prepare_dpo_dataset(
            preferences_file=train_preferences,
            max_length=max_length,
            include_plan=include_plan,
            include_reasoning=include_reasoning,
        )
        
        val_dataset = None
        if val_preferences:
            print("[INFO] Preparing validation dataset...")
            val_dataset = self.prepare_dpo_dataset(
                preferences_file=val_preferences,
                max_length=max_length,
                include_plan=include_plan,
                include_reasoning=include_reasoning,
            )
        
        # DPO Configuration
        training_args = DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=eval_steps if val_dataset else None,
            save_steps=save_steps,
            save_total_limit=3,
            eval_strategy="steps" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            fp16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            report_to="wandb" if use_wandb else "none",
            remove_unused_columns=False,
            beta=self.beta,  # DPO temperature
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )
        
        # Initialize DPO Trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
        )
        
        # Train
        print("\n" + "="*60)
        print("Starting GRPO Training (via DPO)")
        print("="*60)
        
        trainer.train()
        
        # Save final model
        print("\n[INFO] Saving final model...")
        trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))
        
        # Save training metrics
        metrics = trainer.state.log_history
        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n[SUCCESS] Training complete! Model saved to: {self.output_dir / 'final'}")
        
        if use_wandb:
            wandb.finish()
        
        return trainer


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Controller Policy")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to SFT model or base model"
    )
    parser.add_argument(
        "--ref_model_name_or_path",
        type=str,
        default=None,
        help="Reference model path (defaults to model_name_or_path)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/controller_grpo",
        help="Output directory for trained model"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_preferences",
        type=str,
        required=True,
        help="Path to training preference pairs JSONL"
    )
    parser.add_argument(
        "--val_preferences",
        type=str,
        default=None,
        help="Path to validation preference pairs JSONL"
    )
    parser.add_argument(
        "--include_plan",
        action="store_true",
        default=True,
        help="Include planning context"
    )
    parser.add_argument(
        "--include_reasoning",
        action="store_true",
        default=False,
        help="Include full reasoning in responses"
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Quantization arguments
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="controller-grpo")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model_name_or_path=args.model_name_or_path,
        ref_model_name_or_path=args.ref_model_name_or_path,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        beta=args.beta,
    )
    
    # Train
    trainer.train(
        train_preferences=args.train_preferences,
        val_preferences=args.val_preferences,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        include_plan=args.include_plan,
        include_reasoning=args.include_reasoning,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
