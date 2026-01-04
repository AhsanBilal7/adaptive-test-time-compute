"""
Supervised Fine-Tuning (SFT) for Controller Policy
Trains the model to predict optimal tool and compute strategy selections
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from src.trainer_files.training_datasets import ControllerSFTDataset, create_dataloaders


class SFTTrainer:
    """
    SFT Trainer for Controller Policy.
    
    Trains the model using supervised learning on correct trajectories
    to learn optimal tool and compute strategy selections.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_dir: str = "./models/controller_sft",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_lora = use_lora
        self.lora_config = None
        
        # Initialize tokenizer
        print(f"[INFO] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Initialize model
        print(f"[INFO] Loading model: {model_name}")
        
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
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization_config else torch.float32,
        )
        
        # Setup LoRA if enabled
        if use_lora:
            print(f"[INFO] Setting up LoRA (r={lora_r}, alpha={lora_alpha})")
            
            if quantization_config:
                self.model = prepare_model_for_kbit_training(self.model)
            
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing for memory efficiency
        self.model.config.use_cache = False
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
    
    def train(
        self,
        train_trajectories: str,
        val_trajectories: Optional[str] = None,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        max_length: int = 2048,
        use_correct_only: bool = True,
        include_plan: bool = True,
        format_style: str = "conversation",
        use_wandb: bool = False,
        wandb_project: str = "controller-sft",
    ):
        """
        Train the controller policy using supervised fine-tuning.
        
        Args:
            train_trajectories: Path to training trajectories JSONL
            val_trajectories: Path to validation trajectories JSONL
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Learning rate warmup steps
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            max_length: Maximum sequence length
            use_correct_only: Use only correct trajectories
            include_plan: Include planning in training data
            format_style: "conversation" or "instruction"
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        # Initialize W&B if enabled
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=f"sft_{self.model_name.split('/')[-1]}",
                config={
                    "model": self.model_name,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "use_lora": self.use_lora,
                    "use_correct_only": use_correct_only,
                }
            )
        
        # Create datasets
        print("\n[INFO] Creating training dataset...")
        train_dataset = ControllerSFTDataset(
            trajectories_file=train_trajectories,
            tokenizer=self.tokenizer,
            max_length=max_length,
            use_correct_only=use_correct_only,
            include_plan=include_plan,
            format_style=format_style,
        )
        
        val_dataset = None
        if val_trajectories:
            print("[INFO] Creating validation dataset...")
            val_dataset = ControllerSFTDataset(
                trajectories_file=val_trajectories,
                tokenizer=self.tokenizer,
                max_length=max_length,
                use_correct_only=use_correct_only,
                include_plan=include_plan,
                format_style=format_style,
            )
        
        # Setup training arguments
        training_args = TrainingArguments(
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
            save_safetensors=True,
            remove_unused_columns=True,  # This removes non-standard columns
        )
        
        # Data collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\n" + "="*60)
        print("Starting SFT Training")
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
    
    def save_model(self, save_path: str):
        """Save model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        print(f"[INFO] Model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Controller Policy")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/controller_sft",
        help="Output directory for trained model"
    )
    
    # Data arguments
    parser.add_argument(
        "--train_trajectories",
        type=str,
        required=True,
        help="Path to training trajectories JSONL file"
    )
    parser.add_argument(
        "--val_trajectories",
        type=str,
        default=None,
        help="Path to validation trajectories JSONL file"
    )
    parser.add_argument(
        "--use_correct_only",
        action="store_true",
        default=True,
        help="Use only correct trajectories for training"
    )
    parser.add_argument(
        "--include_plan",
        action="store_true",
        default=True,
        help="Include planning in training data"
    )
    parser.add_argument(
        "--format_style",
        type=str,
        choices=["conversation", "instruction"],
        default="conversation",
        help="Data formatting style"
    )
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=2048)
    
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
    parser.add_argument("--wandb_project", type=str, default="controller-sft")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SFTTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    
    # Train
    trainer.train(
        train_trajectories=args.train_trajectories,
        val_trajectories=args.val_trajectories,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_length=args.max_length,
        use_correct_only=args.use_correct_only,
        include_plan=args.include_plan,
        format_style=args.format_style,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()