"""
Main Training Script for Controller Policy

This is the primary entry point for training the controller policy.
Handles the complete workflow from trajectory data to trained model.

Usage:
    # Full pipeline
    python training_main.py --config training_config.yaml
    
    # Individual stages
    python training_main.py --config training_config.yaml --stage preferences
    python training_main.py --config training_config.yaml --stage sft
    python training_main.py --config training_config.yaml --stage grpo
    python training_main.py --config training_config.yaml --stage evaluate
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Import training components
from src.trainer_files.preference_pair_generator import PreferencePairGenerator, create_preference_pairs_for_splits
from src.trainer_files.train_sft import SFTTrainer
from src.trainer_files.train_grpo import GRPOTrainer
from src.trainer_files.controller_inference import TrainedControllerPolicy
from src.trainer_files.evaluate_controller import ControllerEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='*name)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ControllerTrainingMain:
    """
    Main training orchestrator for controller policy.
    
    This class manages the complete training workflow:
    1. Load and validate configuration
    2. Generate preference pairs from trajectories
    3. Train SFT model
    4. Train GRPO model
    5. Evaluate trained controller
    6. Save results and artifacts
    """
    
    def __init__(self, config_path: str):
        """
        Initialize training pipeline.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Create output directory
        self.output_dir = Path(self.config.get('output_dir', './models'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up experiment tracking
        self.experiment_name = self.config.get(
            'experiment_name',
            f"controller_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save config to experiment directory
        with open(self.experiment_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info("="*70)
        logger.info("CONTROLLER POLICY TRAINING")
        logger.info("="*70)
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Output directory: {self.experiment_dir}")
        logger.info(f"Config: {self.config_path}")
        logger.info("="*70)
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['data', 'model']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        logger.info(f"Loaded configuration from: {self.config_path}")
        return config
    
    def _set_random_seeds(self, seed: int = None):
        """Set random seeds for reproducibility."""
        if seed is None:
            seed = self.config.get('random_seed', 42)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        logger.info(f"Set random seed to: {seed}")
    
    def _check_data_files(self) -> bool:
        """Check if required data files exist."""
        data_config = self.config['data']
        
        required_files = {
            'train_trajectories': data_config.get('train_trajectories'),
            'val_trajectories': data_config.get('val_trajectories'),
        }
        
        missing_files = []
        for name, path in required_files.items():
            if path and not Path(path).exists():
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            logger.error("Missing required data files:")
            for file in missing_files:
                logger.error(f"  - {file}")
            return False
        
        logger.info("✓ All required data files found")
        return True
    
    def stage_1_generate_preferences(self) -> bool:
        """
        Stage 1: Generate preference pairs from trajectories.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("")
        logger.info("="*70)
        logger.info("STAGE 1: Generating Preference Pairs")
        logger.info("="*70)
        
        try:
            data_config = self.config['data']
            
            # Determine data directory from trajectory files
            train_traj = Path(data_config['train_trajectories'])
            data_dir = train_traj.parent
            
            # Get preference pair configuration
            pref_config = self.config.get('preference_generation', {})
            min_score_margin = pref_config.get('min_score_margin', 0.1)
            max_pairs_per_problem = pref_config.get('max_pairs_per_problem', None)
            
            # Generate for train split
            logger.info("Generating training preference pairs...")
            train_generator = PreferencePairGenerator(
                rollouts_file=data_config['train_trajectories'],
                output_file=str(data_dir / 'preferences_train.jsonl'),
                min_score_margin=min_score_margin,
                max_pairs_per_problem=max_pairs_per_problem,
            )
            train_pairs, train_stats = train_generator.generate_and_save()
            
            # Generate for validation split if available
            if data_config.get('val_trajectories'):
                logger.info("\nGenerating validation preference pairs...")
                val_generator = PreferencePairGenerator(
                    rollouts_file=data_config['val_trajectories'],
                    output_file=str(data_dir / 'preferences_dev.jsonl'),
                    min_score_margin=min_score_margin,
                    max_pairs_per_problem=max_pairs_per_problem,
                )
                val_pairs, val_stats = val_generator.generate_and_save()
            
            # Update config with preference file paths
            self.config['data']['train_preferences'] = str(data_dir / 'preferences_train.jsonl')
            self.config['data']['val_preferences'] = str(data_dir / 'preferences_dev.jsonl')
            
            # Save updated config
            with open(self.experiment_dir / 'config.yaml', 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            logger.info("\n✓ Preference pair generation complete")
            return True
            
        except Exception as e:
            logger.error(f"Error in preference pair generation: {e}", exc_info=True)
            return False
    
    def stage_2_train_sft(self) -> Optional[Path]:
        """
        Stage 2: Train SFT model on correct trajectories.
        
        Returns:
            Path to trained model if successful, None otherwise
        """
        logger.info("")
        logger.info("="*70)
        logger.info("STAGE 2: Supervised Fine-Tuning (SFT)")
        logger.info("="*70)
        
        try:
            # Get configurations
            data_config = self.config['data']
            model_config = self.config['model']
            sft_config = self.config.get('sft', {})
            lora_config = self.config.get('lora', {})
            logging_config = self.config.get('logging', {})
            
            # Prepare output directory
            sft_output_dir = self.experiment_dir / 'sft'
            sft_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize SFT trainer
            logger.info(f"Initializing SFT trainer with model: {model_config['base_model']}")
            
            trainer = SFTTrainer(
                model_name=model_config['base_model'],
                output_dir=str(sft_output_dir),
                use_lora=lora_config.get('use_lora', True),
                lora_r=lora_config.get('r', 16),
                lora_alpha=lora_config.get('alpha', 32),
                lora_dropout=lora_config.get('dropout', 0.05),
                load_in_4bit=model_config.get('load_in_4bit', True),
                load_in_8bit=model_config.get('load_in_8bit', False),
            )
            
            # Start training
            logger.info("Starting SFT training...")
            
            trainer.train(
                train_trajectories=data_config['train_trajectories'],
                val_trajectories=data_config.get('val_trajectories'),
                batch_size=sft_config.get('batch_size', 4),
                gradient_accumulation_steps=sft_config.get('gradient_accumulation_steps', 4),
                num_epochs=sft_config.get('num_epochs', 3),
                learning_rate=sft_config.get('learning_rate', 2e-4),
                warmup_steps=sft_config.get('warmup_steps', 100),
                logging_steps=sft_config.get('logging_steps', 10),
                eval_steps=sft_config.get('eval_steps', 100),
                save_steps=sft_config.get('save_steps', 500),
                max_length=sft_config.get('max_length', 2048),
                use_correct_only=sft_config.get('use_correct_only', True),
                include_plan=sft_config.get('include_plan', True),
                format_style=sft_config.get('format_style', 'conversation'),
                use_wandb=logging_config.get('use_wandb', False),
                wandb_project=logging_config.get('wandb_project', 'controller-sft'),
            )
            
            model_path = sft_output_dir / 'final'
            logger.info(f"\n✓ SFT training complete. Model saved to: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error in SFT training: {e}", exc_info=True)
            return None
    
    def stage_3_train_grpo(self, sft_model_path: Optional[Path] = None) -> Optional[Path]:
        """
        Stage 3: Train GRPO model using preference pairs.
        
        Args:
            sft_model_path: Path to SFT model (if None, uses base model)
        
        Returns:
            Path to trained model if successful, None otherwise
        """
        logger.info("")
        logger.info("="*70)
        logger.info("STAGE 3: Group Relative Policy Optimization (GRPO)")
        logger.info("="*70)
        
        try:
            # Get configurations
            data_config = self.config['data']
            model_config = self.config['model']
            grpo_config = self.config.get('grpo', {})
            lora_config = self.config.get('lora', {})
            logging_config = self.config.get('logging', {})
            
            # Determine model to use
            if sft_model_path is None:
                sft_model_path = self.experiment_dir / 'sft' / 'final'
            
            if not sft_model_path.exists():
                logger.warning(f"SFT model not found at {sft_model_path}, using base model")
                model_name = model_config['base_model']
            else:
                model_name = str(sft_model_path)
            
            # Prepare output directory
            grpo_output_dir = self.experiment_dir / 'grpo'
            grpo_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check for preference files
            if not Path(data_config['train_preferences']).exists():
                logger.error("Training preferences not found. Run stage 1 first.")
                return None
            
            # Initialize GRPO trainer
            logger.info(f"Initializing GRPO trainer with model: {model_name}")
            
            trainer = GRPOTrainer(
                model_name_or_path=model_name,
                ref_model_name_or_path=model_name,  # Use same as reference
                output_dir=str(grpo_output_dir),
                use_lora=lora_config.get('use_lora', True),
                lora_r=lora_config.get('r', 16),
                lora_alpha=lora_config.get('alpha', 32),
                lora_dropout=lora_config.get('dropout', 0.05),
                load_in_4bit=model_config.get('load_in_4bit', True),
                load_in_8bit=model_config.get('load_in_8bit', False),
                beta=grpo_config.get('beta', 0.1),
            )
            
            # Start training
            logger.info("Starting GRPO training...")
            
            trainer.train(
                train_preferences=data_config['train_preferences'],
                val_preferences=data_config.get('val_preferences'),
                batch_size=grpo_config.get('batch_size', 4),
                gradient_accumulation_steps=grpo_config.get('gradient_accumulation_steps', 4),
                num_epochs=grpo_config.get('num_epochs', 3),
                learning_rate=grpo_config.get('learning_rate', 5e-5),
                warmup_steps=grpo_config.get('warmup_steps', 100),
                logging_steps=grpo_config.get('logging_steps', 10),
                eval_steps=grpo_config.get('eval_steps', 100),
                save_steps=grpo_config.get('save_steps', 500),
                max_length=grpo_config.get('max_length', 2048),
                max_prompt_length=grpo_config.get('max_prompt_length', 1024),
                include_plan=grpo_config.get('include_plan', True),
                include_reasoning=grpo_config.get('include_reasoning', True),
                use_wandb=logging_config.get('use_wandb', False),
                wandb_project=logging_config.get('wandb_project', 'controller-grpo'),
            )
            
            model_path = grpo_output_dir / 'final'
            logger.info(f"\n✓ GRPO training complete. Model saved to: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error in GRPO training: {e}", exc_info=True)
            return None
    
    def stage_4_evaluate(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Stage 4: Evaluate trained controller.
        
        Args:
            model_path: Path to trained model (if None, uses GRPO model)
        
        Returns:
            Evaluation results dictionary
        """
        logger.info("")
        logger.info("="*70)
        logger.info("STAGE 4: Evaluating Trained Controller")
        logger.info("="*70)
        
        try:
            # Determine model path
            if model_path is None:
                model_path = self.experiment_dir / 'grpo' / 'final'
            
            if not model_path.exists():
                logger.error(f"Model not found at {model_path}")
                return {}
            
            # Load controller
            logger.info(f"Loading controller from: {model_path}")
            
            inference_config = self.config.get('inference', {})
            controller = TrainedControllerPolicy(
                model_path=str(model_path),
                temperature=inference_config.get('temperature', 0.7),
                max_new_tokens=inference_config.get('max_new_tokens', 256),
            )
            
            # Get test problems
            eval_config = self.config.get('evaluation', {})
            test_problems = eval_config.get('test_problems', [
                "What is the sum of the first 100 positive integers?",
                "Solve the quadratic equation x^2 - 5x + 6 = 0",
                "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
                "A rectangle has length 12 and width 8. What is its area?",
                "If 3x + 5 = 20, what is the value of x?",
            ])
            
            # Test on sample problems
            logger.info("\nTesting controller on sample problems:")
            results = []
            
            for i, problem in enumerate(test_problems, 1):
                logger.info(f"\n--- Problem {i} ---")
                logger.info(f"Problem: {problem}")
                
                decision = controller.predict(problem, return_reasoning=False)
                
                logger.info(f"Tool: {decision.tool}")
                logger.info(f"Strategy: {decision.compute_strategy}(n={decision.compute_param})")
                logger.info(f"Confidence: {decision.confidence:.3f}")
                
                results.append({
                    'problem': problem,
                    'tool': decision.tool,
                    'strategy': decision.compute_strategy,
                    'param': decision.compute_param,
                    'confidence': decision.confidence,
                })
            
            # Save results
            results_file = self.experiment_dir / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"\n✓ Evaluation complete. Results saved to: {results_file}")
            
            return {'sample_results': results}
            
        except Exception as e:
            logger.error(f"Error in evaluation: {e}", exc_info=True)
            return {}
    
    def run_full_pipeline(
        self,
        skip_preferences: bool = False,
        skip_sft: bool = False,
        skip_grpo: bool = False,
        skip_evaluation: bool = False,
    ) -> bool:
        """
        Run the complete training pipeline.
        
        Args:
            skip_preferences: Skip preference pair generation
            skip_sft: Skip SFT training
            skip_grpo: Skip GRPO training
            skip_evaluation: Skip evaluation
        
        Returns:
            True if all stages completed successfully, False otherwise
        """
        logger.info("\nRunning full training pipeline...")
        
        # Check data files
        if not self._check_data_files():
            logger.error("Data file check failed. Please ensure all required files exist.")
            return False
        
        # Stage 1: Preference pairs
        if not skip_preferences:
            if not self.stage_1_generate_preferences():
                logger.error("Preference pair generation failed")
                return False
        else:
            logger.info("Skipping preference pair generation (as requested)")
        
        # Stage 2: SFT
        
        sft_model_path = Path("./models/controller_20260112_132555/sft/final")
        
        
        # Stage 3: GRPO
        grpo_model_path = None
        if not skip_grpo:
            grpo_model_path = self.stage_3_train_grpo(sft_model_path)
            if grpo_model_path is None:
                logger.error("GRPO training failed")
                return False
        else:
            logger.info("Skipping GRPO training (as requested)")
            grpo_model_path = self.experiment_dir / 'grpo' / 'final'
        
        # Stage 4: Evaluation
        if not skip_evaluation:
            results = self.stage_4_evaluate(grpo_model_path)
            if not results:
                logger.warning("Evaluation completed with warnings")
        else:
            logger.info("Skipping evaluation (as requested)")
        
        # Final summary
        logger.info("")
        logger.info("="*70)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"Experiment directory: {self.experiment_dir}")
        if sft_model_path:
            logger.info(f"SFT model: {sft_model_path}")
        if grpo_model_path:
            logger.info(f"GRPO model: {grpo_model_path}")
        logger.info("="*70)
        
        return True


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Controller Policy Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python training_main.py --config training_config.yaml
  
  # Run specific stage
  python training_main.py --config training_config.yaml --stage sft
  
  # Skip certain stages
  python training_main.py --config training_config.yaml --skip-sft --skip-grpo
  
  # Custom experiment name
  python training_main.py --config training_config.yaml --experiment my_experiment
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    # Stage control
    parser.add_argument(
        '--stage',
        type=str,
        choices=['preferences', 'sft', 'grpo', 'evaluate', 'all'],
        default='all',
        help='Run specific stage only (default: all)'
    )
    
    # Skip flags
    parser.add_argument('--skip-preferences', action='store_true',
                       help='Skip preference pair generation')
    parser.add_argument('--skip-sft', action='store_true',
                       help='Skip SFT training')
    parser.add_argument('--skip-grpo', action='store_true',
                       help='Skip GRPO training')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation')
    
    # Optional arguments
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated timestamp)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize training pipeline
        trainer = ControllerTrainingMain(args.config)
        
        # Override config if specified
        if args.experiment:
            trainer.experiment_name = args.experiment
            trainer.experiment_dir = trainer.output_dir / args.experiment
            trainer.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        if args.output_dir:
            trainer.output_dir = Path(args.output_dir)
            trainer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run appropriate stage(s)
        success = False
        
        if args.stage == 'preferences':
            success = trainer.stage_1_generate_preferences()
        
        elif args.stage == 'sft':
            model_path = trainer.stage_2_train_sft()
            success = model_path is not None
        
        elif args.stage == 'grpo':
            model_path = trainer.stage_3_train_grpo()
            success = model_path is not None
        
        elif args.stage == 'evaluate':
            results = trainer.stage_4_evaluate()
            success = len(results) > 0
        
        else:  # 'all'
            success = trainer.run_full_pipeline(
                skip_preferences=args.skip_preferences,
                skip_sft=args.skip_sft,
                skip_grpo=args.skip_grpo,
                skip_evaluation=args.skip_evaluation,
            )
        
        # Exit with appropriate code
        if success:
            logger.info("\n✓ Training completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n✗ Training failed!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()