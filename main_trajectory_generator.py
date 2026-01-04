import argparse
import yaml
from omegaconf import DictConfig

from BALROG.balrog.client import create_llm_client
from src.help_functions.prompts_templates import *
from src.dataset_generation.trajectory_generator import *


if __name__ == "__main__":


    def load_config(config_path: str) -> dict:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    parser = argparse.ArgumentParser(description="Training data generation / pipeline runner")
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Build client_factory using the config["client"] section
    client_factory = create_llm_client(DictConfig(config["client"]))


    prompts = {
        "system_prompt": MATH_SYSTEM_PROMPT,
        "planning_prompt_template": PLANNING_PROMPT_TEMPLATE,
        "tool_selector_prompt": TOOL_SELECTOR_PROMPT,
        "tool_selector_system_prompt": TOOL_SELECTOR_SYSTEM_PROMPT,
        "compute_selector_prompt": COMPUTE_SELECTOR_PROMPT,
        "compute_selector_system_prompt": COMPUTE_SELECTOR_SYSTEM_PROMPT,
        "self_reflection_instruction_prompt": SELF_REFLECTION_INSTRUCTION_PROMPT,
        "cot_instruction_prompt": COT_INSTRUCTION_PROMPT,
        "prm_scoring_prompt": PRM_SCORING_PROMPT,
        "final_answer_system_prompt": FINAL_ANSWER_SYSTEM_PROMPT,
        "final_answer_user_prompt": FINAL_ANSWER_USER_PROMPT,
        "unstructured_final_answer_system_prompt": UNSTRUCTURED_FINAL_ANSWER_SYSTEM_PROMPT,
        "unstructured_final_answer_user_prompt": UNSTRUCTURED_FINAL_ANSWER_USER_PROMPT,
        "direct_solve_prompt": DIRECT_SOLVE_PROMPT,
        "direct_solve_system_prompt": DIRECT_SOLVE_SYSTEM_PROMPT,
    }

    run_full_pipeline(
        client_factory=client_factory,
        prompts=prompts,
        output_dir="./training_data",
        max_train_problems=2,
        max_dev_problems=1,
        k_rollouts=10,
        num_epochs=10,
        batch_size=32
    )




    # # ===== STAGE 2: Training =====
    # print("\n" + "="*60)
    # print("STAGE 2: Training")
    # print("="*60)
    
    # # Load datasets
    # train_dataset = TrajectoryDataset(
    #     rollouts_file=output_dir / "rollouts_train.jsonl",
    #     prefs_file=output_dir / "prefs_train.jsonl"
    # )
    
    # val_dataset = TrajectoryDataset(
    #     rollouts_file=output_dir / "rollouts_dev.jsonl",
    #     prefs_file=output_dir / "prefs_dev.jsonl"
    # )
    

