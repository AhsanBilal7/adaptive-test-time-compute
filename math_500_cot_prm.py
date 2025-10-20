"""
Generate Chain-of-Thought trajectories for MATH-500 dataset and score them using ReasonFlux-PRM

This script demonstrates:
1. Loading MATH-500 dataset
2. Generating trajectory-response pairs with CoT reasoning
3. Scoring trajectories using ReasonFlux-PRM
4. Selecting high-quality data based on PRM scores
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
import torch.nn.functional as F
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import re

# ============================================================================
# PART 1: Load MATH-500 Dataset
# ============================================================================

def load_math500_dataset():
    """
    Load MATH-500 dataset from HuggingFace
    
    Dataset structure:
    - problem: The math question
    - solution: Step-by-step solution
    - answer: Final answer
    - subject: Math domain (Algebra, Geometry, etc.)
    - level: Difficulty level (1-5)
    """
    print("Loading MATH-500 dataset...")
    ds = load_dataset("HuggingFaceH4/MATH-500")
    print(f"Dataset loaded: {len(ds['test'])} problems")
    return ds['test']


# ============================================================================
# PART 2: Generate Chain-of-Thought Trajectories
# ============================================================================

class CoTGenerator:
    """Generate trajectory-response pairs with chain-of-thought reasoning"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-Math-7B-Instruct"):
        """
        Initialize generator model for creating CoT reasoning
        
        Alternative models:
        - "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        - "Qwen/QwQ-32B-Preview"
        - "meta-llama/Llama-3.1-70B-Instruct"
        """
        print(f"Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
    
    def create_trajectory_prompt(self, problem: str) -> str:
        """
        Create prompt for trajectory-response generation
        
        The prompt encourages:
        1. Extended thinking trajectory (exploration, drafting ideas)
        2. Structured final response (step-by-step solution)
        """
        prompt = f"""<|im_start|>system
You are a mathematical reasoning expert. For the given problem, first think through the problem extensively in a thinking trajectory section, then provide a clear step-by-step solution.

Structure your response as:
<trajectory>
[Your extended thinking process here - explore ideas, consider approaches, work through concepts]
</trajectory>

<response>
[Your structured step-by-step solution here]
Final Answer: [Answer in \\boxed{{}} format]
</response>
<|im_end|>
<|im_start|>user
{problem}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def generate_trajectory_response(
        self, 
        problem: str,
        max_length: int = 4096,
        temperature: float = 0.7
    ) -> Dict[str, str]:
        """
        Generate trajectory-response pair for a given problem
        
        Returns:
            Dictionary with 'trajectory', 'response', 'full_output'
        """
        prompt = self.create_trajectory_prompt(problem)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse trajectory and response
        trajectory, response = self.parse_output(generated_text)
        
        return {
            'trajectory': trajectory,
            'response': response,
            'full_output': generated_text
        }
    
    def parse_output(self, text: str) -> Tuple[str, str]:
        """
        Parse generated text into trajectory and response sections
        """
        trajectory_match = re.search(
            r'<trajectory>(.*?)</trajectory>', 
            text, 
            re.DOTALL
        )
        response_match = re.search(
            r'<response>(.*?)</response>', 
            text, 
            re.DOTALL
        )
        
        trajectory = trajectory_match.group(1).strip() if trajectory_match else ""
        response = response_match.group(1).strip() if response_match else text
        
        # Fallback: if no tags found, split by heuristic
        if not trajectory and not response:
            parts = text.split('\n\n', 1)
            if len(parts) == 2:
                trajectory = parts[0]
                response = parts[1]
            else:
                response = text
        
        return trajectory, response


# ============================================================================
# PART 3: ReasonFlux-PRM Scorer
# ============================================================================

class ReasonFluxPRM:
    """
    ReasonFlux Process Reward Model for scoring trajectory-response pairs
    
    Note: This implementation follows the expected pattern based on the paper.
    The actual model might have a different interface when officially released.
    """
    
    def __init__(self, model_name="Gen-Verse/ReasonFlux-PRM-7B"):
        """
        Initialize ReasonFlux-PRM
        
        Alternative models:
        - "Gen-Verse/ReasonFlux-PRM-1.5B" (for resource-constrained settings)
        - "Qwen/Qwen2.5-Math-PRM-7B" (similar PRM as fallback)
        """
        print(f"Loading PRM model: {model_name}")
        
        # Check if ReasonFlux-PRM is available, otherwise use fallback
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load {model_name}")
            print("Falling back to Qwen2.5-Math-PRM-7B...")
            model_name = "Qwen/Qwen2.5-Math-PRM-7B"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        self.model_name = model_name
    
    def format_input(
        self, 
        problem: str, 
        trajectory: str, 
        response: str
    ) -> str:
        """
        Format problem, trajectory, and response for PRM scoring
        """
        # Combine trajectory and response with step separators
        steps = response.split('\n\n')
        formatted_steps = '\n\n'.join(steps)
        
        # Add trajectory context
        full_text = f"""Problem: {problem}

Thinking Trajectory:
{trajectory}

Solution Steps:
{formatted_steps}"""
        
        return full_text
    
    def score_trajectory_response(
        self,
        problem: str,
        trajectory: str,
        response: str
    ) -> Dict[str, float]:
        """
        Score a trajectory-response pair
        
        Returns:
            Dictionary with:
            - 'total_reward': Overall score
            - 'step_rewards': List of step-level scores
            - 'trajectory_reward': Trajectory-level score
        """
        formatted_input = self.format_input(problem, trajectory, response)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # Extract step-level rewards
        step_rewards = self.extract_step_rewards(
            logits, 
            inputs, 
            response
        )
        
        # Compute trajectory-level reward
        trajectory_reward = self.compute_trajectory_reward(
            trajectory, 
            response
        )
        
        # Aggregate rewards (beta = 0.5 for balanced weighting)
        beta = 0.5
        avg_step_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.5
        total_reward = (1 - beta) * avg_step_reward + beta * trajectory_reward
        
        return {
            'total_reward': total_reward,
            'step_rewards': step_rewards,
            'trajectory_reward': trajectory_reward,
            'avg_step_reward': avg_step_reward
        }
    
    def extract_step_rewards(
        self, 
        logits: torch.Tensor, 
        inputs: Dict,
        response: str
    ) -> List[float]:
        """
        Extract reward scores for each reasoning step
        """
        # Get probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Find step boundaries (double newlines or special tokens)
        steps = response.split('\n\n')
        num_steps = len(steps)
        
        # Extract positive class probabilities
        # For ReasonFlux-PRM, we look at the probability of correct reasoning
        step_scores = []
        
        if probabilities.size(-1) >= 2:
            # Binary classification: incorrect (0) vs correct (1)
            positive_probs = probabilities[0, :, 1]
            
            # Sample rewards uniformly across the sequence for each step
            seq_len = positive_probs.shape[0]
            step_size = max(seq_len // num_steps, 1)
            
            for i in range(num_steps):
                start_idx = i * step_size
                end_idx = min((i + 1) * step_size, seq_len)
                step_prob = positive_probs[start_idx:end_idx].mean().item()
                step_scores.append(step_prob)
        else:
            # Fallback: use uniform scores
            step_scores = [0.7] * num_steps
        
        return step_scores
    
    def compute_trajectory_reward(
        self, 
        trajectory: str, 
        response: str
    ) -> float:
        """
        Compute trajectory-level reward based on:
        1. Alignment between trajectory and response
        2. Trajectory quality (length, coherence)
        3. Template matching
        """
        # Alignment score: check if trajectory concepts appear in response
        trajectory_words = set(trajectory.lower().split())
        response_words = set(response.lower().split())
        alignment_score = len(trajectory_words & response_words) / max(len(trajectory_words), 1)
        
        # Quality score: longer, more detailed trajectories score higher
        trajectory_length = len(trajectory.split())
        quality_score = min(trajectory_length / 200, 1.0)  # Normalize by expected length
        
        # Template score: check for mathematical reasoning patterns
        reasoning_patterns = [
            'therefore', 'thus', 'hence', 'because', 'since',
            'let', 'suppose', 'assume', 'consider', 'given'
        ]
        template_matches = sum(
            1 for pattern in reasoning_patterns 
            if pattern in trajectory.lower()
        )
        template_score = min(template_matches / len(reasoning_patterns), 1.0)
        
        # Weighted combination
        trajectory_reward = (
            0.4 * alignment_score +
            0.3 * quality_score +
            0.3 * template_score
        )
        
        return trajectory_reward


# ============================================================================
# PART 4: Main Pipeline
# ============================================================================

def generate_and_score_dataset(
    dataset,
    generator: CoTGenerator,
    prm: ReasonFluxPRM,
    num_samples: int = 50,
    output_file: str = "math500_scored_trajectories.jsonl"
):
    """
    Generate CoT trajectories for MATH-500 and score them with PRM
    
    Args:
        dataset: MATH-500 dataset
        generator: CoT generator model
        prm: ReasonFlux-PRM scorer
        num_samples: Number of problems to process
        output_file: Output file for scored trajectories
    """
    print(f"\nProcessing {num_samples} problems from MATH-500...")
    
    scored_data = []
    
    for idx, sample in enumerate(tqdm(dataset.select(range(num_samples)))):
        problem = sample['problem']
        ground_truth_answer = sample['answer']
        
        # Generate trajectory-response pair
        generated = generator.generate_trajectory_response(problem)
        
        # Score with PRM
        scores = prm.score_trajectory_response(
            problem=problem,
            trajectory=generated['trajectory'],
            response=generated['response']
        )
        
        # Store results
        result = {
            'id': idx,
            'problem': problem,
            'trajectory': generated['trajectory'],
            'response': generated['response'],
            'ground_truth_answer': ground_truth_answer,
            'subject': sample.get('subject', 'Unknown'),
            'level': sample.get('level', 0),
            'prm_scores': scores
        }
        
        scored_data.append(result)
        
        # Save intermediate results every 10 samples
        if (idx + 1) % 10 == 0:
            with open(output_file, 'w') as f:
                for item in scored_data:
                    f.write(json.dumps(item) + '\n')
    
    # Save final results
    with open(output_file, 'w') as f:
        for item in scored_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nResults saved to {output_file}")
    
    return scored_data


def analyze_scores(scored_data: List[Dict]):
    """
    Analyze PRM scores and select high-quality data
    """
    print("\n" + "="*60)
    print("SCORE ANALYSIS")
    print("="*60)
    
    # Calculate statistics
    total_rewards = [item['prm_scores']['total_reward'] for item in scored_data]
    trajectory_rewards = [item['prm_scores']['trajectory_reward'] for item in scored_data]
    avg_step_rewards = [item['prm_scores']['avg_step_reward'] for item in scored_data]
    
    print(f"\nTotal Reward Statistics:")
    print(f"  Mean: {sum(total_rewards) / len(total_rewards):.4f}")
    print(f"  Min:  {min(total_rewards):.4f}")
    print(f"  Max:  {max(total_rewards):.4f}")
    
    print(f"\nTrajectory Reward Statistics:")
    print(f"  Mean: {sum(trajectory_rewards) / len(trajectory_rewards):.4f}")
    print(f"  Min:  {min(trajectory_rewards):.4f}")
    print(f"  Max:  {max(trajectory_rewards):.4f}")
    
    # Select top-k high-quality samples
    sorted_data = sorted(
        scored_data, 
        key=lambda x: x['prm_scores']['total_reward'], 
        reverse=True
    )
    
    top_k = min(10, len(sorted_data))
    print(f"\nTop {top_k} High-Quality Samples:")
    for i, item in enumerate(sorted_data[:top_k]):
        print(f"\n{i+1}. Problem: {item['problem'][:80]}...")
        print(f"   Total Reward: {item['prm_scores']['total_reward']:.4f}")
        print(f"   Trajectory Length: {len(item['trajectory'].split())} words")
        print(f"   Response Length: {len(item['response'].split())} words")
    
    return sorted_data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution pipeline
    """
    print("="*60)
    print("MATH-500 Chain-of-Thought Generation with ReasonFlux-PRM")
    print("="*60)
    
    # Step 1: Load dataset
    dataset = load_math500_dataset()
    
    # Step 2: Initialize generator
    generator = CoTGenerator(model_name="Qwen/Qwen2.5-Math-7B-Instruct")
    
    # Step 3: Initialize PRM
    prm = ReasonFluxPRM(model_name="Gen-Verse/ReasonFlux-PRM-7B")
    
    # Step 4: Generate and score trajectories
    scored_data = generate_and_score_dataset(
        dataset=dataset,
        generator=generator,
        prm=prm,
        num_samples=50,  # Process first 50 problems (adjust as needed)
        output_file="math500_scored_trajectories.jsonl"
    )
    
    # Step 5: Analyze scores
    sorted_data = analyze_scores(scored_data)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()