"""
Inference Module for Trained Controller Policy

Loads trained model and provides interface for:
1. Tool selection
2. Compute strategy selection
3. Integration with the agentic framework
"""

import re
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class ControllerDecision:
    """Controller's decision for a given problem."""
    tool: str  # "cot", "self_reflection", etc.
    compute_strategy: str  # "best_of_n", "beam_search", "lookahead"
    compute_param: int  # n value or beam width
    confidence: float  # Model's confidence in decision
    reasoning: Optional[str] = None  # Optional reasoning trace


class TrainedControllerPolicy:
    """
    Inference interface for trained controller policy.
    
    Provides methods to:
    - Select optimal reasoning tool for a problem
    - Select optimal compute strategy and parameters
    - Integrate with existing UniversalAgent
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_4bit: bool = True,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        """
        Initialize trained controller.
        
        Args:
            model_path: Path to trained model (SFT or GRPO)
            device: Device to load model on
            load_in_4bit: Whether to use 4-bit quantization
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = Path(model_path)
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        print(f"[INFO] Loading controller from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if quantization_config else torch.float32,
        )
        
        self.model.eval()
        
        print(f"[SUCCESS] Controller loaded successfully")
    
    def predict(
        self,
        problem: str,
        plan: Optional[str] = None,
        return_reasoning: bool = False,
    ) -> ControllerDecision:
        """
        Predict optimal tool and compute strategy for a problem.
        
        Args:
            problem: Mathematical problem text
            plan: Optional planning context
            return_reasoning: Whether to return model's reasoning
        
        Returns:
            ControllerDecision with tool, strategy, and parameters
        """
        # Format prompt
        prompt = self._format_prompt(problem, plan)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode response
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse response to extract tool and strategy
        decision = self._parse_response(response)
        
        # Compute confidence from generation scores
        if hasattr(outputs, "scores") and outputs.scores:
            decision.confidence = self._compute_confidence(outputs.scores)
        else:
            decision.confidence = 0.5  # Default
        
        if return_reasoning:
            decision.reasoning = response
        
        return decision
    
    def _format_prompt(self, problem: str, plan: Optional[str] = None) -> str:
        """Format input prompt for controller."""
        prompt = f"<|user|>\nProblem: {problem}\n"
        
        if plan:
            prompt += f"\nPlan: {plan}\n"
        
        prompt += "\nSelect the best reasoning tool and compute strategy for this problem.\n<|assistant|>\n"
        
        return prompt
    
    def _parse_response(self, response: str) -> ControllerDecision:
        """
        Parse model response to extract tool and compute strategy.
        
        Expected format:
        "I will use the {tool} reasoning tool with {strategy}(n={param}) strategy."
        """
        # Default values
        tool = "cot"
        strategy = "best_of_n"
        param = 1
        
        # Extract tool
        tool_pattern = r"(?:use the|use)\s+(\w+(?:_\w+)*)\s+(?:reasoning\s+)?tool"
        tool_match = re.search(tool_pattern, response, re.IGNORECASE)
        if tool_match:
            tool = tool_match.group(1).lower()
        
        # Extract strategy and param
        strategy_pattern = r"(\w+(?:_\w+)*)\s*\(\s*n?\s*=?\s*(\d+)\s*\)"
        strategy_match = re.search(strategy_pattern, response, re.IGNORECASE)
        if strategy_match:
            strategy = strategy_match.group(1).lower()
            param = int(strategy_match.group(2))
        else:
            # Try alternative pattern without parentheses
            alt_pattern = r"(\w+(?:_\w+)*)\s+(?:with|strategy)\s+n\s*=\s*(\d+)"
            alt_match = re.search(alt_pattern, response, re.IGNORECASE)
            if alt_match:
                strategy = alt_match.group(1).lower()
                param = int(alt_match.group(2))
        
        # Validate and normalize
        valid_tools = ["cot", "self_reflection", "heuristic_script"]
        if tool not in valid_tools:
            # Try to map to valid tool
            if "reflect" in tool or "reflection" in tool:
                tool = "self_reflection"
            elif "cot" in tool or "chain" in tool:
                tool = "cot"
            else:
                tool = "cot"  # Default
        
        valid_strategies = ["best_of_n", "beam_search", "lookahead"]
        if strategy not in valid_strategies:
            # Try to map
            if "beam" in strategy:
                strategy = "beam_search"
            elif "look" in strategy or "ahead" in strategy:
                strategy = "lookahead"
            else:
                strategy = "best_of_n"  # Default
        
        # Clamp param to reasonable range
        param = max(1, min(param, 16))
        
        return ControllerDecision(
            tool=tool,
            compute_strategy=strategy,
            compute_param=param,
            confidence=0.0  # Will be set later
        )
    
    def _compute_confidence(self, scores) -> float:
        """
        Compute confidence score from generation probabilities.
        
        Uses average probability of generated tokens as confidence.
        """
        if not scores:
            return 0.5
        
        # Convert scores to probabilities
        probs = []
        for score in scores[:10]:  # Use first 10 tokens
            prob = torch.softmax(score[0], dim=-1)
            max_prob = prob.max().item()
            probs.append(max_prob)
        
        # Average probability
        confidence = sum(probs) / len(probs) if probs else 0.5
        
        return confidence
    
    def get_tool_and_compute_config(
        self,
        problem: str,
        plan: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get tool name and compute config dict.
        
        Returns format compatible with UniversalAgent:
            tool: str
            compute_config: {"strategy": str, "param": int}
        """
        decision = self.predict(problem, plan, return_reasoning=False)
        
        compute_config = {
            "strategy": decision.compute_strategy,
            "param": decision.compute_param
        }
        
        return decision.tool, compute_config
    
    def batch_predict(
        self,
        problems: list,
        plans: Optional[list] = None,
        batch_size: int = 8,
    ) -> list:
        """
        Batch prediction for multiple problems.
        
        Args:
            problems: List of problem texts
            plans: Optional list of plans (same length as problems)
            batch_size: Batch size for inference
        
        Returns:
            List of ControllerDecisions
        """
        if plans is None:
            plans = [None] * len(problems)
        
        decisions = []
        
        for i in range(0, len(problems), batch_size):
            batch_problems = problems[i:i+batch_size]
            batch_plans = plans[i:i+batch_size]
            
            # Format prompts
            prompts = [
                self._format_prompt(prob, plan)
                for prob, plan in zip(batch_problems, batch_plans)
            ]
            
            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode and parse
            for j, output in enumerate(outputs):
                generated_ids = output[inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                decision = self._parse_response(response)
                decision.confidence = 0.5  # Approximate for batch
                decisions.append(decision)
        
        return decisions


class ControllerIntegrator:
    """
    Integrates trained controller policy with UniversalAgent.
    
    Replaces ToolSelector and ComputeSelector with learned policy.
    """
    
    def __init__(
        self,
        controller: TrainedControllerPolicy,
        fallback_tool: str = "cot",
        fallback_compute: Dict[str, Any] = None,
        min_confidence: float = 0.3,
    ):
        """
        Initialize integrator.
        
        Args:
            controller: Trained controller policy
            fallback_tool: Fallback tool if confidence is low
            fallback_compute: Fallback compute config
            min_confidence: Minimum confidence to use controller decision
        """
        self.controller = controller
        self.fallback_tool = fallback_tool
        self.fallback_compute = fallback_compute or {"strategy": "best_of_n", "param": 1}
        self.min_confidence = min_confidence
    
    def select_tool_and_compute(
        self,
        problem: str,
        plan: Optional[str] = None,
        use_fallback_on_low_confidence: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select tool and compute strategy using controller.
        
        Returns:
            (tool_name, compute_config)
        """
        decision = self.controller.predict(problem, plan, return_reasoning=False)
        
        # Use fallback if confidence is too low
        if use_fallback_on_low_confidence and decision.confidence < self.min_confidence:
            print(f"[WARNING] Low confidence ({decision.confidence:.2f}), using fallback")
            return self.fallback_tool, self.fallback_compute
        
        compute_config = {
            "strategy": decision.compute_strategy,
            "param": decision.compute_param
        }
        
        return decision.tool, compute_config


def create_controller_from_checkpoint(
    checkpoint_path: str,
    **kwargs
) -> TrainedControllerPolicy:
    """
    Convenience function to create controller from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        **kwargs: Additional arguments for TrainedControllerPolicy
    
    Returns:
        Initialized TrainedControllerPolicy
    """
    return TrainedControllerPolicy(
        model_path=checkpoint_path,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test trained controller policy")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="What is the sum of the first 100 positive integers?",
        help="Problem to solve"
    )
    
    args = parser.parse_args()
    
    # Load controller
    controller = TrainedControllerPolicy(
        model_path=args.model_path,
        temperature=0.7,
    )
    
    # Make prediction
    decision = controller.predict(
        problem=args.problem,
        return_reasoning=True
    )
    
    print(f"\n[CONTROLLER DECISION]")
    print(f"Tool: {decision.tool}")
    print(f"Compute Strategy: {decision.compute_strategy}(n={decision.compute_param})")
    print(f"Confidence: {decision.confidence:.3f}")
    
    if decision.reasoning:
        print(f"\nReasoning:\n{decision.reasoning}")
