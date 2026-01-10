import re
import signal
from typing import Optional, Dict, List, Any

from datasets import load_dataset
import sympy
from sympy.parsing.sympy_parser import parse_expr
import logging

from src.universal_agent import UniversalAgent, UniversalResponse

eval_logger = logging.getLogger(__name__)


def get_unnormalized_answer(text: str) -> str:
    """Extract answer from text, similar to MATH implementation."""
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?)\. I hope it is correct\.",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
]

REMOVED_EXPRESSIONS = [
    "dollars", "mph", "inches", "ft", "hours", "km", "units", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "minutes", "days",
    "weeks", "months", "years", "\\text{}", "\\$",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize GSM8K answers by removing formatting."""
    if final_answer is None:
        return ""
    
    # Extract from equations if present
    final_answer = final_answer.split("=")[-1]
    
    # Apply substitutions
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    
    # Remove common expressions
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    
    # Clean up text formatting
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    
    # Remove dollar signs and clean up
    final_answer = final_answer.replace("$", "")
    final_answer = final_answer.strip()
    
    # Remove commas from numbers
    if final_answer.replace(",", "").replace(".", "").replace("-", "").isdigit():
        final_answer = final_answer.replace(",", "")
    
    return final_answer


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """Check numerical equivalence for GSM8K answers."""
    try:
        with timeout(seconds=5):
            # First try direct numerical comparison
            try:
                x1_normalized = normalize_final_answer(x1)
                x2_normalized = normalize_final_answer(x2)
                
                # Try exact string match
                if x1_normalized == x2_normalized:
                    return True
                
                # Try numerical comparison
                num1 = float(x1_normalized)
                num2 = float(x2_normalized)
                return abs(num1 - num2) < 1e-6
            except (ValueError, TypeError):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2} as number")
            
            # Try sympy parsing as fallback
            try:
                parsed_x1 = parse_expr(x1_normalized)
                parsed_x2 = parse_expr(x2_normalized)
            except (sympy.SympifyError, TypeError, SyntaxError):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2} with sympy")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def load_gsm8k_dataset(
    split: str = "test",
    max_problems: Optional[int] = None,
) -> List[Dict[str, Any]]:
    print(f"[INFO] Loading GSM8K dataset from Hugging Face")
    
    ds = load_dataset("openai/gsm8k", "main")
    dataset = ds[split]
    
    print(f"[INFO] Loaded {len(dataset)} problems from {split} split")
    
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
        print(f"[INFO] Limited to {len(dataset)} problems")
    
    problems = []
    for idx in range(len(dataset)):
        problem_data = dataset[idx]
        
        question = problem_data["question"]
        answer_text = problem_data["answer"]
        
        # Extract gold answer from #### format
        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
        if match:
            gold_answer_extracted = match.group(1).replace(',', '')
        else:
            # Fallback: try to extract any number
            numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
            gold_answer_extracted = numbers[-1].replace(',', '') if numbers else answer_text
        
        gold_answer_normalized = normalize_final_answer(gold_answer_extracted)
        
        problems.append({
            "problem_id": idx,
            "problem": question,
            "gold_answer_raw": answer_text,
            "gold_answer_extracted": gold_answer_extracted,
            "gold_answer_normalized": gold_answer_normalized,
            "problem_type": "arithmetic",
            "level": "grade_school",
            "solution": answer_text,
        })
    
    return problems


def check_gsm8k_answer_equivalence(pred: str, gold: str) -> bool:
    """Check if prediction matches gold answer."""
    # Extract unnormalized prediction
    pred_extracted = get_unnormalized_answer(pred)
    
    if pred_extracted == "[invalidanswer]":
        pred_extracted = pred
    
    # Normalize both answers
    pred_normalized = normalize_final_answer(pred_extracted)
    gold_normalized = normalize_final_answer(gold)
    
    # Check exact match first
    if pred_normalized == gold_normalized:
        return True
    
    # Check numerical equivalence
    return is_equiv(pred_normalized, gold_normalized)


def extract_gsm8k_prediction(response: UniversalResponse, problem_data: Dict) -> str:
    """Extract prediction from UniversalResponse."""
    predicted_answer = get_unnormalized_answer(response.answer_unstructured)
    
    if predicted_answer == "[invalidanswer]":
        predicted_answer = str(response.answer)
    
    return predicted_answer


class GSM8KCore:
    def __init__(self, agent: UniversalAgent, prompts: Dict[str, str]):
        self.agent = agent
        self.prompts = prompts
    
    @staticmethod
    def load_dataset(
        split: str = "test",
        max_problems: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return load_gsm8k_dataset(split, max_problems)
    
    @staticmethod
    def check_answer(pred: str, gold: str) -> bool:
        return check_gsm8k_answer_equivalence(pred, gold)
    
    @staticmethod
    def extract_prediction(response: UniversalResponse, problem_data: Dict) -> str:
        return extract_gsm8k_prediction(response, problem_data)
    
    @staticmethod
    def get_last_dollar_normalize_final_answer(answer: str) -> str:
        """
        Returns the last numerical value in the answer.
        Handles $ formatting and extracts clean numbers.
        """
        # Look for numbers with optional $ prefix and commas
        numbers = re.findall(r'\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer)
        if numbers:
            # Return last number, normalized
            return numbers[-1].replace(',', '').strip()
        return None
    
    def solve(self, problem: str) -> UniversalResponse:
        return self.agent.solve(problem, self.prompts)
    
    def reset(self):
        self.agent.reset()
    
    def get_stats(self):
        return self.agent.get_stats()