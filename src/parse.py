# src/parse.py
import re
from typing import Tuple


def split_reason_answer(text: str) -> Tuple[str, str]:
    """
    Split text into reasoning and answer parts.
    Expects format: "Reasoning: ... Answer: ..."
    """
    text = text.strip()
    
    # Try to find Answer: marker
    answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
    
    if answer_match:
        # Split at the Answer: marker
        answer_start = answer_match.start()
        reasoning = text[:answer_start].strip()
        answer = answer_match.group(1).strip()
        
        # Clean up reasoning
        reasoning = re.sub(r'^Reasoning:\s*', '', reasoning, flags=re.IGNORECASE).strip()
        
        return reasoning, answer
    
    # No clear Answer: marker found
    # Try to extract from end of text
    lines = text.split('\n')
    if lines:
        potential_answer = lines[-1].strip()
        reasoning = '\n'.join(lines[:-1]).strip()
        reasoning = re.sub(r'^Reasoning:\s*', '', reasoning, flags=re.IGNORECASE).strip()
        return reasoning, potential_answer
    
    return text, ""


def has_answer_marker(text: str, kind: str = "math") -> bool:
    """
    Check if text contains an answer marker like "Answer:" or "Final answer:"
    """
    text_lower = text.lower()
    markers = ["answer:", "final answer:", "the answer is"]
    return any(marker in text_lower for marker in markers)


def extract_final_answer(text: str, kind: str = "math") -> str:
    """
    Extract the final answer from a reasoning chain.
    Looks for "Answer:" marker or tries to extract from the end.
    Handles both full reasoning chains and clean answer-only responses.
    """
    # First try to split normally
    _, answer = split_reason_answer(text)
    
    # If we got a clean answer, return it
    if answer and answer.strip():
        return answer
    
    # Try to find "Final Answer:" marker (from our final extraction step)
    final_match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)
    if final_match:
        return final_match.group(1).strip()
    
    # Try alternative patterns
    patterns = [
        r'(?:final answer|answer|therefore)[:=\s]+(.+?)(?:\.|$)',
        r'(?:the answer is)[:=\s]+(.+?)(?:\.|$)',
        r'=\s*([0-9.,]+)\s*(?:$|\n)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Remove any trailing punctuation or comments
            answer = re.sub(r'[.\s]+$', '', answer)
            return answer
    
    # If still nothing, try to get last line as answer
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # Remove "Answer:" prefix if present
        last_line = re.sub(r'^(?:Final\s+)?Answer:\s*', '', last_line, flags=re.IGNORECASE)
        return last_line.strip()
    
    return ""


def parse_gsm8k_numeric(answer_str: str) -> str:
    """
    Parse GSM8K-style answers to extract the numeric value.
    Handles formats like:
    - "#### 42"
    - "42"
    - "The answer is 42."
    - "42 apples" -> "42"
    """
    if not answer_str:
        return ""
    
    answer_str = str(answer_str).strip()
    
    # GSM8K format: #### number
    if "####" in answer_str:
        match = re.search(r'####\s*([-+]?[0-9.,]+)', answer_str)
        if match:
            return match.group(1).replace(",", "")
    
    # Look for "Answer: number" or similar
    match = re.search(r'(?:answer|final answer|result)[:=\s]+([-+]?[0-9.,]+)', answer_str, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for just a number (possibly with units)
    match = re.search(r'([-+]?[0-9.,]+)', answer_str)
    if match:
        return match.group(1).replace(",", "")
    
    # Return as-is if no number found
    return answer_str.lower().strip()


def parse_yesno(answer_str: str) -> str:
    """
    Parse yes/no answers, normalizing to lowercase "yes" or "no".
    """
    if not answer_str:
        return ""
    
    answer_str = str(answer_str).strip().lower()
    
    # Direct yes/no
    if answer_str.startswith("yes"):
        return "yes"
    if answer_str.startswith("no"):
        return "no"
    
    # Check for True/False
    if answer_str in ["true", "t"]:
        return "yes"
    if answer_str in ["false", "f"]:
        return "no"
    
    # Look for yes/no in the text
    if "yes" in answer_str:
        return "yes"
    if "no" in answer_str:
        return "no"
    
    return answer_str


def parse_aime_answer(answer_str: str) -> str:
    """
    Parse AIME-style answers. AIME answers are typically integers from 0-999.
    """
    if not answer_str:
        return ""
    
    answer_str = str(answer_str).strip()
    
    # Look for #### format (if preprocessed like GSM8K)
    if "####" in answer_str:
        match = re.search(r'####\s*([0-9]+)', answer_str)
        if match:
            return match.group(1)
    
    # Look for "Answer: number" or similar
    match = re.search(r'(?:answer|final answer)[:=\s]+([0-9]+)', answer_str, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Look for the last number in the string
    numbers = re.findall(r'\b([0-9]+)\b', answer_str)
    if numbers:
        return numbers[-1]
    
    return answer_str.strip()


def normalize_answer(answer_str: str, task: str = "math") -> str:
    """
    Normalize an answer based on the task type.
    """
    if task == "math":
        return parse_gsm8k_numeric(answer_str)
    elif task == "qa":
        return parse_yesno(answer_str)
    elif task == "aime":
        return parse_aime_answer(answer_str)
    else:
        return str(answer_str).strip().lower()