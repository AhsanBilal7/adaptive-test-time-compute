import re
from typing import Optional, Dict, List, Any

from datasets import load_dataset

from src.universal_agent import UniversalAgent, UniversalResponse


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
        
        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
        if match:
            gold_answer = match.group(1).replace(',', '')
        else:
            gold_answer = answer_text
        
        problems.append({
            "problem_id": idx,
            "problem": question,
            "gold_answer_raw": answer_text,
            "gold_answer_extracted": gold_answer,
            "gold_answer_normalized": gold_answer.strip(),
            "problem_type": "arithmetic",
            "level": "grade_school",
        })
    
    return problems


def check_gsm8k_answer_equivalence(pred: str, gold: str) -> bool:
    pred_match = re.search(r'Final Answer: The final answer is\s*([^.]+)', str(pred))
    if pred_match:
        pred = pred_match.group(1).strip()
    
    pred = str(pred).replace(',', '').replace('$', '').strip()
    gold = str(gold).replace(',', '').replace('$', '').strip()
    
    if pred == gold:
        return True
    
    try:
        pred_num = float(pred)
        gold_num = float(gold)
        return abs(pred_num - gold_num) < 1e-6
    except (ValueError, TypeError):
        return False


def extract_gsm8k_prediction(response: UniversalResponse, problem_data: Dict) -> str:
    pred = response.answer_unstructured
    
    patterns = [
        r'Final Answer: The final answer is\s*([^.]+)',
        r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'answer is\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, pred)
        if match:
            return match.group(1).strip()
    
    return response.answer


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
    
    def solve(self, problem: str) -> UniversalResponse:
        return self.agent.solve(problem, self.prompts)
    
    def reset(self):
        self.agent.reset()
    
    def get_stats(self):
        return self.agent.get_stats()
