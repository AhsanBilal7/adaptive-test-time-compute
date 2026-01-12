import re
import signal
from typing import Optional, Dict, List, Any

from datasets import load_dataset
import sympy
from sympy.parsing.latex import parse_latex
import logging
from src.universal_agent import UniversalAgent, UniversalResponse

eval_logger = logging.getLogger(__name__)


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if s is None:
        return ""
    
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    
    if not s.startswith(left):
        return s

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours",
    "km", "units", "\\ldots", "sue", "points", "feet", "minutes", "digits",
    "cents", "degrees", "cm", "gm", "pounds", "meters", "meals", "edges",
    "students", "childrentickets", "multiples", "\\text{s}", "\\text{.}",
    "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    if final_answer is None:
        return ""
    
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
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
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
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


# def get_unnormalized_answer(text: str) -> str:
#     INVALID_ANSWER = "[invalidanswer]"
#     end_seq = "I hope it is correct."
#     text += end_seq
#     match = re.search(
#         r"Final Answer: The final answer is(.*?)\. I hope it is correct\.",
#         text,
#     )
#     if match:
#         return match.group(1).strip()
#     else:
#         return INVALID_ANSWER
def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"

    # Look for the final boxed answer:  \boxed{...}
    match = re.findall(r"\\boxed\{([^}]*)\}", text)

    # Valid only if exactly one boxed answer is found
    if len(match) == 1:
        return match[0].strip()

    return INVALID_ANSWER




def load_math_dataset(
    split: str = "train",
    problem_types: Optional[List[str]] = None,
    difficulty_levels: Optional[List[str]] = None,
    max_problems: Optional[int] = None,
) -> List[Dict[str, Any]]:
    print(f"[INFO] Loading MATH dataset from Hugging Face (HuggingFaceH4/MATH-500)")
    
    ds = load_dataset("HuggingFaceH4/MATH-500")
    
    print(f"[INFO] Dataset splits available: {list(ds.keys())}")
    dataset = ds[split]
    
    print(f"[INFO] Loaded {len(dataset)} problems from {split} split")
    
    if problem_types:
        dataset = dataset.filter(lambda x: x["subject"] in problem_types)
        print(f"[INFO] Filtered to {len(dataset)} problems of subjects: {problem_types}")
    
    if difficulty_levels:
        dataset = dataset.filter(lambda x: x["level"] in difficulty_levels)
        print(f"[INFO] Filtered to {len(dataset)} problems of levels: {difficulty_levels}")
    
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
        print(f"[INFO] Limited to {len(dataset)} problems")
    
    problems = []
    for idx in range(len(dataset)):
        problem_data = dataset[idx]
        
        problem = problem_data["problem"]
        gold_answer_raw = problem_data["answer"]
        problem_type = problem_data["subject"]
        level = problem_data["level"]
        solution = problem_data["solution"]
        
        boxed_answer = last_boxed_only_string(solution)
        if boxed_answer:
            gold_answer_extracted = remove_boxed(boxed_answer)
        else:
            gold_answer_extracted = gold_answer_raw
        
        gold_answer_normalized = normalize_final_answer(gold_answer_extracted)
        
        problems.append({
            "problem_id": idx,
            "problem": problem,
            "gold_answer_raw": gold_answer_raw,
            "gold_answer_extracted": gold_answer_extracted,
            "gold_answer_normalized": gold_answer_normalized,
            "problem_type": problem_type,
            "level": level,
            "solution": solution,
        })
    
    return problems


def check_math_answer_equivalence(pred: str, gold: str) -> bool:
    pred_extracted = get_unnormalized_answer(pred)
    
    if pred_extracted == "[invalidanswer]":
        pred_extracted = pred
    
    pred_normalized = normalize_final_answer(pred_extracted)
    gold_normalized = normalize_final_answer(gold)
    
    if pred_normalized == gold_normalized:
        return True
    
    return is_equiv(pred_normalized, gold_normalized)


def extract_math_prediction(response: UniversalResponse, problem_data: Dict) -> str:
    predicted_answer = get_unnormalized_answer(response.answer_unstructured)
    
    if predicted_answer == "[invalidanswer]":
        predicted_answer = None
    
    return predicted_answer


class MATHCore:
    def __init__(self, agent: UniversalAgent, prompts: Dict[str, str]):
        self.agent = agent
        self.prompts = prompts
    
    @staticmethod
    def load_dataset(
        split: str = "train",
        problem_types: Optional[List[str]] = None,
        difficulty_levels: Optional[List[str]] = None,
        max_problems: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return load_math_dataset(split, problem_types, difficulty_levels, max_problems)
    
    @staticmethod
    def check_answer(pred: str, gold: str) -> bool:
        return check_math_answer_equivalence(pred, gold)
    
    @staticmethod
    def extract_prediction(response: UniversalResponse, problem_data: Dict) -> str:
        return extract_math_prediction(response, problem_data)
    

    @staticmethod
    def get_last_dollar_normalize_final_answer(answer: str) -> str:
        """
        Returns the last LaTeX expression enclosed in $...$ or $$...$$.
        Ignores escaped dollars like \$
        """

        # 1) display math: $$...$$  (DOTALL so it spans newlines)
        disp = list(re.finditer(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", answer, flags=re.DOTALL))

        # 2) inline math: $...$ (but not $$...$$)
        # Negative lookahead/lookbehind to avoid grabbing the $$ delimiters
        inline = list(re.finditer(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", answer, flags=re.DOTALL))
        # Combine, pick the last by end position
        all_matches = disp + inline
        if not all_matches:
            return None

        last = max(all_matches, key=lambda m: m.end())
        return last.group(1).strip()


    def solve(self, problem: str) -> UniversalResponse:
        return self.agent.solve(problem, self.prompts)
    
    def reset(self):
        self.agent.reset()
    
    def get_stats(self):
        return self.agent.get_stats()
