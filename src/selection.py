# src/selection.py
from typing import List, Tuple
from src.lzw import lzw_ratio


def score_chains(chains: List[str]) -> List[float]:
    return [lzw_ratio(ch) for ch in chains]


def pick_sweet_spot(chains: List[str], low: float, high: float, prefer_center=True) -> Tuple[int, float]:
    ratios = score_chains(chains)
    # print("Ratios:", ratios)
    center = (low + high) / 2.0
    in_win = [(i, r) for i, r in enumerate(ratios) if low <= r <= high]
    
    if in_win:
        if prefer_center:
            i_best, r_best = min(in_win, key=lambda t: abs(t[1] - center))
        else:
            in_win_sorted = sorted(in_win, key=lambda t: t[1])
            i_best, r_best = in_win_sorted[len(in_win_sorted) // 2]
        return i_best, r_best
    
    def dist(r):
        if r < low:
            return low - r
        if r > high:
            return r - high
        return 0.0
    
    i_best, r_best = min(enumerate(ratios), key=lambda t: dist(t[1]))
    return i_best, r_best


def pick_lowest_compression(chains: List[str]) -> Tuple[int, float]:
    """Select the chain with the LOWEST LZW compression ratio (most compressible)."""
    ratios = score_chains(chains)
    i_best = min(enumerate(ratios), key=lambda t: t[1])[0]
    return i_best, ratios[i_best]


def pick_highest_compression(chains: List[str]) -> Tuple[int, float]:
    """Select the chain with the HIGHEST LZW compression ratio (least compressible)."""
    ratios = score_chains(chains)
    i_best = max(enumerate(ratios), key=lambda t: t[1])[0]
    return i_best, ratios[i_best]


def pick_median_compression(chains: List[str]) -> Tuple[int, float]:
    """Select the chain with the MEDIAN LZW compression ratio."""
    ratios = score_chains(chains)
    sorted_with_idx = sorted(enumerate(ratios), key=lambda t: t[1])
    median_idx = len(sorted_with_idx) // 2
    i_best, r_best = sorted_with_idx[median_idx]
    return i_best, r_best


def self_consistency_majority(answers: List[str]) -> str:
    from collections import Counter
    c = Counter(a for a in answers if a is not None)
    if not c:
        return ""
    return c.most_common(1)[0][0]


def iterative_reasoning_selection(
    client,
    question: str,
    kind: str,
    K: int,
    low: float,
    high: float,
    prefer_center: bool,
    max_steps: int
) -> Tuple[str, List[dict]]:
    """
    Iteratively build reasoning by generating K reasoning steps at each iteration,
    selecting the one with the best LZW compression ratio in the sweet spot,
    and continuing until we detect a final answer or reach max_steps.
    
    After building reasoning, makes a final call to extract clean answer without comments.
    
    Returns:
        final_reasoning: The complete reasoning chain
        step_trace: List of dicts with info about each step selection
    """
    from src.templates import build_iterative_prompt, build_final_answer_prompt
    from src.generation import generate_k_paths, generate_greedy
    from src.parse import has_answer_marker
    
    accumulated_reasoning = ""
    step_trace = []
    
    # Iteratively build reasoning
    for step_num in range(max_steps):
        # Generate K candidate next steps
        prompt = build_iterative_prompt(question, accumulated_reasoning, kind=kind)
        candidates = generate_k_paths(client, prompt, K=K)
        
        # Check if any candidate contains the answer marker
        contains_answer = [has_answer_marker(c, kind=kind) for c in candidates]
        
        # Score candidates with LZW compression
        idx, ratio = pick_sweet_spot(candidates, low, high, prefer_center)
        selected = candidates[idx]
        
        # Record this step
        step_trace.append({
            "step": step_num,
            "selected_text": selected,
            "ratio": ratio,
            "all_ratios": score_chains(candidates),
            "has_answer": contains_answer[idx],
        })
        
        # Append selected reasoning
        accumulated_reasoning = accumulated_reasoning + " " + selected if accumulated_reasoning else selected
        
        # Check if we've reached an answer
        if contains_answer[idx]:
            print(f"Found answer at step {step_num}")
            break
    
    # Final step: Get clean answer without comments
    # This ensures the final output is just "Answer: <value>" with no extra text
    # print("Extracting final answer...")
    final_prompt = build_final_answer_prompt(question, accumulated_reasoning, kind=kind)
    final_answer_text = generate_greedy(client, final_prompt)
    
    print(f"Final answer extraction output: {final_answer_text}")
    # Append the clean final answer to reasoning
    final_reasoning = accumulated_reasoning + "\n\n" + final_answer_text.strip()
    
    # Record final answer extraction step
    step_trace.append({
        "step": len(step_trace),
        "selected_text": final_answer_text.strip(),
        "ratio": lzw_ratio(final_answer_text),
        "all_ratios": [lzw_ratio(final_answer_text)],
        "has_answer": True,
        "is_final_extraction": True,
    })
    
    return final_reasoning.strip(), step_trace