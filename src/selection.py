# src/selection.py
from typing import List, Tuple
from src.lzw import lzw_ratio


def score_chains(chains: List[str]) -> List[float]:
    return [lzw_ratio(ch) for ch in chains]


def pick_sweet_spot(chains: List[str], low: float, high: float, prefer_center=True) -> Tuple[int, float]:
    ratios = score_chains(chains)
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


def self_consistency_majority(answers: List[str]) -> str:
    from collections import Counter
    c = Counter(a for a in answers if a is not None)
    if not c:
        return ""
    return c.most_common(1)[0][0]