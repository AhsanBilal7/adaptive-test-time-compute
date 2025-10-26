# tests/test_selection.py
from src.selection import pick_sweet_spot


def test_pick_window_middle():
    chains = ["A" * 200, "logical step 1\nstep 2\ntherefore", "noise: zqjx ..."]
    idx, ratio = pick_sweet_spot(chains, low=0.3, high=0.6, prefer_center=True)
    assert isinstance(idx, int)
    assert isinstance(ratio, float)