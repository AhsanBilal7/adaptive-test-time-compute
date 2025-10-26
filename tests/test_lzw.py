# tests/test_lzw.py
from src.lzw import lzw_ratio


def test_lzw_empty():
    assert lzw_ratio("") == 1.0


def test_lzw_repetition():
    s = "abcabcabcabcabcabc"
    r = lzw_ratio(s)
    assert r < 0.6


def test_lzw_randomish():
    s = "qzjxv wpkm rtyu ioas dlkj qwep zmxn asdl k"
    r = lzw_ratio(s)
    assert r > 0.8