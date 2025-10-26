# src/lzw.py
from __future__ import annotations

def lzw_compress_bytes(data: bytes) -> list[int]:
    """Classic LZW over bytes. Returns a list of integer codes."""
    if not data:
        return []

    # Initialize dictionary with all single-byte entries
    dictionary = {bytes([i]): i for i in range(256)}
    dict_size = 256

    result: list[int] = []
    w = b""

    for b in data:
        c = bytes([b])
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            # w must be in dictionary
            result.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    if w:
        result.append(dictionary[w])

    return result


def lzw_compress(text: str) -> list[int]:
    """Wrapper: UTF-8 encode then run byte-level LZW."""
    return lzw_compress_bytes(text.encode("utf-8"))


def lzw_ratio(text: str) -> float:
    """
    Compression ratio = compressed_size / original_size.
    We approximate each code as 2 bytes (16 bits), which is a common cap for LZW.
    For a more precise estimate, compute dynamic code widths (9..16 bits).
    """
    if not text:
        return 1.0

    compressed_codes = lzw_compress(text)
    original_bytes = len(text.encode("utf-8"))

    # Simple upper-bound approximation: 2 bytes per code
    compressed_bytes = len(compressed_codes) * 2

    return compressed_bytes / max(1, original_bytes)


# (Optional) precise bit counting if you want better ratios:
def lzw_ratio_precise(text: str) -> float:
    """
    Compute ratio with dynamic code widths growing from 9 to 16 bits
    as the dictionary expands beyond {0..255}. This is closer to real LZW.
    """
    data = text.encode("utf-8")
    if not data:
        return 1.0

    dictionary = {bytes([i]): i for i in range(256)}
    dict_size = 256
    w = b""
    compressed_bits = 0

    def code_width(n: int) -> int:
        # 0..255 -> 9 bits (since 256..511 needs 9), grow as dict grows
        # You may choose to start at 8; many LZW variants start at 9.
        if n < 512: return 9
        if n < 1024: return 10
        if n < 2048: return 11
        if n < 4096: return 12
        if n < 8192: return 13
        if n < 16384: return 14
        if n < 32768: return 15
        return 16  # cap

    for b in data:
        c = bytes([b])
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            # emit code for w
            compressed_bits += code_width(dict_size)
            dictionary[wc] = dict_size
            dict_size += 1
            w = c

    if w:
        compressed_bits += code_width(dict_size)

    original_bytes = len(data)
    compressed_bytes = (compressed_bits + 7) // 8
    return compressed_bytes / max(1, original_bytes)
