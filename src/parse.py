# src/parse.py
import re


def split_reason_answer(text: str) -> tuple[str, str]:
    text = text.strip()
    
    ans_match = re.search(r"Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if ans_match:
        answer = ans_match.group(1).strip()
        reasoning = text[:ans_match.start()].strip()
        reasoning = re.sub(r"^Reasoning:\s*", "", reasoning, flags=re.IGNORECASE).strip()
        return reasoning, answer
    
    return text, ""


def parse_gsm8k_numeric(text: str) -> str:
    text = text.strip()
    
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    
    if numbers:
        last_num = numbers[-1].replace(",", "")
        return last_num
    
    return text


def parse_yesno(text: str) -> str:
    text = text.lower().strip()
    
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    
    return text