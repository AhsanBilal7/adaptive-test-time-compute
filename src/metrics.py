# src/metrics.py
from typing import List, Union


def accuracy(predictions: List[str], gold: List[str]) -> float:
    """
    Calculate accuracy as the fraction of predictions that match the gold labels.
    
    Args:
        predictions: List of predicted answers
        gold: List of gold/ground-truth answers
        
    Returns:
        Accuracy as a float between 0.0 and 1.0
    """
    if not predictions or not gold:
        print("Warning: Empty predictions or gold list")
        return 0.0
    
    if len(predictions) != len(gold):
        print(f"Warning: Length mismatch - predictions: {len(predictions)}, gold: {len(gold)}")
        return 0.0
    
    # Handle both string and numeric comparisons
    correct = 0
    for p, g in zip(predictions, gold):
        # Normalize both to strings and strip whitespace
        p_str = str(p).strip().lower()
        g_str = str(g).strip().lower()
        
        if p_str == g_str:
            correct += 1
        # Also check numeric equality for math problems
        else:
            try:
                p_num = float(p_str.replace(",", ""))
                g_num = float(g_str.replace(",", ""))
                if abs(p_num - g_num) < 1e-6:  # Handle floating point precision
                    correct += 1
            except (ValueError, AttributeError):
                # Not numeric, already checked string equality
                pass
    
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def exact_match(predictions: List[str], gold: List[str]) -> float:
    """
    Calculate exact string match accuracy (case-sensitive).
    """
    if not predictions or not gold or len(predictions) != len(gold):
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, gold) if str(p).strip() == str(g).strip())
    return correct / len(predictions)


def print_accuracy_report(predictions: List[str], gold: List[str], name: str = "Model"):
    """
    Print a detailed accuracy report with examples of correct and incorrect predictions.
    """
    acc = accuracy(predictions, gold)
    print(f"\n{'='*60}")
    print(f"{name} Accuracy Report")
    print(f"{'='*60}")
    print(f"Total examples: {len(predictions)}")
    print(f"Accuracy: {acc:.4f} ({int(acc * len(predictions))}/{len(predictions)} correct)")
    
    # Show some examples
    correct_examples = []
    incorrect_examples = []
    
    for i, (p, g) in enumerate(zip(predictions, gold)):
        p_str = str(p).strip().lower()
        g_str = str(g).strip().lower()
        
        if p_str == g_str:
            correct_examples.append((i, p, g))
        else:
            incorrect_examples.append((i, p, g))
    
    if correct_examples:
        print(f"\nSample correct predictions (showing up to 3):")
        for i, p, g in correct_examples[:3]:
            print(f"  Example {i}: Predicted '{p}' == Gold '{g}' ✓")
    
    if incorrect_examples:
        print(f"\nSample incorrect predictions (showing up to 5):")
        for i, p, g in incorrect_examples[:5]:
            print(f"  Example {i}: Predicted '{p}' != Gold '{g}' ✗")
    
    print(f"{'='*60}\n")