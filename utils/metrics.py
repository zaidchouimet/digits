"""
Metrics utilities for digit recognition benchmarking.
Implements Character Error Rate (CER), Word Error Rate (WER), and accuracy calculations.
"""

import numpy as np
from typing import List, Tuple, Optional
import jiwer
import difflib


def _normalize_digit_text(text: str) -> str:
    """Normalize text for digit-sequence benchmarking."""
    digits = "".join(char for char in text if char.isdigit())
    return digits if digits else text.strip()


def calculate_cer(ground_truth: str, prediction: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        ground_truth: The correct text
        prediction: The predicted text
        
    Returns:
        CER as a float between 0 and 1
    """
    ground_truth = _normalize_digit_text(ground_truth)
    prediction = _normalize_digit_text(prediction)

    if not ground_truth:
        return 1.0 if prediction else 0.0
    
    if not prediction:
        return 1.0
    
    # Use jiwer for robust CER calculation
    try:
        cer = jiwer.cer(ground_truth, prediction)
        return cer
    except Exception:
        # Fallback to manual calculation
        gt_chars = list(ground_truth)
        pred_chars = list(prediction)
        
        # Simple Levenshtein distance
        dp = [[0] * (len(pred_chars) + 1) for _ in range(len(gt_chars) + 1)]
        
        for i in range(len(gt_chars) + 1):
            dp[i][0] = i
        for j in range(len(pred_chars) + 1):
            dp[0][j] = j
            
        for i in range(1, len(gt_chars) + 1):
            for j in range(1, len(pred_chars) + 1):
                if gt_chars[i-1] == pred_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1,    # insertion
                        dp[i-1][j-1] + 1   # substitution
                    )
        
        return dp[len(gt_chars)][len(pred_chars)] / len(gt_chars)


def calculate_wer(ground_truth: str, prediction: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Args:
        ground_truth: The correct text
        prediction: The predicted text
        
    Returns:
        WER as a float between 0 and 1
    """
    ground_truth = _normalize_digit_text(ground_truth)
    prediction = _normalize_digit_text(prediction)

    if not ground_truth:
        return 1.0 if prediction else 0.0
    
    if not prediction:
        return 1.0
    
    try:
        wer = jiwer.wer(" ".join(ground_truth), " ".join(prediction))
        return wer
    except Exception:
        # Fallback: treat each character as a "word"
        return calculate_cer(ground_truth, prediction)


def calculate_digit_accuracy(ground_truth: str, prediction: str) -> float:
    """
    Calculate digit-only accuracy.
    
    Args:
        ground_truth: The correct text
        prediction: The predicted text
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Extract only digits
    gt_digits = ''.join([c for c in ground_truth if c.isdigit()])
    pred_digits = ''.join([c for c in prediction if c.isdigit()])
    
    if not gt_digits:
        return 1.0 if not pred_digits else 0.0
    
    if not pred_digits:
        return 0.0
    
    matcher = difflib.SequenceMatcher(None, gt_digits, pred_digits)
    return matcher.ratio()


def calculate_sequence_accuracy(ground_truth: str, prediction: str) -> float:
    """
    Calculate exact sequence accuracy (entire string must match).
    
    Args:
        ground_truth: The correct text
        prediction: The predicted text
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    # Compare only digits
    gt_digits = ''.join([c for c in ground_truth if c.isdigit()])
    pred_digits = ''.join([c for c in prediction if c.isdigit()])
    
    return 1.0 if gt_digits == pred_digits else 0.0


def calculate_metrics_batch(ground_truths: List[str], predictions: List[str]) -> dict:
    """
    Calculate metrics for a batch of predictions.
    
    Args:
        ground_truths: List of ground truth strings
        predictions: List of prediction strings
        
    Returns:
        Dictionary with average metrics
    """
    if len(ground_truths) != len(predictions):
        raise ValueError("Ground truths and predictions must have same length")
    
    cer_scores = []
    wer_scores = []
    digit_acc_scores = []
    seq_acc_scores = []
    
    for gt, pred in zip(ground_truths, predictions):
        cer_scores.append(calculate_cer(gt, pred))
        wer_scores.append(calculate_wer(gt, pred))
        digit_acc_scores.append(calculate_digit_accuracy(gt, pred))
        seq_acc_scores.append(calculate_sequence_accuracy(gt, pred))
    
    return {
        'cer_mean': np.mean(cer_scores),
        'cer_std': np.std(cer_scores),
        'wer_mean': np.mean(wer_scores),
        'wer_std': np.std(wer_scores),
        'digit_accuracy_mean': np.mean(digit_acc_scores),
        'digit_accuracy_std': np.std(digit_acc_scores),
        'sequence_accuracy_mean': np.mean(seq_acc_scores),
        'sequence_accuracy_std': np.std(seq_acc_scores),
        'total_samples': len(ground_truths)
    }


def format_metrics_summary(metrics: dict) -> str:
    """
    Format metrics into a readable summary string.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics_batch
        
    Returns:
        Formatted summary string
    """
    return f"""
Metrics Summary:
- Character Error Rate (CER): {metrics['cer_mean']:.3f} +/- {metrics['cer_std']:.3f}
- Word Error Rate (WER): {metrics['wer_mean']:.3f} +/- {metrics['wer_std']:.3f}
- Digit Accuracy: {metrics['digit_accuracy_mean']:.3f} +/- {metrics['digit_accuracy_std']:.3f}
- Sequence Accuracy: {metrics['sequence_accuracy_mean']:.3f} +/- {metrics['sequence_accuracy_std']:.3f}
- Total Samples: {metrics['total_samples']}
"""
