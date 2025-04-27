"""
Metrics for OCR evaluation.

This module provides functions for evaluating OCR results against ground truth
using various metrics like character accuracy, word accuracy, etc.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..engine.base import OCRResult
from ..utils.logging import logger


def normalize_text(text: str, keep_spaces: bool = True, keep_case: bool = False) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Text to normalize.
        keep_spaces: Whether to keep spaces or remove them.
        keep_case: Whether to preserve case or convert to lowercase.

    Returns:
        Normalized text.
    """
    # Remove non-alphanumeric characters except spaces if keep_spaces is True
    if keep_spaces:
        text = re.sub(r"[^\w\s]", "", text)
    else:
        text = re.sub(r"[^\w]", "", text)

    # Convert to lowercase if keep_case is False
    if not keep_case:
        text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def character_accuracy(
    result: Union[str, OCRResult],
    ground_truth: str,
    normalize: bool = True,
    keep_case: bool = False,
) -> float:
    """
    Calculate character accuracy (character-level correctness).

    Args:
        result: OCR result (string or OCRResult object).
        ground_truth: Ground truth text.
        normalize: Whether to normalize texts before comparison.
        keep_case: Whether to consider case in accuracy calculation.

    Returns:
        Character accuracy as a float between 0 and 1.
    """
    # Get text from OCRResult if needed
    pred_text = result.text if isinstance(result, OCRResult) else result

    # Normalize if requested
    if normalize:
        pred_text = normalize_text(pred_text, keep_case=keep_case)
        ground_truth = normalize_text(ground_truth, keep_case=keep_case)

    # Empty ground truth or prediction
    if not ground_truth:
        logger.warning("Empty ground truth for character accuracy calculation")
        return 1.0 if not pred_text else 0.0

    if not pred_text:
        logger.warning("Empty prediction for character accuracy calculation")
        return 0.0

    # Levenshtein distance calculation
    m, n = len(pred_text), len(ground_truth)

    # Create distance matrix
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_text[i - 1] == ground_truth[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Deletion
                    dp[i][j - 1],  # Insertion
                    dp[i - 1][j - 1],
                )  # Substitution

    # Calculate accuracy
    edit_distance = dp[m][n]
    max_length = max(m, n)
    accuracy = 1.0 - (edit_distance / max_length) if max_length > 0 else 1.0

    return accuracy


def word_accuracy(
    result: Union[str, OCRResult],
    ground_truth: str,
    normalize: bool = True,
    keep_case: bool = False,
) -> float:
    """
    Calculate word accuracy (word-level correctness).

    Args:
        result: OCR result (string or OCRResult object).
        ground_truth: Ground truth text.
        normalize: Whether to normalize texts before comparison.
        keep_case: Whether to consider case in accuracy calculation.

    Returns:
        Word accuracy as a float between 0 and 1.
    """
    # Get text from OCRResult if needed
    pred_text = result.text if isinstance(result, OCRResult) else result

    # Normalize if requested
    if normalize:
        pred_text = normalize_text(pred_text, keep_case=keep_case)
        ground_truth = normalize_text(ground_truth, keep_case=keep_case)

    # Split into words
    pred_words = pred_text.split()
    gt_words = ground_truth.split()

    # Empty ground truth or prediction
    if not gt_words:
        logger.warning("Empty ground truth for word accuracy calculation")
        return 1.0 if not pred_words else 0.0

    if not pred_words:
        logger.warning("Empty prediction for word accuracy calculation")
        return 0.0

    # Count correct words
    correct = 0
    for i, gt_word in enumerate(gt_words):
        if i < len(pred_words) and pred_words[i] == gt_word:
            correct += 1

    # Calculate accuracy
    accuracy = correct / max(len(gt_words), len(pred_words))

    return accuracy


def word_error_rate(
    result: Union[str, OCRResult],
    ground_truth: str,
    normalize: bool = True,
    keep_case: bool = False,
) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (Substitutions + Insertions + Deletions) / Number of Words in Ground Truth

    Args:
        result: OCR result (string or OCRResult object).
        ground_truth: Ground truth text.
        normalize: Whether to normalize texts before comparison.
        keep_case: Whether to consider case in error calculation.

    Returns:
        Word Error Rate as a float. Lower is better.
    """
    # Get text from OCRResult if needed
    pred_text = result.text if isinstance(result, OCRResult) else result

    # Normalize if requested
    if normalize:
        pred_text = normalize_text(pred_text, keep_case=keep_case)
        ground_truth = normalize_text(ground_truth, keep_case=keep_case)

    # Split into words
    pred_words = pred_text.split()
    gt_words = ground_truth.split()

    # Empty ground truth
    if not gt_words:
        logger.warning("Empty ground truth for WER calculation")
        return float("inf") if pred_words else 0.0

    # Calculate Levenshtein distance at word level
    m, n = len(pred_words), len(gt_words)

    # Create distance matrix
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == gt_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # Deletion
                    dp[i][j - 1],  # Insertion
                    dp[i - 1][j - 1],
                )  # Substitution

    # Calculate WER
    word_distance = dp[m][n]
    wer = word_distance / n if n > 0 else float("inf")

    return wer


# Dictionary of available metrics
METRICS: Dict[str, Callable] = {
    "character_accuracy": character_accuracy,
    "word_accuracy": word_accuracy,
    "wer": word_error_rate,
}


def evaluate_result(
    result: Union[str, OCRResult],
    ground_truth: str,
    metrics: Optional[List[str]] = None,
    normalize: bool = True,
    keep_case: bool = False,
) -> Dict[str, float]:
    """
    Evaluate OCR result against ground truth using multiple metrics.

    Args:
        result: OCR result (string or OCRResult object).
        ground_truth: Ground truth text.
        metrics: List of metric names to calculate. If None, calculate all metrics.
        normalize: Whether to normalize texts before comparison.
        keep_case: Whether to consider case in evaluation.

    Returns:
        Dictionary with metric names as keys and scores as values.
    """
    # Default to all metrics if none specified
    metrics_to_use = metrics or list(METRICS.keys())

    # Calculate each metric
    results = {}
    for metric_name in metrics_to_use:
        if metric_name in METRICS:
            try:
                results[metric_name] = METRICS[metric_name](
                    result, ground_truth, normalize, keep_case
                )
            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {e}")
                results[metric_name] = float("nan")
        else:
            logger.warning(f"Unknown metric: {metric_name}")

    return results
