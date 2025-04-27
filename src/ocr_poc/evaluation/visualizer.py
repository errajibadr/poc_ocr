"""
Visualization utilities for OCR results.

This module provides functions for visualizing OCR results, including
text overlays on images and comparison visualizations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config.settings import settings
from ..engine.base import OCRResult
from ..utils.logging import logger


def draw_text_boxes(
    image: Union[str, Path, Image.Image, np.ndarray],
    ocr_result: OCRResult,
    output_path: Optional[Union[str, Path]] = None,
    color: Tuple[int, int, int] = (255, 0, 0),  # Red
    thickness: int = 2,
    show_confidence: bool = True,
    font_path: Optional[str] = None,
    font_size: int = 12,
) -> Image.Image:
    """
    Draw bounding boxes and recognized text on an image.

    Args:
        image: Input image.
        ocr_result: OCR result object.
        output_path: Path to save the output image. If None, the image is not saved.
        color: RGB color for bounding boxes and text.
        thickness: Thickness of bounding box lines.
        show_confidence: Whether to show confidence scores.
        font_path: Path to font file. If None, default font is used.
        font_size: Font size for text.

    Returns:
        PIL Image with bounding boxes and text drawn.
    """
    # Load image if needed
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Create a copy to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Try to load font, use default if not available
    font = None
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            logger.warning(f"Failed to load font: {e}")

    if font is None:
        # Use default font
        try:
            font = ImageFont.load_default()
        except Exception as e:
            logger.warning(f"Failed to load default font: {e}")
            font = None

    # Draw bounding boxes and text
    if ocr_result.bounding_boxes:
        for i, box in enumerate(ocr_result.bounding_boxes):
            # Draw bounding box
            if len(box) == 4:  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = box
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)

                # Get text for this box (if available)
                text = (
                    ocr_result.text.split("\n")[i] if i < len(ocr_result.text.split("\n")) else ""
                )

                # Add confidence if available and requested
                if show_confidence and hasattr(ocr_result, "confidence"):
                    text = f"{text} ({ocr_result.confidence:.2f})"

                # Draw text above the box
                if font:
                    draw.text((x1, y1 - font_size - 5), text, fill=color, font=font)
                else:
                    draw.text((x1, y1 - font_size - 5), text, fill=color)
    else:
        # If no bounding boxes, draw text at the top
        if font:
            draw.text((10, 10), ocr_result.text, fill=color, font=font)
        else:
            draw.text((10, 10), ocr_result.text, fill=color)

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        draw_image.save(output_path)
        logger.debug(f"Saved annotated image to {output_path}")

    return draw_image


def create_comparison_visualization(
    image: Union[str, Path, Image.Image, np.ndarray],
    ocr_results: List[OCRResult],
    ground_truth: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 100,
) -> None:
    """
    Create a comparison visualization of multiple OCR results.

    Args:
        image: Input image.
        ocr_results: List of OCR result objects from different engines.
        ground_truth: Ground truth text for comparison.
        output_path: Path to save the visualization. If None, the visualization is displayed.
        figsize: Figure size (width, height) in inches.
        dpi: DPI for the figure.
    """
    # Load image if needed
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Determine number of subplots
    n_results = len(ocr_results)
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols  # Ceiling division

    # Add one row for the original image and ground truth
    n_rows += 1

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # First subplot: original image
    axes[0].imshow(np.array(image))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Second subplot: ground truth (if provided)
    if ground_truth:
        axes[1].text(
            0.1, 0.5, f"Ground Truth:\n{ground_truth}", fontsize=10, verticalalignment="center"
        )
        axes[1].axis("off")

    # Plot OCR results
    start_idx = n_cols  # Start after the first row
    for i, result in enumerate(ocr_results):
        idx = start_idx + i
        if idx < len(axes):
            # Draw text boxes on image
            annotated_img = draw_text_boxes(image, result)

            # Display annotated image
            axes[idx].imshow(np.array(annotated_img))
            axes[idx].set_title(f"{result.engine_name} ({result.confidence:.2f})")
            axes[idx].axis("off")

    # Hide any unused subplots
    for i in range(start_idx + n_results, len(axes)):
        axes[i].axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        logger.debug(f"Saved comparison visualization to {output_path}")
        plt.close(fig)
    else:
        plt.show()


def create_metrics_visualization(
    results: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    title: str = "OCR Performance Comparison",
) -> None:
    """
    Create a bar chart visualization comparing metrics across OCR engines.

    Args:
        results: Dictionary with engine names as keys and dictionaries of metrics as values.
        output_path: Path to save the visualization. If None, the visualization is displayed.
        figsize: Figure size (width, height) in inches.
        dpi: DPI for the figure.
        title: Title for the chart.
    """
    # Extract engine names and metrics
    engines = list(results.keys())
    if not engines:
        logger.warning("No results to visualize")
        return

    metrics = list(results[engines[0]].keys())
    if not metrics:
        logger.warning("No metrics to visualize")
        return

    # Set up figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, dpi=dpi, sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    # Create a bar chart for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Extract values for this metric
        values = [results[engine].get(metric, 0) for engine in engines]

        # Create bars
        bars = ax.bar(engines, values, color="skyblue")

        # Add values on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Set title and labels
        ax.set_title(f"{metric}")
        ax.set_ylim(0, 1.1 if metric != "wer" else max(values) * 1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path)
        logger.debug(f"Saved metrics visualization to {output_path}")
        plt.close(fig)
    else:
        plt.show()
