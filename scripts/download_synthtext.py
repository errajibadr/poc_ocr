#!/usr/bin/env python3
"""
Download and process the SynthText dataset for OCR testing.

This script downloads the SynthText dataset, extracts it,
and converts it to the OCR POC format.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ocr_poc.data.dataset import DatasetCollection
from src.ocr_poc.data.dataset_research import DatasetRegistry
from src.ocr_poc.data.dataset_utils import (
    DataAugmentation,
    DatasetDownloader,
    FormatConverter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_synthtext")


def download_synthtext(output_dir: Path, sample_only: bool = False) -> Path:
    """
    Download the SynthText dataset.

    Args:
        output_dir: Directory to download the dataset to.
        sample_only: Whether to download only a small sample of the dataset.

    Returns:
        Path to the downloaded dataset.
    """
    logger.info(f"Downloading SynthText dataset to {output_dir}")

    # Create downloader
    downloader = DatasetDownloader(output_dir)

    # Get dataset info from registry
    dataset_info = DatasetRegistry.get_dataset_info("SynthText")
    if not dataset_info:
        logger.error("SynthText dataset not found in registry")
        raise ValueError("SynthText dataset not found in registry")

    # Configure download
    if sample_only:
        # For a sample, use a smaller version if available
        logger.info("Downloading sample only")
        download_config = {
            "name": "SynthText_sample",
            "url": "https://github.com/ankush-me/SynthText/raw/master/samples.tgz",
            "format": "tgz",
        }
    else:
        # Full dataset
        logger.info("Downloading full dataset")
        download_config = {"name": "SynthText", "url": dataset_info["url"], "format": "zip"}

    # Download the dataset
    try:
        dataset_path = downloader.download_dataset(download_config)
        logger.info(f"Downloaded dataset to {dataset_path}")
        return dataset_path
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def process_synthtext(dataset_path: Path, output_collection: DatasetCollection) -> None:
    """
    Process the SynthText dataset and add it to the collection.

    Args:
        dataset_path: Path to the downloaded dataset.
        output_collection: DatasetCollection to add the dataset to.
    """
    logger.info(f"Processing SynthText dataset from {dataset_path}")

    converter = FormatConverter()

    # Convert to OCRDataset format
    dataset = converter.convert_to_ocr_dataset(
        dataset_path, "SynthText", "synthtext", output_collection
    )

    logger.info(f"Processed {len(dataset)} samples")


def create_dataset_variations(
    dataset_collection: DatasetCollection, dataset_name: str, num_variations: int = 2
) -> None:
    """
    Create variations of dataset images.

    Args:
        dataset_collection: DatasetCollection containing the dataset.
        dataset_name: Name of the dataset to augment.
        num_variations: Number of variations to create for each image.
    """
    logger.info(f"Creating {num_variations} variations for each image in {dataset_name} dataset")

    # Load dataset
    dataset = dataset_collection.get_dataset(dataset_name)

    # Create augmenter
    augmenter = DataAugmentation()

    # Create variations for each sample
    for sample in dataset:
        try:
            sample_id = sample["id"]
            # Load the original image
            image = dataset.load_image(sample_id)

            # Create variations
            variations = augmenter.create_variations(image, num_variations)

            # Add variations to dataset
            for i, var_image in enumerate(variations):
                # Save variation to temporary file
                temp_filename = f"variation_{sample_id}_{i}.png"
                temp_path = dataset.base_dir / temp_filename
                var_image.save(temp_path)

                # Add to dataset
                dataset.add_sample(
                    temp_path,
                    dataset.get_ground_truth(sample_id),
                    f"variation_{i}",
                    metadata={
                        "original_sample_id": sample_id,
                        "variation_type": "synthetic",
                        "variation_index": i,
                    },
                )

                # Remove temporary file
                os.remove(temp_path)

            logger.info(f"Created {num_variations} variations for sample {sample_id}")

        except Exception as e:
            logger.error(f"Error creating variations for sample {sample_id}: {e}")

    logger.info(f"Created variations for {len(dataset)} samples")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download and process the SynthText dataset for OCR testing"
    )

    parser.add_argument(
        "--output-dir", type=str, default="data/raw", help="Directory to download the dataset to"
    )

    parser.add_argument(
        "--sample-only", action="store_true", help="Download only a small sample of the dataset"
    )

    parser.add_argument(
        "--create-variations", action="store_true", help="Create variations of the dataset images"
    )

    parser.add_argument(
        "--num-variations",
        type=int,
        default=2,
        help="Number of variations to create for each image",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset collection
    dataset_collection = DatasetCollection(output_dir)

    try:
        # Download dataset
        dataset_path = download_synthtext(output_dir, args.sample_only)

        # Process dataset
        process_synthtext(dataset_path, dataset_collection)

        # Create variations if requested
        if args.create_variations:
            create_dataset_variations(dataset_collection, "SynthText", args.num_variations)

        logger.info("Dataset download and processing complete")

    except Exception as e:
        logger.error(f"Error downloading and processing dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
