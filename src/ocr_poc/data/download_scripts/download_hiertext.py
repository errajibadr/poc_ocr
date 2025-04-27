"""
HierText Dataset Download Script.

This script downloads the HierText dataset and processes it for OCR evaluation.
HierText is a hierarchical text detection dataset with word, line, and paragraph annotations.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ocr_poc.config.settings import settings
from ocr_poc.data.dataset import DatasetCollection, OCRDataset
from ocr_poc.data.dataset_utils import DatasetDownloader, FormatConverter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr_poc.data.download_scripts.download_hiertext")


class HierTextDownloader:
    """Downloader for the HierText dataset."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the HierText downloader.

        Args:
            output_dir: Directory to save the dataset to. If None, use default from settings.
        """
        self.output_dir = output_dir or settings.dataset.raw_data_dir / "HierText"
        self.downloader = DatasetDownloader(self.output_dir)
        self.converter = FormatConverter()
        self.dataset_collection = DatasetCollection(self.output_dir)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Initialized HierText downloader with output directory: {self.output_dir}")

    def download(self) -> Path:
        """
        Download the HierText dataset.

        Returns:
            Path to the downloaded dataset.
        """
        logger.info("Downloading HierText dataset...")

        # Dataset config
        dataset_config = {
            "name": "HierText",
            "format": "directory",
            "files": [
                {
                    "url": "https://github.com/google-research-datasets/hiertext/raw/main/gt/train.jsonl",
                    "path": "train.jsonl",
                },
                {
                    "url": "https://github.com/google-research-datasets/hiertext/raw/main/gt/validation.jsonl",
                    "path": "validation.jsonl",
                },
            ],
        }

        # For the image data, we need to download from Google Cloud Storage
        # This is a large dataset (>10GB), so we'll only download annotations in this script
        # and provide instructions for downloading the image data separately

        try:
            # Download the dataset annotations
            dataset_path = self.downloader.download_dataset(dataset_config)
            logger.info(f"Downloaded HierText annotations to {dataset_path}")

            # Note that we're only downloading annotations here
            logger.warning(
                "Only annotations were downloaded. Image data must be downloaded separately."
            )
            logger.warning(
                "See https://github.com/google-research-datasets/hiertext for instructions on downloading images."
            )

            return dataset_path
        except Exception as e:
            logger.error(f"Error downloading HierText dataset: {e}")
            raise

    def process(
        self,
        dataset_path: Path,
        image_dir: Optional[Path] = None,
        split: str = "validation",
        max_samples: int = 1000,
    ) -> OCRDataset:
        """
        Process the HierText dataset for OCR evaluation.

        Args:
            dataset_path: Path to the downloaded annotations.
            image_dir: Directory containing the image files. If None, assume images are in dataset_path/images.
            split: Which split to process ("train" or "validation").
            max_samples: Maximum number of samples to process.

        Returns:
            OCRDataset with processed data.
        """
        logger.info(f"Processing HierText dataset from {dataset_path} (split: {split})")

        # Set image directory
        image_dir = image_dir or dataset_path / "images"

        # Check if the split file exists
        split_file = dataset_path / f"{split}.jsonl"
        if not split_file.exists():
            logger.error(f"HierText {split} file not found at {split_file}")
            raise FileNotFoundError(f"HierText {split} file not found at {split_file}")

        # Create dataset
        dataset = self.dataset_collection.create_dataset(f"HierText_{split}")

        try:
            # Check if image directory exists
            if not image_dir.exists():
                logger.warning(f"Image directory not found at {image_dir}")
                logger.warning(
                    "Will process annotations only. Images will need to be added separately."
                )

            # Process the JSONL file
            logger.info(f"Loading HierText annotations from {split_file}")

            samples_processed = 0
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    if samples_processed >= max_samples:
                        break

                    # Parse JSON line
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing JSON line: {e}")
                        continue

                    # Extract image information
                    img_id = data.get("image_id")
                    if not img_id:
                        logger.warning("Missing image_id in annotation")
                        continue

                    # Expected image path
                    img_path = image_dir / f"{img_id}.jpg"

                    # Process text annotations
                    all_text = []
                    all_boxes = []

                    # HierText has a hierarchical structure: paragraphs -> lines -> words
                    paragraphs = data.get("paragraphs", [])
                    for paragraph in paragraphs:
                        lines = paragraph.get("lines", [])
                        for line in lines:
                            words = line.get("words", [])
                            for word in words:
                                text = word.get("text", "")
                                if text:
                                    all_text.append(text)

                                    # Get bounding box (vertices)
                                    vertices = word.get("vertices", [])
                                    if len(vertices) == 4:
                                        # Convert to [x1, y1, x2, y2, x3, y3, x4, y4] format
                                        flat_box = []
                                        for vertex in vertices:
                                            flat_box.extend(
                                                [int(vertex.get("x", 0)), int(vertex.get("y", 0))]
                                            )
                                        all_boxes.append(flat_box)

                    # Combine all text
                    combined_text = " ".join(all_text)

                    # Attempt to get image dimensions
                    width = None
                    height = None
                    if img_path.exists():
                        try:
                            img = Image.open(img_path)
                            width, height = img.size
                        except Exception as e:
                            logger.warning(f"Error reading image {img_path}: {e}")

                    # Create metadata
                    metadata = {
                        "image_id": img_id,
                        "bounding_boxes": all_boxes,
                        "width": width,
                        "height": height,
                        "paragraph_count": len(paragraphs),
                    }

                    # Add sample to dataset
                    try:
                        if img_path.exists():
                            # If image exists, add it to the dataset
                            dataset.add_sample(
                                image_path=img_path,
                                ground_truth=combined_text,
                                category="scene_text",
                                metadata=metadata,
                            )
                        else:
                            # If image doesn't exist, still add the annotation
                            # with a reference to where the image should be
                            metadata["missing_image"] = True
                            dataset.add_sample(
                                image_path=str(
                                    img_path
                                ),  # Store as string since file doesn't exist
                                ground_truth=combined_text,
                                category="scene_text",
                                metadata=metadata,
                            )
                            logger.warning(f"Image not found: {img_path}")
                    except Exception as e:
                        logger.warning(f"Error adding sample for image {img_id}: {e}")
                        continue

                    samples_processed += 1
                    if samples_processed % 100 == 0:
                        logger.info(f"Processed {samples_processed}/{max_samples} samples")

            logger.info(f"Processed {samples_processed} samples from HierText {split} split")
            dataset.save_metadata()
            return dataset

        except Exception as e:
            logger.error(f"Error processing HierText dataset: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and process HierText dataset")

    parser.add_argument("--output-dir", type=str, help="Directory to save the dataset to")

    parser.add_argument(
        "--image-dir", type=str, help="Directory containing the image files (if already downloaded)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation"],
        help="Which split to process",
    )

    parser.add_argument(
        "--max-samples", type=int, default=1000, help="Maximum number of samples to process"
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    image_dir = Path(args.image_dir) if args.image_dir else None
    split = args.split
    max_samples = args.max_samples

    downloader = HierTextDownloader(output_dir)

    try:
        # Download the dataset
        dataset_path = downloader.download()

        # Process the dataset
        dataset = downloader.process(
            dataset_path=dataset_path, image_dir=image_dir, split=split, max_samples=max_samples
        )

        logger.info(
            f"Successfully processed HierText dataset ({split} split) with {len(dataset)} samples"
        )
        logger.info(f"Dataset saved to {dataset.processed_dir}")

        # Print instructions for image download if no images were found
        if image_dir is None or not image_dir.exists():
            logger.warning("\nTo download the image data:")
            logger.warning(
                "1. Install gsutil: https://cloud.google.com/storage/docs/gsutil_install"
            )
            logger.warning(
                "2. Run: gsutil -m cp -r gs://gresearch/hiertext/imgs/<SPLIT> <image_dir>"
            )
            logger.warning(
                "   where <SPLIT> is 'train' or 'validation' and <image_dir> is your image directory"
            )
            logger.warning(
                "3. Run this script again with --image-dir=<image_dir> to process with images"
            )

    except Exception as e:
        logger.error(f"Error downloading/processing HierText dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
