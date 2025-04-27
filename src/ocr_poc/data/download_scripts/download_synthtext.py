"""
SynthText Dataset Download Script.

This script downloads the SynthText dataset and processes it for OCR evaluation.
SynthText contains synthetic text rendered onto natural images with ground truth.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
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
logger = logging.getLogger("ocr_poc.data.download_scripts.download_synthtext")


class SynthTextDownloader:
    """Downloader for the SynthText dataset."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the SynthText downloader.

        Args:
            output_dir: Directory to save the dataset to. If None, use default from settings.
        """
        self.output_dir = output_dir or settings.dataset.raw_data_dir / "SynthText"
        self.downloader = DatasetDownloader(self.output_dir)
        self.converter = FormatConverter()
        self.dataset_collection = DatasetCollection(self.output_dir)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Initialized SynthText downloader with output directory: {self.output_dir}")

    def download(self) -> Path:
        """
        Download the SynthText dataset.

        Returns:
            Path to the downloaded dataset.
        """
        logger.info("Downloading SynthText dataset...")

        # Dataset details from registry
        dataset_config = {
            "name": "SynthText",
            "url": "https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip",
            "format": "zip",
        }

        try:
            # Download the dataset
            dataset_path = self.downloader.download_dataset(dataset_config)
            logger.info(f"Downloaded SynthText dataset to {dataset_path}")
            return dataset_path
        except Exception as e:
            logger.error(f"Error downloading SynthText dataset: {e}")
            raise

    def process(self, dataset_path: Path, max_samples: int = 1000) -> OCRDataset:
        """
        Process the SynthText dataset for OCR evaluation.

        Args:
            dataset_path: Path to the downloaded dataset.
            max_samples: Maximum number of samples to process (SynthText is very large).

        Returns:
            OCRDataset with processed data.
        """
        logger.info(f"Processing SynthText dataset from {dataset_path}")

        # Import scipy here to avoid dependency for those who don't need it
        try:
            import scipy.io
        except ImportError:
            logger.error("scipy is required to process SynthText dataset but not installed")
            raise ImportError(
                "scipy is required to process SynthText dataset. Install it with 'pip install scipy'."
            )

        # Create dataset
        dataset = self.dataset_collection.create_dataset("SynthText")

        try:
            # The SynthText dataset contains a single .mat file with all annotations
            mat_file = dataset_path / "gt.mat"

            if not mat_file.exists():
                logger.error(f"SynthText annotation file not found at {mat_file}")
                raise FileNotFoundError(f"SynthText annotation file not found at {mat_file}")

            logger.info(f"Loading SynthText annotations from {mat_file}")

            # Load the .mat file
            mat_data = scipy.io.loadmat(str(mat_file))

            # Extract data
            img_paths = mat_data["imnames"][0]
            word_boxes = mat_data["wordBB"][0]  # Bounding boxes for each word
            word_texts = mat_data["txt"][0]  # Text for each word

            logger.info(f"Found {len(img_paths)} images in SynthText dataset")

            # Process up to max_samples
            for i, (img_path, boxes, texts) in enumerate(zip(img_paths, word_boxes, word_texts)):
                if i >= max_samples:
                    break

                if i % 100 == 0:
                    logger.info(f"Processing image {i}/{min(len(img_paths), max_samples)}")

                # Convert path to string
                img_path_str = str(img_path[0])

                # Full path to the image
                full_img_path = dataset_path / img_path_str

                if not full_img_path.exists():
                    logger.warning(f"Image not found: {full_img_path}")
                    continue

                # Load the image to get dimensions
                try:
                    img = Image.open(full_img_path)
                    width, height = img.size
                except Exception as e:
                    logger.warning(f"Error loading image {full_img_path}: {e}")
                    continue

                # Process texts
                if isinstance(texts, np.ndarray):
                    # Handle case where texts is an array of arrays
                    all_text = []
                    for text_arr in texts:
                        if isinstance(text_arr, np.ndarray):
                            for t in text_arr:
                                if isinstance(t, str):
                                    all_text.append(t)
                                else:
                                    # Handle case where text is not a string
                                    all_text.append(str(t))
                        else:
                            all_text.append(str(text_arr))

                    text = " ".join(all_text)
                else:
                    text = str(texts)

                # Process bounding boxes
                if isinstance(boxes, np.ndarray):
                    # SynthText uses 2x4xN format (2 points, 4 corners, N words)
                    if boxes.ndim == 3:
                        # Convert to format expected by our dataset
                        box_list = []
                        for n in range(boxes.shape[2]):
                            # Extract corners for this word
                            box = boxes[:, :, n]
                            # Convert to [x1, y1, x2, y2, x3, y3, x4, y4] format
                            flat_box = [
                                int(box[0, 0]),
                                int(box[1, 0]),  # Top-left
                                int(box[0, 1]),
                                int(box[1, 1]),  # Top-right
                                int(box[0, 2]),
                                int(box[1, 2]),  # Bottom-right
                                int(box[0, 3]),
                                int(box[1, 3]),  # Bottom-left
                            ]
                            box_list.append(flat_box)
                    else:
                        # Handle unexpected format
                        logger.warning(f"Unexpected bounding box format for {img_path_str}")
                        box_list = []
                else:
                    box_list = []

                # Add sample to dataset
                metadata = {
                    "original_path": img_path_str,
                    "bounding_boxes": box_list,
                    "width": width,
                    "height": height,
                    "synthetic": True,
                }

                try:
                    dataset.add_sample(
                        image_path=full_img_path,
                        ground_truth=text,
                        category="synthetic",
                        metadata=metadata,
                    )
                except Exception as e:
                    logger.warning(f"Error adding sample {img_path_str}: {e}")

            logger.info(
                f"Processed {min(len(img_paths), max_samples)} images from SynthText dataset"
            )
            dataset.save_metadata()
            return dataset

        except Exception as e:
            logger.error(f"Error processing SynthText dataset: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and process SynthText dataset")

    parser.add_argument("--output-dir", type=str, help="Directory to save the dataset to")

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process (SynthText is very large)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    max_samples = args.max_samples

    downloader = SynthTextDownloader(output_dir)

    try:
        # Download the dataset
        dataset_path = downloader.download()

        # Process the dataset
        dataset = downloader.process(dataset_path, max_samples)

        logger.info(
            f"Successfully downloaded and processed SynthText dataset with {len(dataset)} samples"
        )
        logger.info(f"Dataset saved to {dataset.processed_dir}")

    except Exception as e:
        logger.error(f"Error downloading/processing SynthText dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
