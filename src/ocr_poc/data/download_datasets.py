"""
Dataset Download Script for OCR POC.

This script downloads selected datasets from the DatasetRegistry
and processes them for use in the OCR evaluation framework.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from ..config.settings import settings
from .dataset import DatasetCollection
from .dataset_research import DatasetRegistry
from .dataset_utils import DatasetDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ocr_poc.data.download_datasets")


# Datasets that are freely available for download
FREE_DATASETS = [
    "SynthText",  # Synthetic data, good for training and testing
    "HierText",  # Good for hierarchical text detection
]

# Datasets that require registration but are useful for research
REGISTRATION_DATASETS = [
    "IAM-Handwriting",  # Standard handwriting dataset
    "ICDAR-2015-Robust-Reading",  # Good for scene text
]


class DatasetDownloadManager:
    """Manages the downloading and processing of OCR datasets."""

    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize dataset download manager.

        Args:
            download_dir: Directory to download datasets to. If None, use default from settings.
        """
        self.download_dir = download_dir or settings.dataset.raw_data_dir
        self.downloader = DatasetDownloader(self.download_dir)
        self.dataset_collection = DatasetCollection(self.download_dir)
        logger.info(
            f"Initialized DatasetDownloadManager with download directory: {self.download_dir}"
        )

    def download_dataset(self, dataset_id: str) -> Path:
        """
        Download a dataset by its ID from the registry.

        Args:
            dataset_id: ID of the dataset in the DatasetRegistry.

        Returns:
            Path to the downloaded dataset.
        """
        # Get dataset info from registry
        dataset_info = DatasetRegistry.get_dataset_info(dataset_id)
        if not dataset_info:
            logger.error(f"Dataset {dataset_id} not found in registry")
            raise ValueError(f"Dataset {dataset_id} not found in registry")

        logger.info(f"Downloading dataset: {dataset_info['name']} from {dataset_info['url']}")

        # Create download configuration
        download_config = {
            "name": dataset_id,
            "url": dataset_info["url"],
            # Additional configuration could be added here based on dataset requirements
        }

        # Download the dataset
        try:
            dataset_path = self.downloader.download_dataset(download_config)
            logger.info(f"Downloaded dataset {dataset_id} to {dataset_path}")
            return dataset_path
        except Exception as e:
            logger.error(f"Error downloading dataset {dataset_id}: {e}")
            raise

    def download_selected_datasets(self, dataset_ids: List[str]) -> None:
        """
        Download multiple datasets by their IDs.

        Args:
            dataset_ids: List of dataset IDs to download.
        """
        for dataset_id in dataset_ids:
            try:
                self.download_dataset(dataset_id)
                logger.info(f"Completed download of dataset: {dataset_id}")
            except Exception as e:
                logger.error(f"Failed to download dataset {dataset_id}: {e}")

    def download_free_datasets(self) -> None:
        """Download all freely available datasets."""
        logger.info(f"Downloading {len(FREE_DATASETS)} freely available datasets")
        self.download_selected_datasets(FREE_DATASETS)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download OCR datasets for evaluation")

    parser.add_argument("--datasets", nargs="+", help="IDs of datasets to download")

    parser.add_argument(
        "--list", action="store_true", help="List available datasets instead of downloading"
    )

    parser.add_argument(
        "--download-free", action="store_true", help="Download all freely available datasets"
    )

    parser.add_argument("--download-dir", type=str, help="Directory to download datasets to")

    return parser.parse_args()


def list_available_datasets():
    """Display information about available datasets."""
    print("Available OCR Datasets:")
    print("=" * 80)

    all_datasets = DatasetRegistry.list_all_datasets()
    free_datasets = set(FREE_DATASETS)
    registration_datasets = set(REGISTRATION_DATASETS)

    print("\nFreely Available Datasets:")
    for dataset_id in all_datasets:
        if dataset_id in free_datasets:
            dataset_info = DatasetRegistry.get_dataset_info(dataset_id)
            print(f"- {dataset_id}: {dataset_info['name']}")
            print(f"  Description: {dataset_info['description']}")
            print(f"  URL: {dataset_info['url']}")
            print(f"  Categories: {', '.join(dataset_info['categories'])}")
            print(f"  Size: {dataset_info['size']}")
            print()

    print("\nDatasets Requiring Registration:")
    for dataset_id in all_datasets:
        if dataset_id in registration_datasets:
            dataset_info = DatasetRegistry.get_dataset_info(dataset_id)
            print(f"- {dataset_id}: {dataset_info['name']}")
            print(f"  Description: {dataset_info['description']}")
            print(f"  URL: {dataset_info['url']}")
            print(f"  Categories: {', '.join(dataset_info['categories'])}")
            print(f"  Size: {dataset_info['size']}")
            print()

    print("\nOther Available Datasets:")
    other_datasets = set(all_datasets) - free_datasets - registration_datasets
    for dataset_id in other_datasets:
        dataset_info = DatasetRegistry.get_dataset_info(dataset_id)
        print(f"- {dataset_id}: {dataset_info['name']}")
        print(f"  Categories: {', '.join(dataset_info['categories'])}")
        print(f"  License: {dataset_info['license']}")
        print()


def main():
    """Main entry point for the dataset download script."""
    args = parse_args()

    if args.list:
        list_available_datasets()
        return

    download_dir = Path(args.download_dir) if args.download_dir else None
    manager = DatasetDownloadManager(download_dir)

    if args.download_free:
        logger.info("Downloading all freely available datasets")
        manager.download_free_datasets()

    elif args.datasets:
        logger.info(f"Downloading selected datasets: {', '.join(args.datasets)}")
        manager.download_selected_datasets(args.datasets)

    else:
        logger.info(
            "No datasets specified for download. Use --datasets or --download-free options."
        )
        print("Run with --list to see available datasets")


if __name__ == "__main__":
    main()
