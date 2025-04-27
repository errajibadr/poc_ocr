"""
Dataset handling module for OCR evaluation.

This module provides classes for managing OCR datasets, including loading,
preprocessing, and organizing data for evaluation.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from PIL import Image

from ..config.settings import settings
from ..utils.logging import logger


class OCRDataset:
    """Class for managing OCR datasets."""

    def __init__(self, name: str, base_dir: Optional[Path] = None):
        """
        Initialize OCR dataset.

        Args:
            name: Name of the dataset.
            base_dir: Base directory for the dataset. If None, use default from settings.
        """
        self.name = name
        self.base_dir = base_dir or settings.dataset.raw_data_dir / name
        self.processed_dir = settings.dataset.processed_data_dir / name
        self.metadata_file = self.processed_dir / "metadata.json"

        # Create directories if they don't exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.metadata: Dict[str, Any] = {}
        self._load_metadata()

        logger.info(f"Initialized dataset: {name} at {self.base_dir}")

    def _load_metadata(self) -> None:
        """Load dataset metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.debug(f"Loaded metadata for dataset {self.name}")
            except Exception as e:
                logger.error(f"Error loading metadata for dataset {self.name}: {e}")
                self.metadata = {"name": self.name, "samples": []}
        else:
            logger.debug(f"No metadata file found for dataset {self.name}, creating new")
            self.metadata = {"name": self.name, "samples": []}

    def save_metadata(self) -> None:
        """Save dataset metadata to JSON file."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"Saved metadata for dataset {self.name}")
        except Exception as e:
            logger.error(f"Error saving metadata for dataset {self.name}: {e}")

    def add_sample(
        self,
        image_path: Union[str, Path],
        ground_truth: str,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a sample to the dataset.

        Args:
            image_path: Path to the image file.
            ground_truth: Ground truth text for the image.
            category: Category for the sample (e.g., "handwritten", "printed").
            metadata: Additional metadata for the sample.

        Returns:
            Dictionary with sample metadata.
        """
        # Convert to Path and get relative path if within base_dir
        image_path = Path(image_path)
        try:
            rel_path = image_path.relative_to(self.base_dir)
            storage_path = str(rel_path)
        except ValueError:
            # If image is outside base_dir, copy it to base_dir
            storage_path = str(image_path.name)
            if not (self.base_dir / storage_path).exists():
                import shutil

                shutil.copy(image_path, self.base_dir / storage_path)
                logger.debug(f"Copied image to dataset directory: {storage_path}")

        # Create sample metadata
        sample = {
            "id": len(self.metadata["samples"]),
            "image_path": storage_path,
            "ground_truth": ground_truth,
            "category": category or "default",
            "metadata": metadata or {},
        }

        # Add to metadata
        self.metadata["samples"].append(sample)
        self.save_metadata()

        logger.info(f"Added sample {sample['id']} to dataset {self.name}")
        return sample

    def get_sample(self, sample_id: int) -> Dict[str, Any]:
        """
        Get sample by ID.

        Args:
            sample_id: ID of the sample to get.

        Returns:
            Dictionary with sample metadata.

        Raises:
            IndexError: If sample not found.
        """
        if not (0 <= sample_id < len(self.metadata["samples"])):
            raise IndexError(f"Sample ID {sample_id} not found in dataset {self.name}")

        return self.metadata["samples"][sample_id]

    def get_image_path(self, sample_id: int) -> Path:
        """
        Get path to sample image.

        Args:
            sample_id: ID of the sample to get image for.

        Returns:
            Path to the image file.
        """
        sample = self.get_sample(sample_id)
        return self.base_dir / sample["image_path"]

    def load_image(self, sample_id: int) -> Image.Image:
        """
        Load sample image.

        Args:
            sample_id: ID of the sample to load image for.

        Returns:
            PIL Image object.
        """
        image_path = self.get_image_path(sample_id)
        try:
            return Image.open(image_path)
        except Exception as e:
            logger.error(f"Error loading image for sample {sample_id}: {e}")
            raise

    def get_ground_truth(self, sample_id: int) -> str:
        """
        Get ground truth text for sample.

        Args:
            sample_id: ID of the sample to get ground truth for.

        Returns:
            Ground truth text.
        """
        sample = self.get_sample(sample_id)
        return sample["ground_truth"]

    def get_samples_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get samples by category.

        Args:
            category: Category to filter by.

        Returns:
            List of sample dictionaries.
        """
        return [s for s in self.metadata["samples"] if s["category"] == category]

    def __len__(self) -> int:
        """Get number of samples in the dataset."""
        return len(self.metadata["samples"])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset."""
        return iter(self.metadata["samples"])


class DatasetCollection:
    """Collection of OCR datasets."""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize dataset collection.

        Args:
            base_dir: Base directory for all datasets. If None, use default from settings.
        """
        self.base_dir = base_dir or settings.dataset.raw_data_dir
        self.datasets: Dict[str, OCRDataset] = {}

        logger.info(f"Initialized dataset collection at {self.base_dir}")

    def load_dataset(self, name: str) -> OCRDataset:
        """
        Load a dataset by name.

        Args:
            name: Name of the dataset to load.

        Returns:
            OCRDataset object.
        """
        if name not in self.datasets:
            self.datasets[name] = OCRDataset(name, self.base_dir / name)
        return self.datasets[name]

    def create_dataset(self, name: str) -> OCRDataset:
        """
        Create a new dataset.

        Args:
            name: Name of the new dataset.

        Returns:
            OCRDataset object.
        """
        if name in self.datasets:
            logger.warning(f"Dataset {name} already exists, returning existing dataset")
            return self.datasets[name]

        dataset = OCRDataset(name, self.base_dir / name)
        self.datasets[name] = dataset
        return dataset

    def list_datasets(self) -> List[str]:
        """
        List all available datasets.

        Returns:
            List of dataset names.
        """
        # Look for directories in the base directory
        dataset_dirs = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        # Add loaded datasets that might not be in the base directory
        dataset_dirs.extend([name for name in self.datasets if name not in dataset_dirs])
        return sorted(dataset_dirs)

    def get_dataset(self, name: str) -> OCRDataset:
        """
        Get a dataset by name, loading it if necessary.

        Args:
            name: Name of the dataset to get.

        Returns:
            OCRDataset object.

        Raises:
            KeyError: If dataset not found.
        """
        if name not in self.datasets:
            if (self.base_dir / name).exists() and (self.base_dir / name).is_dir():
                return self.load_dataset(name)
            raise KeyError(f"Dataset not found: {name}")
        return self.datasets[name]
