"""
Dataset utilities for OCR POC.

This module provides utilities for downloading, processing, augmenting,
and inspecting OCR datasets.
"""

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter
from skimage import transform as sk_transform

from ..config.settings import settings
from .dataset import DatasetCollection, OCRDataset

# Optional dependencies


class DatasetDownloader:
    """Utilities for downloading OCR datasets from various sources."""

    def __init__(self, download_dir: Optional[Path] = None):
        """
        Initialize dataset downloader.

        Args:
            download_dir: Directory to download datasets to. If None, use default from settings.
        """
        self.download_dir = download_dir or settings.dataset.raw_data_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__ + ".DatasetDownloader")

    def download_file(self, url: str, filename: Optional[str] = None) -> Path:
        """
        Download a file from a URL.

        Args:
            url: URL to download from.
            filename: Name to save the file as. If None, use the basename of the URL.

        Returns:
            Path to the downloaded file.
        """
        self.logger.info(f"Downloading file from {url}")
        if filename is None:
            filename = os.path.basename(url)

        file_path = self.download_dir / filename

        try:
            # Create parent directories if they don't exist
            os.makedirs(file_path.parent, exist_ok=True)

            # Download the file with progress reporting
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192  # 8 KB

            self.logger.info(f"Saving to {file_path} (size: {total_size / (1024 * 1024):.2f} MB)")

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)

            self.logger.info(f"Download completed: {file_path}")
            return file_path

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading file from {url}: {e}")
            raise

    def extract_archive(
        self, archive_path: Union[str, Path], extract_dir: Optional[Path] = None
    ) -> Path:
        """
        Extract an archive file.

        Args:
            archive_path: Path to the archive file.
            extract_dir: Directory to extract to. If None, use a directory with the same
                name as the archive in the download directory.

        Returns:
            Path to the extracted directory.
        """
        archive_path = Path(archive_path)
        self.logger.info(f"Extracting archive {archive_path}")

        if extract_dir is None:
            extract_dir = self.download_dir / archive_path.stem

        os.makedirs(extract_dir, exist_ok=True)

        archive_suffix = archive_path.suffix.lower()

        try:
            if archive_suffix == ".zip":
                self._extract_zip(archive_path, extract_dir)
            elif archive_suffix in (".tar", ".gz", ".tgz", ".bz2", ".tbz2"):
                self._extract_tar(archive_path, extract_dir)
            elif archive_suffix == ".rar":
                self._extract_rar(archive_path, extract_dir)
            else:
                self.logger.warning(f"Unsupported archive format: {archive_suffix}")
                raise ValueError(f"Unsupported archive format: {archive_suffix}")

            self.logger.info(f"Extraction completed: {extract_dir}")
            return extract_dir

        except Exception as e:
            self.logger.error(f"Error extracting archive {archive_path}: {e}")
            raise

    def _extract_zip(self, archive_path: Path, extract_dir: Path) -> None:
        """Extract a ZIP archive."""
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    def _extract_tar(self, archive_path: Path, extract_dir: Path) -> None:
        """Extract a TAR archive (including compressed variants)."""
        import tarfile

        with tarfile.open(archive_path, "r:*") as tar_ref:
            tar_ref.extractall(extract_dir)

    def _extract_rar(self, archive_path: Path, extract_dir: Path) -> None:
        """Extract a RAR archive."""
        # Make rarfile an optional dependency

        import rarfile

        with rarfile.RarFile(archive_path, "r") as rar_ref:
            rar_ref.extractall(extract_dir)

    def download_dataset(self, dataset_config: Dict[str, Any]) -> Path:
        """
        Download a dataset based on configuration.

        Args:
            dataset_config: Dictionary with dataset configuration.
                Must contain 'url' key, and optionally 'name', 'format', etc.

        Returns:
            Path to the downloaded/extracted dataset.
        """
        dataset_name = dataset_config.get("name", "unknown_dataset")
        self.logger.info(f"Downloading dataset: {dataset_name}")

        # Create dataset directory
        dataset_dir = self.download_dir / dataset_name
        os.makedirs(dataset_dir, exist_ok=True)

        # Check if URL is provided
        if "url" not in dataset_config:
            self.logger.error(f"No URL provided for dataset {dataset_name}")
            raise ValueError(f"Dataset configuration for {dataset_name} must contain 'url' key")

        url = dataset_config["url"]
        file_format = dataset_config.get("format", self._guess_format_from_url(url))

        # Handle different dataset formats
        if file_format in ["zip", "tar", "tgz", "gz", "rar"]:
            # Download archive file
            archive_filename = dataset_config.get("filename", os.path.basename(url))
            archive_path = self.download_file(url, archive_filename)

            # Extract archive
            extract_path = dataset_config.get("extract_path")
            extract_dir = self.extract_archive(archive_path, extract_path)

            return extract_dir

        elif file_format == "directory":
            # This is a multi-file dataset with a directory structure
            # We'll need to download files according to a manifest or listing
            if "files" in dataset_config:
                for file_info in dataset_config["files"]:
                    file_url = file_info["url"]
                    file_path = file_info.get("path", "")
                    save_path = dataset_dir / file_path
                    os.makedirs(save_path.parent, exist_ok=True)
                    self.download_file(file_url, str(save_path))
            else:
                self.logger.warning(f"No files specified for directory-type dataset {dataset_name}")

            return dataset_dir

        elif file_format in ["csv", "json", "xml", "txt"]:
            # Single metadata file
            metadata_path = self.download_file(url, f"{dataset_name}.{file_format}")
            return metadata_path.parent

        else:
            # Default: just download the file
            file_path = self.download_file(url)
            return file_path.parent

    def _guess_format_from_url(self, url: str) -> str:
        """Guess the format of a dataset from its URL."""
        url_lower = url.lower()

        # Check for common archive extensions
        if url_lower.endswith(".zip"):
            return "zip"
        elif url_lower.endswith(".tar.gz") or url_lower.endswith(".tgz"):
            return "tgz"
        elif url_lower.endswith(".tar"):
            return "tar"
        elif url_lower.endswith(".rar"):
            return "rar"
        elif url_lower.endswith(".gz"):
            return "gz"

        # Check for data file extensions
        elif url_lower.endswith(".json"):
            return "json"
        elif url_lower.endswith(".csv"):
            return "csv"
        elif url_lower.endswith(".xml"):
            return "xml"
        elif url_lower.endswith(".txt"):
            return "txt"

        # Default to binary format
        return "binary"


class FormatConverter:
    """Utilities for converting dataset formats."""

    def __init__(self):
        """Initialize format converter."""
        self.logger = logging.getLogger(__name__ + ".FormatConverter")

    def convert_to_ocr_dataset(
        self,
        input_dir: Union[str, Path],
        dataset_name: str,
        format_type: str,
        target_collection: Optional[DatasetCollection] = None,
    ) -> OCRDataset:
        """
        Convert a dataset to OCRDataset format.

        Args:
            input_dir: Directory with the input dataset.
            dataset_name: Name for the resulting dataset.
            format_type: Input format type (e.g., 'icdar', 'iam', 'coco').
            target_collection: DatasetCollection to add the dataset to.
                If None, create a new dataset without adding to a collection.

        Returns:
            OCRDataset with the converted data.
        """
        self.logger.info(f"Converting dataset {dataset_name} from {format_type} format")

        # Create dataset
        if target_collection:
            dataset = target_collection.create_dataset(dataset_name)
        else:
            dataset = OCRDataset(dataset_name)

        # Implementation will handle different format types
        # TODO: Implement format-specific conversion logic

        return dataset

    def parse_ground_truth(self, gt_file: Union[str, Path], format_type: str) -> Dict[str, Any]:
        """
        Parse ground truth file.

        Args:
            gt_file: Path to ground truth file.
            format_type: Format type of the ground truth file.

        Returns:
            Dictionary with parsed ground truth data.
        """
        self.logger.info(f"Parsing ground truth file {gt_file} as {format_type}")
        gt_file = Path(gt_file)

        # Handle different ground truth formats
        if format_type.lower() == "icdar":
            return self._parse_icdar_gt(gt_file)
        elif format_type.lower() == "txt":
            return self._parse_txt_gt(gt_file)
        elif format_type.lower() == "json":
            return self._parse_json_gt(gt_file)
        elif format_type.lower() == "xml":
            return self._parse_xml_gt(gt_file)
        else:
            self.logger.warning(
                f"Unknown ground truth format: {format_type}, attempting generic parsing"
            )
            return self._parse_generic_gt(gt_file)

    def _parse_icdar_gt(self, gt_file: Path) -> Dict[str, Any]:
        """Parse ICDAR format ground truth."""
        # ICDAR format can be:
        # 1. TXT format with x1,y1,x2,y2,x3,y3,x4,y4,text
        # 2. XML format with TextRegion elements
        # 3. JSON format with regions array

        if gt_file.suffix.lower() == ".txt":
            try:
                bounding_boxes = []
                texts = []

                with open(gt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        # Try to parse line as comma-separated values
                        parts = line.strip().split(",")
                        if len(parts) >= 9:  # At least 8 coordinates + text
                            # Extract coordinates and text
                            coords = [int(p) for p in parts[:8]]
                            text = ",".join(parts[8:])

                            bounding_boxes.append(coords)
                            texts.append(text)

                return {"text": " ".join(texts), "bounding_boxes": bounding_boxes}

            except Exception as e:
                self.logger.error(f"Error parsing ICDAR TXT format: {e}")
                return {"text": ""}

        elif gt_file.suffix.lower() == ".xml":
            return self._parse_xml_gt(gt_file)

        elif gt_file.suffix.lower() == ".json":
            return self._parse_json_gt(gt_file)

        else:
            self.logger.warning(f"Unknown ICDAR ground truth file format: {gt_file.suffix}")
            return {"text": ""}

    def _parse_txt_gt(self, gt_file: Path) -> Dict[str, Any]:
        """Parse plain text ground truth."""
        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            return {"text": text}
        except Exception as e:
            self.logger.error(f"Error parsing TXT ground truth: {e}")
            return {"text": ""}

    def _parse_json_gt(self, gt_file: Path) -> Dict[str, Any]:
        """Parse JSON format ground truth."""
        try:
            with open(gt_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract text from various JSON formats
            texts = []

            # Try different common JSON structures
            if "text" in data:
                texts.append(data["text"])

            if "regions" in data:
                for region in data["regions"]:
                    if "text" in region:
                        texts.append(region["text"])

            if "words" in data:
                for word in data["words"]:
                    if "text" in word:
                        texts.append(word["text"])

            if "lines" in data:
                for line in data["lines"]:
                    if "text" in line:
                        texts.append(line["text"])

            return {"text": " ".join(texts), "json_data": data}

        except Exception as e:
            self.logger.error(f"Error parsing JSON ground truth: {e}")
            return {"text": ""}

    def _parse_xml_gt(self, gt_file: Path) -> Dict[str, Any]:
        """Parse XML format ground truth."""
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(gt_file)
            root = tree.getroot()

            # Extract text from various XML formats
            texts = []

            # Look for text elements with different tags
            for text_elem in root.findall(".//TextRegion/TextEquiv/Unicode"):
                if text_elem.text:
                    texts.append(text_elem.text)

            if not texts:
                for text_elem in root.findall(".//TextLine/TextEquiv/Unicode"):
                    if text_elem.text:
                        texts.append(text_elem.text)

            if not texts:
                for text_elem in root.findall(".//Word/TextEquiv/Unicode"):
                    if text_elem.text:
                        texts.append(text_elem.text)

            return {"text": " ".join(texts)}

        except Exception as e:
            self.logger.error(f"Error parsing XML ground truth: {e}")
            return {"text": ""}

    def _parse_generic_gt(self, gt_file: Path) -> Dict[str, Any]:
        """Parse ground truth with a generic approach based on file extension."""
        if gt_file.suffix.lower() == ".txt":
            return self._parse_txt_gt(gt_file)
        elif gt_file.suffix.lower() == ".json":
            return self._parse_json_gt(gt_file)
        elif gt_file.suffix.lower() == ".xml":
            return self._parse_xml_gt(gt_file)
        else:
            # Try plain text as a fallback
            try:
                with open(gt_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                return {"text": text}
            except Exception as e:
                self.logger.error(f"Error parsing generic ground truth: {e}")
                return {"text": ""}


class DataAugmentation:
    """Utilities for augmenting OCR dataset images."""

    def __init__(self):
        """Initialize data augmentation utilities."""
        self.logger = logging.getLogger(__name__ + ".DataAugmentation")

    def add_noise(
        self, image: Image.Image, noise_type: str = "gaussian", intensity: float = 0.1
    ) -> Image.Image:
        """
        Add noise to an image.

        Args:
            image: Input image.
            noise_type: Type of noise to add ('gaussian', 'salt_pepper', 'speckle').
            intensity: Intensity of noise (0.0 to 1.0).

        Returns:
            Image with added noise.
        """
        self.logger.debug(f"Adding {noise_type} noise with intensity {intensity}")

        # Check if numpy is available
        if np is None:
            self.logger.warning("numpy is required for noise addition but not installed")
            return image

        # Convert PIL image to numpy array for processing
        img_array = np.array(image).astype(np.float32) / 255.0

        if noise_type == "gaussian":
            # Add Gaussian noise
            noise = np.random.normal(0, intensity, img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 1.0)

        elif noise_type == "salt_pepper":
            # Add salt and pepper noise
            noisy_img = img_array.copy()
            # Salt (white) noise
            salt_mask = np.random.random(img_array.shape) < (intensity / 2)
            noisy_img[salt_mask] = 1.0
            # Pepper (black) noise
            pepper_mask = np.random.random(img_array.shape) < (intensity / 2)
            noisy_img[pepper_mask] = 0.0

        elif noise_type == "speckle":
            # Add speckle noise (multiplicative)
            noise = np.random.normal(0, intensity, img_array.shape)
            noisy_img = img_array * (1 + noise)
            noisy_img = np.clip(noisy_img, 0, 1.0)

        else:
            self.logger.warning(f"Unknown noise type: {noise_type}, returning original image")
            return image

        # Convert back to PIL image
        noisy_img = (noisy_img * 255).astype(np.uint8)
        if len(noisy_img.shape) == 3:
            result_image = Image.fromarray(noisy_img, mode="RGB")
        else:
            result_image = Image.fromarray(noisy_img, mode="L")

        return result_image

    def adjust_lighting(
        self, image: Image.Image, brightness: float = 1.0, contrast: float = 1.0
    ) -> Image.Image:
        """
        Adjust lighting of an image.

        Args:
            image: Input image.
            brightness: Brightness adjustment factor (0.5 to 1.5).
            contrast: Contrast adjustment factor (0.5 to 1.5).

        Returns:
            Image with adjusted lighting.
        """
        self.logger.debug(f"Adjusting lighting: brightness={brightness}, contrast={contrast}")

        # Use PIL's enhancement functionality
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        return image

    def geometric_transform(
        self, image: Image.Image, rotation: float = 0.0, shear: float = 0.0, scale: float = 1.0
    ) -> Image.Image:
        """
        Apply geometric transformations to an image.

        Args:
            image: Input image.
            rotation: Rotation angle in degrees.
            shear: Shear angle in degrees.
            scale: Scale factor.

        Returns:
            Transformed image.
        """
        self.logger.debug(
            f"Applying geometric transform: rotation={rotation}, shear={shear}, scale={scale}"
        )

        # Convert PIL image to numpy array
        img_array = np.array(image)

        # Create transformation matrix
        tform = sk_transform.AffineTransform(
            scale=(scale, scale), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear)
        )

        # Apply transformation
        transformed = sk_transform.warp(
            img_array, tform.inverse, mode="constant", preserve_range=True
        ).astype(img_array.dtype)

        # Convert back to PIL image
        if len(transformed.shape) == 3:
            result_image = Image.fromarray(transformed, mode="RGB")
        else:
            result_image = Image.fromarray(transformed, mode="L")

        return result_image

    def apply_blur(self, image: Image.Image, radius: float = 2.0) -> Image.Image:
        """
        Apply blur to an image.

        Args:
            image: Input image.
            radius: Blur radius.

        Returns:
            Blurred image.
        """
        self.logger.debug(f"Applying blur with radius={radius}")
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_jpeg_compression(self, image: Image.Image, quality: int = 50) -> Image.Image:
        """
        Simulate JPEG compression artifacts.

        Args:
            image: Input image.
            quality: JPEG quality (1-100).

        Returns:
            Image with JPEG compression artifacts.
        """
        self.logger.debug(f"Applying JPEG compression with quality={quality}")

        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def create_variations(
        self,
        image: Image.Image,
        num_variations: int = 3,
        variation_params: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Image.Image]:
        """
        Create multiple variations of an image.

        Args:
            image: Input image.
            num_variations: Number of variations to create.
            variation_params: List of parameters for each variation.
                If None, generate random variations.

        Returns:
            List of image variations.
        """
        self.logger.info(f"Creating {num_variations} variations of image")
        import random

        variations = []

        if variation_params is None:
            # Generate random variation parameters
            variation_params = []
            for _ in range(num_variations):
                params = {
                    "noise_type": random.choice(["gaussian", "salt_pepper", "speckle"]),
                    "noise_intensity": random.uniform(0.01, 0.1),
                    "brightness": random.uniform(0.8, 1.2),
                    "contrast": random.uniform(0.8, 1.2),
                    "rotation": random.uniform(-5, 5),
                    "blur_radius": random.uniform(0, 1.5),
                    "jpeg_quality": random.randint(60, 95),
                }
                variation_params.append(params)

        # Apply the variations
        for i, params in enumerate(variation_params[:num_variations]):
            img_variation = image.copy()

            # Apply transformations in a sensible order
            # 1. Adjust lighting
            if "brightness" in params or "contrast" in params:
                brightness = params.get("brightness", 1.0)
                contrast = params.get("contrast", 1.0)
                img_variation = self.adjust_lighting(img_variation, brightness, contrast)

            # 2. Apply geometric transformations
            if "rotation" in params or "shear" in params or "scale" in params:
                rotation = params.get("rotation", 0.0)
                shear = params.get("shear", 0.0)
                scale = params.get("scale", 1.0)
                img_variation = self.geometric_transform(img_variation, rotation, shear, scale)

            # 3. Add noise
            if "noise_type" in params and "noise_intensity" in params:
                noise_type = params.get("noise_type", "gaussian")
                noise_intensity = params.get("noise_intensity", 0.05)
                img_variation = self.add_noise(img_variation, noise_type, noise_intensity)

            # 4. Apply blur
            if "blur_radius" in params and params["blur_radius"] > 0:
                blur_radius = params.get("blur_radius", 0.0)
                img_variation = self.apply_blur(img_variation, blur_radius)

            # 5. Apply JPEG compression artifacts
            if "jpeg_quality" in params:
                quality = params.get("jpeg_quality", 85)
                img_variation = self.apply_jpeg_compression(img_variation, quality)

            self.logger.debug(f"Created variation {i + 1}/{num_variations} with params: {params}")
            variations.append(img_variation)

        return variations


class DatasetInspector:
    """Utilities for inspecting and validating OCR datasets."""

    def __init__(self):
        """Initialize dataset inspector."""
        self.logger = logging.getLogger(__name__ + ".DatasetInspector")

    def generate_statistics(self, dataset: OCRDataset) -> Dict[str, Any]:
        """
        Generate statistics for a dataset.

        Args:
            dataset: Dataset to generate statistics for.

        Returns:
            Dictionary with dataset statistics.
        """
        self.logger.info(f"Generating statistics for dataset {dataset.name}")

        # Initialize statistics
        stats = {
            "name": dataset.name,
            "sample_count": len(dataset),
            "categories": {},
            "text_stats": {
                "min_length": float("inf"),
                "max_length": 0,
                "avg_length": 0,
                "total_chars": 0,
                "total_words": 0,
                "char_frequency": {},
            },
            "image_stats": {
                "min_width": float("inf"),
                "max_width": 0,
                "avg_width": 0,
                "min_height": float("inf"),
                "max_height": 0,
                "avg_height": 0,
                "min_resolution": float("inf"),
                "max_resolution": 0,
                "avg_resolution": 0,
            },
        }

        # Process each sample to collect statistics
        for i, sample in enumerate(dataset):
            # Track progress for large datasets
            if i % 100 == 0 and i > 0:
                self.logger.debug(f"Processed {i} samples for statistics")

            # Count categories
            category = sample.get("category", "uncategorized")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1

            # Text statistics
            text = sample.get("ground_truth", "")
            text_len = len(text)
            word_count = len(text.split())

            stats["text_stats"]["min_length"] = min(stats["text_stats"]["min_length"], text_len)
            stats["text_stats"]["max_length"] = max(stats["text_stats"]["max_length"], text_len)
            stats["text_stats"]["total_chars"] += text_len
            stats["text_stats"]["total_words"] += word_count

            # Character frequency
            for char in text:
                stats["text_stats"]["char_frequency"][char] = (
                    stats["text_stats"]["char_frequency"].get(char, 0) + 1
                )

            # Image statistics if available
            metadata = sample.get("metadata", {})
            if (
                "width" in metadata
                and "height" in metadata
                and metadata["width"]
                and metadata["height"]
            ):
                width = metadata["width"]
                height = metadata["height"]
                resolution = width * height

                stats["image_stats"]["min_width"] = min(stats["image_stats"]["min_width"], width)
                stats["image_stats"]["max_width"] = max(stats["image_stats"]["max_width"], width)
                stats["image_stats"]["min_height"] = min(stats["image_stats"]["min_height"], height)
                stats["image_stats"]["max_height"] = max(stats["image_stats"]["max_height"], height)
                stats["image_stats"]["min_resolution"] = min(
                    stats["image_stats"]["min_resolution"], resolution
                )
                stats["image_stats"]["max_resolution"] = max(
                    stats["image_stats"]["max_resolution"], resolution
                )

                # Accumulators for averages
                stats["image_stats"]["avg_width"] += width
                stats["image_stats"]["avg_height"] += height
                stats["image_stats"]["avg_resolution"] += resolution

        # Calculate averages
        if len(dataset) > 0:
            stats["text_stats"]["avg_length"] = stats["text_stats"]["total_chars"] / len(dataset)
            stats["image_stats"]["avg_width"] = stats["image_stats"]["avg_width"] / len(dataset)
            stats["image_stats"]["avg_height"] = stats["image_stats"]["avg_height"] / len(dataset)
            stats["image_stats"]["avg_resolution"] = stats["image_stats"]["avg_resolution"] / len(
                dataset
            )

        # Handle empty dataset edge cases
        if stats["text_stats"]["min_length"] == float("inf"):
            stats["text_stats"]["min_length"] = 0
        if stats["image_stats"]["min_width"] == float("inf"):
            stats["image_stats"]["min_width"] = 0
        if stats["image_stats"]["min_height"] == float("inf"):
            stats["image_stats"]["min_height"] = 0
        if stats["image_stats"]["min_resolution"] == float("inf"):
            stats["image_stats"]["min_resolution"] = 0

        # Get top 10 most frequent characters
        char_freq = stats["text_stats"]["char_frequency"]
        stats["text_stats"]["most_frequent_chars"] = sorted(
            char_freq.items(), key=lambda x: x[1], reverse=True
        )[:10]

        self.logger.info(f"Completed statistics generation for dataset {dataset.name}")
        return stats

    def validate_dataset(self, dataset: OCRDataset) -> Tuple[bool, List[str]]:
        """
        Validate a dataset for completeness and consistency.

        Args:
            dataset: Dataset to validate.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        self.logger.info(f"Validating dataset {dataset.name}")

        issues = []

        # Check if dataset has samples
        if len(dataset) == 0:
            issues.append("Dataset has no samples")
            return False, issues

        # Validate each sample
        for i, sample in enumerate(dataset):
            # Track progress for large datasets
            if i % 100 == 0 and i > 0:
                self.logger.debug(f"Validated {i} samples")

            sample_id = sample.get("id", i)

            # Check for required fields
            if "image_path" not in sample:
                issues.append(f"Sample {sample_id} missing image_path")

            if "ground_truth" not in sample:
                issues.append(f"Sample {sample_id} missing ground_truth")

            # Check if image exists
            try:
                image_path = dataset.get_image_path(sample_id)
                if not image_path.exists() and not sample.get("metadata", {}).get(
                    "missing_image", False
                ):
                    issues.append(f"Sample {sample_id} image file not found: {image_path}")
            except Exception as e:
                issues.append(f"Sample {sample_id} error checking image path: {e}")

            # Check ground truth
            ground_truth = sample.get("ground_truth", "")
            if not ground_truth:
                issues.append(f"Sample {sample_id} has empty ground truth")

            # Check metadata
            metadata = sample.get("metadata", {})
            if "bounding_boxes" in metadata:
                boxes = metadata["bounding_boxes"]
                if not isinstance(boxes, list):
                    issues.append(f"Sample {sample_id} has invalid bounding_boxes format")
                elif boxes:
                    # Check first box for correct format
                    first_box = boxes[0]
                    if not isinstance(first_box, list) or len(first_box) != 8:
                        issues.append(
                            f"Sample {sample_id} has invalid bounding box format - should have 8 coordinates"
                        )

        # Check if the dataset is valid
        is_valid = len(issues) == 0

        if is_valid:
            self.logger.info(f"Dataset {dataset.name} validation passed")
        else:
            self.logger.warning(
                f"Dataset {dataset.name} validation failed with {len(issues)} issues"
            )

        return is_valid, issues

    def visualize_sample(
        self, dataset: OCRDataset, sample_id: int, output_path: Optional[Path] = None
    ) -> Path:
        """
        Create visualization of a dataset sample.

        Args:
            dataset: Dataset containing the sample.
            sample_id: ID of the sample to visualize.
            output_path: Path to save visualization to. If None, use a default path.

        Returns:
            Path to the visualization file.
        """
        self.logger.info(f"Visualizing sample {sample_id} from dataset {dataset.name}")

        # Set default output path if not provided
        if output_path is None:
            output_dir = settings.dataset.processed_data_dir / dataset.name / "visualizations"
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / f"sample_{sample_id}.png"

        # Ensure output_path is a Path object
        output_path = Path(output_path)

        try:
            # Get the sample data
            sample = dataset.get_sample(sample_id)

            # Open the image
            image = dataset.load_image(sample_id)

            # Prepare for drawing
            try:
                from PIL import ImageDraw, ImageFont
            except ImportError:
                self.logger.warning("PIL ImageDraw is required for visualization")
                # Just save the original image if we can't draw
                image.save(str(output_path))  # Convert Path to str for save method
                return output_path

            # Create a copy of the image to draw on
            visual_img = image.copy()
            draw = ImageDraw.Draw(visual_img)

            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 16)
                except IOError:
                    font = ImageFont.load_default()

            # Draw bounding boxes if available
            metadata = sample.get("metadata", {})
            boxes = metadata.get("bounding_boxes", [])

            for i, box in enumerate(boxes):
                if len(box) == 8:  # x1, y1, x2, y2, x3, y3, x4, y4
                    # Draw polygon
                    polygon = [
                        (box[0], box[1]),  # (x1, y1)
                        (box[2], box[3]),  # (x2, y2)
                        (box[4], box[5]),  # (x3, y3)
                        (box[6], box[7]),  # (x4, y4)
                    ]
                    draw.polygon(polygon, outline="red")

                    # Draw box number at top-left corner
                    draw.text((box[0], box[1] - 20), f"{i}", fill="red", font=font)

            # Draw ground truth and metadata at the bottom of the image
            ground_truth = sample.get("ground_truth", "")
            max_chars = 80  # Maximum characters to display
            if len(ground_truth) > max_chars:
                ground_truth = ground_truth[:max_chars] + "..."

            draw.rectangle(
                [(0, visual_img.height - 60), (visual_img.width, visual_img.height)],
                fill="white",
                outline="black",
            )
            draw.text((10, visual_img.height - 55), f"GT: {ground_truth}", fill="black", font=font)
            draw.text(
                (10, visual_img.height - 35),
                f"Sample ID: {sample_id}, Category: {sample.get('category', 'N/A')}",
                fill="black",
                font=font,
            )

            # Save the visualization - convert Path to str for save method
            visual_img.save(str(output_path))
            self.logger.info(f"Saved visualization to {output_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Error creating visualization for sample {sample_id}: {e}")
            raise


def main():
    """
    Command-line interface for dataset utilities.

    This function will be expanded to provide a CLI for the dataset utilities,
    allowing users to download, convert, and inspect datasets from the command line.
    """
    print("Dataset utilities CLI - to be implemented")
    # TODO: Implement command-line interface


if __name__ == "__main__":
    main()
