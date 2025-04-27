"""
Tests for the dataset module.
"""

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from ocr_poc.data.dataset import DatasetCollection, OCRDataset


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test datasets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new("RGB", (100, 50), color="white")
        img.save(tmp.name)
        yield Path(tmp.name)
        os.unlink(tmp.name)


def test_dataset_creation(temp_dir):
    """Test creating a dataset."""
    dataset = OCRDataset("test_dataset", temp_dir)
    assert dataset.name == "test_dataset"
    assert dataset.base_dir == temp_dir / "test_dataset"
    assert dataset.processed_dir == temp_dir / "test_dataset" / "processed"
    assert os.path.exists(dataset.base_dir)
    assert os.path.exists(dataset.processed_dir)


def test_add_sample(temp_dir, sample_image):
    """Test adding a sample to a dataset."""
    dataset = OCRDataset("test_dataset", temp_dir)
    sample = dataset.add_sample(
        image_path=sample_image,
        ground_truth="Test text",
        category="test",
        metadata={"source": "test"},
    )

    # Check sample metadata
    assert sample["id"] == 0
    assert sample["ground_truth"] == "Test text"
    assert sample["category"] == "test"
    assert sample["metadata"]["source"] == "test"

    # Check that the image was copied to the dataset directory
    assert os.path.exists(dataset.base_dir / sample["image_path"])

    # Check that metadata was saved
    assert os.path.exists(dataset.metadata_file)
    assert len(dataset.metadata["samples"]) == 1
    assert dataset.metadata["samples"][0]["ground_truth"] == "Test text"


def test_dataset_collection(temp_dir):
    """Test the dataset collection."""
    collection = DatasetCollection(temp_dir)

    # Create a dataset
    dataset1 = collection.create_dataset("dataset1")
    assert dataset1.name == "dataset1"
    assert dataset1.base_dir == temp_dir / "dataset1"

    # Get the same dataset
    dataset2 = collection.get_dataset("dataset1")
    assert dataset2 is dataset1

    # List datasets
    datasets = collection.list_datasets()
    assert "dataset1" in datasets
