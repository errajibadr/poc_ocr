"""
Configuration settings for the OCR POC project.

This module uses Pydantic for configuration management, allowing for
type validation and environment variable loading.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define project base directory
PROJECT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
DATA_DIR = PROJECT_DIR / "data"


class OCREngineSettings(BaseSettings):
    """Settings for OCR engines."""

    # Tesseract settings
    tesseract_cmd: str = Field(default="tesseract", description="Path to Tesseract executable")
    tesseract_lang: str = Field(default="eng", description="Default language for Tesseract")
    tesseract_config: str = Field(default="", description="Additional Tesseract configuration")

    # EasyOCR settings (to be expanded after research)
    easyocr_languages: List[str] = Field(default=["en"], description="Languages for EasyOCR")

    model_config = SettingsConfigDict(env_prefix="OCR_ENGINE_", env_file=".env", extra="ignore")


class DatasetSettings(BaseSettings):
    """Settings for dataset management."""

    raw_data_dir: Path = Field(
        default=DATA_DIR / "raw", description="Directory for raw dataset files"
    )
    processed_data_dir: Path = Field(
        default=DATA_DIR / "processed", description="Directory for processed dataset files"
    )
    results_dir: Path = Field(
        default=DATA_DIR / "results", description="Directory for evaluation results"
    )

    # Dataset parameters
    default_image_size: tuple[int, int] = Field(
        default=(800, 600), description="Default image size for preprocessing (width, height)"
    )

    @field_validator("raw_data_dir", "processed_data_dir", "results_dir", mode="after")
    def create_directory_if_not_exists(cls, v: Path) -> Path:
        """Create directory if it doesn't exist."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    model_config = SettingsConfigDict(env_prefix="DATASET_", env_file=".env", extra="ignore")


class EvaluationSettings(BaseSettings):
    """Settings for OCR evaluation."""

    # Evaluation metrics
    metrics: List[str] = Field(
        default=["character_accuracy", "word_accuracy", "wer"],
        description="Metrics to use for evaluation",
    )

    # Visualization settings
    save_visualizations: bool = Field(
        default=True, description="Whether to save visualization results"
    )

    # Performance settings
    parallel_evaluation: bool = Field(
        default=True, description="Whether to use parallel processing for evaluation"
    )

    model_config = SettingsConfigDict(env_prefix="EVAL_", env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Main settings class that combines all settings."""

    # Project settings
    project_dir: Path = Field(default=PROJECT_DIR, description="Project root directory")
    debug: bool = Field(default=False, description="Debug mode")

    # Component settings
    ocr_engine: OCREngineSettings = Field(
        default_factory=OCREngineSettings, description="OCR engine settings"
    )
    dataset: DatasetSettings = Field(
        default_factory=DatasetSettings, description="Dataset settings"
    )
    evaluation: EvaluationSettings = Field(
        default_factory=EvaluationSettings, description="Evaluation settings"
    )

    model_config = SettingsConfigDict(env_prefix="OCR_POC_", env_file=".env", extra="ignore")


# Create a global settings instance
settings = Settings()
