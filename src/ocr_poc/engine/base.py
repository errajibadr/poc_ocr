"""
Base OCR engine interface and plugin system.

This module defines the base interface for OCR engines and implements
the Strategy Pattern for pluggable OCR engines.
"""

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..utils.logging import logger


class OCRResult:
    """Class to store OCR results in a standardized format."""

    def __init__(
        self,
        text: str,
        confidence: float,
        bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        engine_name: str = "unknown",
        processing_time: float = 0.0,
        raw_result: Any = None,
    ):
        """
        Initialize OCR result.

        Args:
            text: Recognized text.
            confidence: Confidence score (0-1).
            bounding_boxes: List of bounding boxes (x1, y1, x2, y2) for recognized text regions.
            engine_name: Name of the OCR engine.
            processing_time: Time taken to process the image in seconds.
            raw_result: Raw result from the OCR engine.
        """
        self.text = text
        self.confidence = confidence
        self.bounding_boxes = bounding_boxes or []
        self.engine_name = engine_name
        self.processing_time = processing_time
        self.raw_result = raw_result

    def __str__(self) -> str:
        """Return string representation of OCR result."""
        return (
            f"OCRResult(engine={self.engine_name}, "
            f"confidence={self.confidence:.2f}, "
            f'text="{self.text[:50]}{"..." if len(self.text) > 50 else ""}")'
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert OCR result to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounding_boxes": self.bounding_boxes,
            "engine_name": self.engine_name,
            "processing_time": self.processing_time,
        }


class BaseOCREngine(abc.ABC):
    """Abstract base class for OCR engines."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OCR engine.

        Args:
            config: Dictionary with engine-specific configuration.
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        logger.info(f"Initializing OCR engine: {self.name}")

    @abc.abstractmethod
    def process_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs: Any,
    ) -> OCRResult:
        """
        Process an image and return OCR result.

        Args:
            image: Image to process. Can be a file path, PIL Image, or numpy array.
            **kwargs: Additional engine-specific parameters.

        Returns:
            OCRResult object with recognized text and metadata.
        """
        raise NotImplementedError

    def load_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load image from various input types.

        Args:
            image: Image to load. Can be a file path, PIL Image, or numpy array.

        Returns:
            PIL Image object.
        """
        if isinstance(image, (str, Path)):
            return Image.open(image)
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def validate(self) -> bool:
        """
        Validate that the OCR engine is properly configured.

        Returns:
            True if the engine is properly configured, False otherwise.
        """
        return True

    def __str__(self) -> str:
        """Return string representation of OCR engine."""
        return f"{self.name} OCR Engine"


class OCREngineRegistry:
    """Registry for OCR engines to implement the Strategy Pattern."""

    _engines: Dict[str, type[BaseOCREngine]] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> callable:
        """
        Register OCR engine class with the registry.

        Args:
            name: Name to register the engine with. If None, use class name.

        Returns:
            Decorator function.
        """

        def decorator(engine_cls: type[BaseOCREngine]) -> type[BaseOCREngine]:
            """Register engine class with registry."""
            engine_name = name or engine_cls.__name__
            cls._engines[engine_name] = engine_cls
            logger.debug(f"Registered OCR engine: {engine_name}")
            return engine_cls

        return decorator

    @classmethod
    def get_engine(cls, name: str) -> type[BaseOCREngine]:
        """
        Get OCR engine class by name.

        Args:
            name: Name of the engine to get.

        Returns:
            OCR engine class.

        Raises:
            KeyError: If engine not found.
        """
        if name not in cls._engines:
            raise KeyError(f"OCR engine not found: {name}")
        return cls._engines[name]

    @classmethod
    def list_engines(cls) -> List[str]:
        """
        List all registered OCR engines.

        Returns:
            List of engine names.
        """
        return list(cls._engines.keys())

    @classmethod
    def create_engine(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseOCREngine:
        """
        Create OCR engine instance by name.

        Args:
            name: Name of the engine to create.
            config: Dictionary with engine-specific configuration.

        Returns:
            OCR engine instance.
        """
        engine_cls = cls.get_engine(name)
        return engine_cls(config)
