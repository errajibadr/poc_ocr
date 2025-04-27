"""
OCR Dataset Research.

This module contains information about well-known OCR datasets
and utilities to help with dataset research.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..config.settings import settings


class DatasetRegistry:
    """Registry of known OCR datasets with metadata."""

    # Dataset categories
    DOCUMENT = "document"  # Full document images (pages, forms, etc.)
    HANDWRITTEN = "handwritten"  # Handwritten text
    SCENE = "scene"  # Text in natural scenes
    HISTORICAL = "historical"  # Historical documents
    SCIENTIFIC = "scientific"  # Scientific documents (math, diagrams)
    MULTILINGUAL = "multilingual"  # Non-English or multilingual text

    # Dataset registry with metadata
    DATASETS = {
        # Document OCR datasets
        "ICDAR-2019-READ": {
            "name": "ICDAR 2019 READ Dataset",
            "description": "Competition dataset for document analysis from the International Conference on Document Analysis and Recognition (ICDAR).",
            "url": "https://rrc.cvc.uab.es/?ch=10",
            "categories": [DOCUMENT],
            "license": "Research only",
            "language": ["English"],
            "paper": "https://ieeexplore.ieee.org/document/8649144",
            "ground_truth_format": "XML",
            "size": "~3,600 images",
            "challenges": ["Document layout analysis", "Text recognition"],
        },
        "PubLayNet": {
            "name": "PubLayNet",
            "description": "Large dataset of document images with annotations for document layout analysis tasks.",
            "url": "https://github.com/ibm-aur-nlp/PubLayNet",
            "categories": [DOCUMENT, SCIENTIFIC],
            "license": "CC BY-NC 4.0",
            "language": ["English"],
            "paper": "https://arxiv.org/abs/1908.07836",
            "ground_truth_format": "COCO JSON",
            "size": "~360,000 document images",
            "challenges": ["Document layout analysis", "Region segmentation"],
        },
        "UW3": {
            "name": "University of Washington Document Image Database III",
            "description": "Collection of document images with ground truth for OCR and document analysis.",
            "url": "http://tc11.cvc.uab.es/datasets/UW3-English_1",
            "categories": [DOCUMENT],
            "license": "Research only",
            "language": ["English"],
            "paper": None,
            "ground_truth_format": "DAFS",
            "size": "~1,600 images",
            "challenges": ["Document structure", "OCR"],
        },
        "DocVQA": {
            "name": "Document Visual Question Answering",
            "description": "Dataset for question answering on document images.",
            "url": "https://www.docvqa.org/",
            "categories": [DOCUMENT],
            "license": "Research only",
            "language": ["English"],
            "paper": "https://arxiv.org/abs/2007.00398",
            "ground_truth_format": "JSON",
            "size": "~50,000 questions on ~12,000 documents",
            "challenges": ["Document understanding", "Visual Question Answering"],
        },
        # Handwritten text datasets
        "IAM-Handwriting": {
            "name": "IAM Handwriting Database",
            "description": "Collection of handwritten English text samples.",
            "url": "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
            "categories": [HANDWRITTEN],
            "license": "Research only",
            "language": ["English"],
            "paper": "http://www.fki.inf.unibe.ch/databases/iam-handwriting-database",
            "ground_truth_format": "ASCII text",
            "size": "~13,000 lines of text from ~500 writers",
            "challenges": ["Handwriting recognition", "Writer identification"],
        },
        "RIMES": {
            "name": "RIMES Database",
            "description": "French handwritten letters and forms.",
            "url": "http://www.a2ialab.com/doku.php?id=rimes_database:start",
            "categories": [HANDWRITTEN, MULTILINGUAL],
            "license": "Research only",
            "language": ["French"],
            "paper": None,
            "ground_truth_format": "XML",
            "size": "~12,000 handwritten pages",
            "challenges": ["Handwriting recognition", "Form analysis"],
        },
        # Scene text datasets
        "COCO-Text": {
            "name": "COCO-Text",
            "description": "Text annotations for the MS COCO dataset with text in natural images.",
            "url": "https://vision.cornell.edu/se3/coco-text-2/",
            "categories": [SCENE],
            "license": "Creative Commons",
            "language": ["Multiple"],
            "paper": "https://arxiv.org/abs/1601.07140",
            "ground_truth_format": "JSON",
            "size": "~63,000 images with ~173,000 text annotations",
            "challenges": ["Scene text detection", "Text recognition in the wild"],
        },
        "ICDAR-2015-Robust-Reading": {
            "name": "ICDAR 2015 Robust Reading Competition",
            "description": "Competition dataset for text detection and recognition in natural scenes.",
            "url": "https://rrc.cvc.uab.es/?ch=4",
            "categories": [SCENE],
            "license": "Research only",
            "language": ["English", "Multiple"],
            "paper": "https://ieeexplore.ieee.org/document/7333942",
            "ground_truth_format": "Text files with coordinates",
            "size": "~1,500 images",
            "challenges": ["Incidental scene text detection", "Text recognition"],
        },
        "SynthText": {
            "name": "SynthText",
            "description": "Synthetic dataset of natural images with text rendered onto them.",
            "url": "https://github.com/ankush-me/SynthText",
            "categories": [SCENE],
            "license": "Research only",
            "language": ["English"],
            "paper": "https://www.robots.ox.ac.uk/~vgg/data/scenetext/",
            "ground_truth_format": "Matlab data files",
            "size": "~800,000 images with ~8 million synthetic word instances",
            "challenges": ["Text detection", "Text recognition"],
        },
        "HierText": {
            "name": "HierText",
            "description": "Hierarchical text detection dataset with word, line, and paragraph annotations.",
            "url": "https://github.com/google-research-datasets/hiertext",
            "categories": [SCENE],
            "license": "CC BY 4.0",
            "language": ["Multiple"],
            "paper": "https://arxiv.org/abs/2203.15143",
            "ground_truth_format": "JSON",
            "size": "~8,000 images",
            "challenges": ["Hierarchical text detection", "Text structure analysis"],
        },
        # Historical document datasets
        "READ-ICDAR2017": {
            "name": "READ Dataset ICDAR 2017",
            "description": "Historical handwritten documents for the ICDAR 2017 competition.",
            "url": "https://readcoop.eu/datasets/",
            "categories": [HISTORICAL, HANDWRITTEN],
            "license": "Research only",
            "language": ["Multiple historical languages"],
            "paper": "https://ieeexplore.ieee.org/document/8270164",
            "ground_truth_format": "PAGE XML",
            "size": "~450,000 images",
            "challenges": ["Historical document analysis", "Handwriting recognition"],
        },
        # Scientific document datasets
        "CROHME": {
            "name": "CROHME: Competition on Recognition of Online Handwritten Mathematical Expressions",
            "description": "Dataset for handwritten mathematical expression recognition.",
            "url": "https://www.isical.ac.in/~crohme/",
            "categories": [SCIENTIFIC, HANDWRITTEN],
            "license": "Research only",
            "language": ["Mathematical notation"],
            "paper": "https://ieeexplore.ieee.org/document/6628795",
            "ground_truth_format": "InkML and LaTeX",
            "size": "~10,000 expressions",
            "challenges": ["Mathematical formula recognition", "Symbol segmentation"],
        },
        # Multilingual datasets
        "MLT-ICDAR2019": {
            "name": "Multi-lingual Text ICDAR 2019",
            "description": "Multi-lingual text detection and recognition dataset.",
            "url": "https://rrc.cvc.uab.es/?ch=15",
            "categories": [SCENE, MULTILINGUAL],
            "license": "Research only",
            "language": [
                "Arabic",
                "Chinese",
                "English",
                "French",
                "German",
                "Hindi",
                "Italian",
                "Japanese",
                "Korean",
                "Russian",
            ],
            "paper": "https://ieeexplore.ieee.org/document/8977597",
            "ground_truth_format": "Text files with coordinates",
            "size": "~20,000 images with ~200,000 annotations",
            "challenges": ["Multi-lingual text detection", "Script identification"],
        },
    }

    @classmethod
    def get_datasets_by_category(cls, category: str) -> Dict[str, Dict]:
        """
        Get datasets by category.

        Args:
            category: Category to filter by.

        Returns:
            Dictionary of datasets matching the category.
        """
        return {k: v for k, v in cls.DATASETS.items() if category in v.get("categories", [])}

    @classmethod
    def get_dataset_info(cls, dataset_id: str) -> Optional[Dict]:
        """
        Get information about a specific dataset.

        Args:
            dataset_id: ID of the dataset to get info for.

        Returns:
            Dictionary with dataset information or None if not found.
        """
        return cls.DATASETS.get(dataset_id)

    @classmethod
    def list_all_datasets(cls) -> List[str]:
        """
        List all available datasets.

        Returns:
            List of dataset IDs.
        """
        return list(cls.DATASETS.keys())

    @classmethod
    def get_research_summary(cls) -> Dict:
        """
        Get a summary of available datasets for research purposes.

        Returns:
            Dictionary with dataset research summary.
        """
        categories = {
            cls.DOCUMENT: "Document OCR",
            cls.HANDWRITTEN: "Handwritten Text",
            cls.SCENE: "Scene Text",
            cls.HISTORICAL: "Historical Documents",
            cls.SCIENTIFIC: "Scientific Documents",
            cls.MULTILINGUAL: "Multilingual Text",
        }

        summary = {
            "total_datasets": len(cls.DATASETS),
            "categories": {
                cat: len(cls.get_datasets_by_category(cat)) for cat in categories.keys()
            },
            "category_names": categories,
            "languages": set(),
        }

        # Collect unique languages
        for dataset in cls.DATASETS.values():
            for lang in dataset.get("language", []):
                summary["languages"].add(lang)

        summary["languages"] = list(summary["languages"])

        return summary


def export_dataset_info(output_file: Optional[Path] = None) -> Path:
    """
    Export dataset information to a JSON file.

    Args:
        output_file: Path to output file. If None, use default from settings.

    Returns:
        Path to the output file.
    """
    if output_file is None:
        # Ensure we have a place to store metadata
        output_dir = Path(settings.dataset.raw_data_dir).parent / "metadata"
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir / "dataset_research.json"

    summary = DatasetRegistry.get_research_summary()
    datasets = {k: v for k, v in DatasetRegistry.DATASETS.items()}

    data = {"summary": summary, "datasets": datasets}

    with open(str(output_file), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Dataset information exported to {output_file}")
    return output_file


def main():
    """CLI entry point for dataset research tools."""
    summary = DatasetRegistry.get_research_summary()
    print(f"Found {summary['total_datasets']} OCR datasets:")

    for category, count in summary["categories"].items():
        if count > 0:
            print(f"- {summary['category_names'][category]}: {count} datasets")

    print(f"\nLanguages covered: {', '.join(sorted(summary['languages']))}")
    print("\nUse export_dataset_info() to export detailed information to a JSON file.")


if __name__ == "__main__":
    main()
