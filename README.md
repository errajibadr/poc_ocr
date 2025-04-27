# OCR Proof of Concept

This project is a proof of concept for Optical Character Recognition (OCR) technology evaluation. It allows for testing and comparing different OCR methods on various datasets.

## Features

- Modular architecture for easy swapping of OCR engines
- Comprehensive dataset management for OCR testing
- Multiple evaluation metrics for OCR performance
- Visualization tools for OCR results comparison
- Extensible plugin system for adding new OCR engines

## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR (for using the Tesseract engine)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ocr-poc.git
   cd ocr-poc
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies using UV:
   ```bash
   pip install uv
   uv pip install -e .
   ```

4. For development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

## Project Structure

```
ocr-poc/
├── src/
│   └── ocr_poc/              # Main package
│       ├── config/           # Configuration settings
│       ├── data/             # Dataset handling
│       ├── engine/           # OCR engines
│       │   └── plugins/      # Pluggable OCR engines
│       ├── evaluation/       # Evaluation metrics and visualization
│       └── utils/            # Utility functions
├── tests/                    # Test suite
├── data/                     # Dataset storage
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed datasets
│   └── results/              # Evaluation results
└── docs/                     # Documentation
```

## Usage

### Loading a Dataset

```python
from ocr_poc.data.dataset import DatasetCollection

# Initialize dataset collection
collection = DatasetCollection()

# Create a new dataset
dataset = collection.create_dataset("my_dataset")

# Add samples to the dataset
dataset.add_sample(
    image_path="path/to/image.png",
    ground_truth="Example text",
    category="printed"
)
```

### Running OCR with Different Engines

```python
from ocr_poc.engine.base import OCREngineRegistry
from PIL import Image

# Load an image
image = Image.open("path/to/image.png")

# Get available OCR engines
engines = OCREngineRegistry.list_engines()
print(f"Available engines: {engines}")

# Create and use an OCR engine
engine = OCREngineRegistry.create_engine("TesseractOCR")
result = engine.process_image(image)
print(f"Recognized text: {result.text}")
print(f"Confidence: {result.confidence}")
```

### Evaluating OCR Performance

```python
from ocr_poc.evaluation.metrics import evaluate_result

# Evaluate OCR result against ground truth
metrics = evaluate_result(result, "Ground truth text")
print(f"Character accuracy: {metrics['character_accuracy']}")
print(f"Word accuracy: {metrics['word_accuracy']}")
print(f"Word error rate: {metrics['wer']}")
```

### Visualizing Results

```python
from ocr_poc.evaluation.visualizer import create_comparison_visualization

# Create a comparison visualization of multiple OCR results
create_comparison_visualization(
    image="path/to/image.png",
    ocr_results=[result1, result2, result3],
    ground_truth="Ground truth text",
    output_path="comparison.png"
)
```

## Adding a New OCR Engine

To add a new OCR engine, create a new class that inherits from `BaseOCREngine` and register it:

```python
from ocr_poc.engine.base import BaseOCREngine, OCREngineRegistry, OCRResult
import time

@OCREngineRegistry.register("MyCustomOCR")
class MyCustomOCREngine(BaseOCREngine):
    def process_image(self, image, **kwargs):
        # Load image if needed
        img = self.load_image(image)
        
        # Record start time
        start_time = time.time()
        
        # Your OCR implementation here
        # ...
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return result
        return OCRResult(
            text="Recognized text",
            confidence=0.95,
            bounding_boxes=[(10, 10, 100, 30)],
            engine_name=self.name,
            processing_time=processing_time
        )
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Tesseract OCR for providing a robust open-source OCR engine
- Various OCR dataset providers
