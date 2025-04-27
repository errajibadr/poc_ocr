# OCR Test Datasets

This directory contains datasets for OCR (Optical Character Recognition) evaluation. The datasets are organized to facilitate testing and comparing different OCR methods.

## Dataset Structure

The datasets are organized as follows:

```
data/
├── raw/                 # Raw, unprocessed datasets
│   ├── SynthText/       # SynthText dataset
│   ├── HierText/        # HierText dataset
│   └── ...
├── processed/           # Processed datasets ready for OCR evaluation
│   ├── SynthText/       # Processed SynthText dataset
│   ├── HierText/        # Processed HierText dataset
│   └── ...
└── README.md            # This file
```

## Downloading Datasets

We provide several scripts to download and process OCR datasets. These scripts are located in the `src/ocr_poc/data/download_scripts` directory.

### Available Download Scripts

1. **SynthText Dataset**: `download_synthtext.py`
2. **HierText Dataset**: `download_hiertext.py`

### Installing Dependencies

Before downloading datasets, make sure you have installed the required dependencies:

```bash
pip install -e .
```

This will install the OCR POC package and all its dependencies in development mode.

### Using the Download Scripts

#### SynthText Dataset

[SynthText](https://github.com/ankush-me/SynthText) is a synthetic dataset with text rendered onto natural images. It's useful for training and testing OCR models.

To download and process the SynthText dataset:

```bash
# From the project root directory
python -m src.ocr_poc.data.download_scripts.download_synthtext

# Limit the number of samples (default is 1000)
python -m src.ocr_poc.data.download_scripts.download_synthtext --max-samples 500

# Specify a custom output directory
python -m src.ocr_poc.data.download_scripts.download_synthtext --output-dir data/custom_dir
```

**Note:** The SynthText dataset is large (>10GB), so downloading may take some time.

#### HierText Dataset

[HierText](https://github.com/google-research-datasets/hiertext) is a hierarchical text detection dataset with word, line, and paragraph annotations.

To download and process the HierText dataset:

```bash
# From the project root directory
python -m src.ocr_poc.data.download_scripts.download_hiertext

# Process the train split (default is validation)
python -m src.ocr_poc.data.download_scripts.download_hiertext --split train

# Limit the number of samples (default is 1000)
python -m src.ocr_poc.data.download_scripts.download_hiertext --max-samples 500

# Specify a custom output directory
python -m src.ocr_poc.data.download_scripts.download_hiertext --output-dir data/custom_dir

# Use images from a specific directory
python -m src.ocr_poc.data.download_scripts.download_hiertext --image-dir path/to/images
```

**Note:** The HierText script only downloads annotations by default. To download the actual images, follow the instructions printed by the script, which involve using Google Cloud Storage.

### Listing Available Datasets

To see a list of all available datasets in the registry:

```bash
python -m src.ocr_poc.data.download_datasets --list
```

This will show information about all datasets in the registry, including freely available datasets and those requiring registration.

### Downloading Multiple Datasets

To download multiple datasets at once:

```bash
python -m src.ocr_poc.data.download_datasets --datasets SynthText HierText
```

To download all freely available datasets:

```bash
python -m src.ocr_poc.data.download_datasets --download-free
```

## Using the Datasets

### Loading Datasets

You can load the downloaded datasets using the `DatasetCollection` class:

```python
from src.ocr_poc.data.dataset import DatasetCollection

# Load a dataset collection
collection = DatasetCollection(Path("data/raw"))

# Get a specific dataset
synthtext = collection.get_dataset("SynthText")

# Get a sample from the dataset
sample = synthtext.get_sample(0)

# Load the image for a sample
image = synthtext.load_image(0)

# Get the ground truth text for a sample
text = synthtext.get_ground_truth(0)
```

### Dataset Inspection

You can inspect and validate datasets using the `DatasetInspector` class:

```python
from src.ocr_poc.data.dataset_utils import DatasetInspector

# Create an inspector
inspector = DatasetInspector()

# Generate statistics for a dataset
stats = inspector.generate_statistics(synthtext)

# Validate a dataset
is_valid, issues = inspector.validate_dataset(synthtext)

# Visualize a sample
visualization_path = inspector.visualize_sample(synthtext, 0)
```

### Data Augmentation

You can create variations of dataset images using the `DataAugmentation` class:

```python
from src.ocr_poc.data.dataset_utils import DataAugmentation

# Create an augmenter
augmenter = DataAugmentation()

# Load an image
image = synthtext.load_image(0)

# Create variations of the image
variations = augmenter.create_variations(image, num_variations=3)

# Add noise to an image
noisy_image = augmenter.add_noise(image, noise_type="gaussian", intensity=0.1)

# Adjust lighting of an image
bright_image = augmenter.adjust_lighting(image, brightness=1.2, contrast=1.1)

# Apply geometric transformations to an image
transformed_image = augmenter.geometric_transform(image, rotation=5.0, scale=1.2)

# Apply blur to an image
blurred_image = augmenter.apply_blur(image, radius=2.0)

# Simulate JPEG compression artifacts
compressed_image = augmenter.apply_jpeg_compression(image, quality=70)
```

## Adding New Datasets

To add a new dataset:

1. Create a new download script in the `src/ocr_poc/data/download_scripts` directory
2. Use the `DatasetDownloader` and `FormatConverter` classes to download and process the dataset
3. Add the dataset information to the `DatasetRegistry` class in `src/ocr_poc/data/dataset_research.py`

## Troubleshooting

### Common Issues

- **Missing dependencies**: If you encounter errors about missing dependencies, install them with `pip install <dependency>`.
- **Download errors**: If downloads fail, check your internet connection and try again. Some datasets may require registration or special access.
- **Memory errors**: Processing large datasets may require substantial memory. Try limiting the number of samples with the `--max-samples` option.

### Getting Help

If you encounter any issues, please:

1. Check that all dependencies are installed correctly
2. Ensure you're running the scripts from the project root directory
3. Try with a smaller number of samples to see if the issue is memory-related

## References

For more information about the available datasets, see:

- SynthText: [GitHub](https://github.com/ankush-me/SynthText) | [Paper](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)
- HierText: [GitHub](https://github.com/google-research-datasets/hiertext) | [Paper](https://arxiv.org/abs/2203.15143) 