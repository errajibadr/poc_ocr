# OCR POC Scripts

This directory contains scripts for the OCR POC project.

## Dataset Download Scripts

### `download_synthtext.py`

This script downloads and processes the SynthText dataset for OCR testing.

#### Usage

```bash
# Download the full SynthText dataset
python scripts/download_synthtext.py

# Download only a sample of the dataset (faster for testing)
python scripts/download_synthtext.py --sample-only

# Download and create variations of the dataset images
python scripts/download_synthtext.py --sample-only --create-variations

# Specify the output directory
python scripts/download_synthtext.py --output-dir data/raw/my_datasets

# Create more variations per image
python scripts/download_synthtext.py --sample-only --create-variations --num-variations 5
```

#### Options

- `--output-dir`: Directory to download the dataset to (default: `data/raw`)
- `--sample-only`: Download only a small sample of the dataset
- `--create-variations`: Create variations of the dataset images
- `--num-variations`: Number of variations to create for each image (default: 2)

#### Notes

The SynthText dataset is a synthetic dataset of natural images with text rendered onto them. It's useful for training and testing OCR systems, as it provides a large number of samples with perfect ground truth.

When using the `--sample-only` option, the script downloads a much smaller version of the dataset, which is useful for testing the processing pipeline without waiting for the full dataset download.

The `--create-variations` option applies various transformations to the dataset images, such as adding noise, adjusting lighting, and applying geometric transformations. This helps test the robustness of OCR systems to different image qualities and variations.

## Running the Scripts

Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Run the scripts from the root directory of the project to ensure the imports work correctly:

```bash
# From the project root directory
python scripts/download_synthtext.py
``` 