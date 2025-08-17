# Sri Lankan Driving License OCR Pipeline

A machine learning pipeline that extracts vehicle class information and validity dates from Sri Lankan driving licenses using computer vision and OCR techniques.

## Overview

This project automatically extracts information from the rear page of Sri Lankan driving licenses, specifically:

1. **Vehicle Classes**: Identifies and extracts permitted vehicle categories (e.g., A1, B, C, D, etc.)
2. **License Validity Period**: Extracts start date and expiry date for each vehicle category

The solution is built using open-source tools and libraries without relying on external cloud-based OCR services.

## Features

- **Intelligent Image Orientation Correction**: Automatically determines the correct orientation of license images (keyword matching using Tesseract)
- **Table Region Detection**: Precisely locates and extracts the tabular area containing vehicle class information (morphological operations and contour analysis)
- **Multi-Engine OCR**: Leverages PaddleOCR for robust text recognition
- **Advanced Text Parsing**: Maps OCR output to known vehicle classes and dates
- **Date Validation and Pairing**: Intelligently matches start and expiry dates for vehicle categories
- **Batch Processing**: Handles multiple images with progress tracking
- **Structured Output**: Results in easy-to-use tabular format (CSV, DataFrame)

### Supported Vehicle Classes

A1, A, B1, B, C1, C, CE, D1, D, DE, G1, G, J

## Project Structure

```text
driving_license-ocr/
│
├── data/
│   ├── raw/                # Raw license images
│   └── output/             # Output CSVs and results
│
├── notebooks/
│   ├── driving_license_ocr.ipynb          # Interactive jupyter notebook for both demonstration and testing purposes. (single image & batch processing)
│   ├── driving_license_ocr_simple.ipynb   # A simpler jupyter notebook without the interactive feature. (single image & batch processing)
│   └── exploratory/                       # A dedicated folder for experimentation, testing, and learning.
│       └── ...
│
├── src/
│   ├── data/               # Data processing
│   ├── models/             # OCR model wrappers
│   ├── extraction/         # Text extraction logic
│   ├── pipeline/           # Orchestration pipeline
│   └── utils/              # Utility functions
│
├── config/                 # Configuration files (e.g., config.yaml)
├── requirements.txt        # Python dependencies
└── main.py                 # Command-line entry point
```

## Jupyter Notebooks (Recommended to Use)

The notebooks directory contains Jupyter notebooks that visually demonstrate how the driving license OCR pipeline works. These notebooks are ideal for both understanding the step-by-step logic of the project and for debugging. 

* driving_license_ocr.ipynb: A detailed, step-by-step notebook with interactive plots for single image and batch processing. (sample outputs provided)

* driving_license_ocr_simple.ipynb: A streamlined version without interactive features, supporting both single and batch processing.

The exploratory folder includes past versions of pipelines, EDA, preprocessing, ocr techniques, and notebooks used for learning computer vision.

## Initialization

### Prerequisites

- Python 3.11.9 or higher
- Tesseract OCR installed on your system

### Step 1: Clone the repository / Unzip the downloaded folder

#### For cloned repo
```bash
git clone https://github.com/yourusername/driving-license-ocr.git
cd driving-license-ocr
```

#### For downloaded and extracted/unzipped folder
```bash
cd path\to\your\folder\driving-license-ocr-main
```

### Step 2: Create a virtual environment (strongly recommended)

```bash
python -m venv venv
```

```bash
# On Windows
venv\Scripts\activate
```

```bash
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Tesseract OCR (Skip if already installed)

#### Windows:
1. Download the installer from [UB Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and follow the instructions
3. Make sure the installation path is added to your system PATH

#### Linux:
```bash
sudo apt update
sudo apt install tesseract-ocr
```

#### macOS:
```bash
brew install tesseract
```

### Step 4: Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Process a single image:

```bash
python main.py --input data/raw/image_name.jpg --output data/output/image_name_result.csv
```

Process all images in a directory:

```bash
python main.py --input data/raw/ --output data/output/batch_results.csv --batch
```

### Python API

```python
from src.pipeline.processor import process_single_image
from src.pipeline.batch_processor import process_all_images
from src.data.data_loader import find_image_files

# Process a single image
result_df = process_single_image("path/to/image.jpg", verbose=True)
print(result_df)

# Process multiple images
image_paths = find_image_files("path/to/image/folder")
all_results_df = process_all_images(image_paths)
all_results_df.to_csv("output_results.csv", index=False)
```

## Configuration

The system can be configured through `config/config.yaml`, which allows you to:

- Set the OCR engines and their parameters
- Configure image preprocessing options
- Adjust date validation criteria
- Set output formats and paths

## Troubleshooting

### Tesseract Not Found Error

If you encounter a "TesseractNotFoundError" when running the application, it means the system cannot find the Tesseract OCR executable. To fix this:

1. **Verify Tesseract Installation**: Make sure Tesseract OCR is properly installed on your system.

2. **Specify the Tesseract Path**:
   
   Run the script with the `--tesseract-path` argument:
   ```bash
   python main.py --input data/raw/sample.jpg --output data/output/results.csv --tesseract-path "C:\Program Files\Tesseract-OCR\tesseract.exe"


### License
[No License]

This project is provided for demonstration purposes only. All rights are reserved, and no license is granted for commercial or public use.

## Author 
**Bhagya Dissanayake**

* GitHub: **bhagya-d36**
