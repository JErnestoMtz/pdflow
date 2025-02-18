# PDFlow

PDFlow is a Python library for document processing and analysis, featuring QR code extraction, document segmentation, and intelligent text extraction. It provides a flexible architecture that can work with different ML models and services.

## Features

- **Document Processing**
  - Convert PDFs and multi-page images to processable images
  - Support for PDF, TIFF, GIF, PNG, JPG, and other image formats
  - High-quality conversion (300 DPI for PDFs)

- **QR Code Operations**
  - Extract QR codes from documents
  - Detect and crop QR code regions
  - Decode QR code contents
  - Support for multi-page documents

- **Document Segmentation**
  - Flexible segmentation model integration
  - Support for custom segmentation models
  - Built-in adapters for popular ML models
  - Region-based document analysis

- **Intelligent Text Extraction**
  - Modular extraction pipeline
  - Support for various OCR services
  - LLM-powered field extraction
  - Structured data output in JSON format

## Installation

You can install PDFlow directly from GitHub using pip or uv:

```bash
# Using uv
uv add git+https://github.com/JErnestoMtz/pdflow

# Using pip
pip install git+https://github.com/JErnestoMtz/pdflow
```

For development installation:

```bash
# Clone the repository
git clone https://github.com/JErnestoMtz/pdflow.git
cd pdflow

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install in development mode with extra dependencies
python -m pip install -e ".[dev]"
```

## Quick Start

### Basic QR Code Extraction

```python
from pdflow import extract_qrs_decoded

# Extract and decode QR codes from a document
qr_codes = extract_qrs_decoded("document.pdf")
print("QR Code Sample:", qr_codes[0] if qr_codes else "No QR codes found")
```

### Document Analysis Setup

```python
from pdflow.document_analyzer import DocumentAnalyzer

# Initialize with your preferred models
document_analyzer = DocumentAnalyzer(
    segmentation_model=your_segmentation_model,  # Any model implementing SegmentationModel interface
    text_extraction_model=your_extraction_model  # Any model implementing TextExtractionModel interface
)
```

### Document Analysis Operations

#### Field Extraction
```python
# Extract specific fields from a document
fields = ['field1', 'field2', 'field3']
extracted = await document_analyzer.extract_fields('document.pdf', fields)
print(extracted)
```

#### Image Segmentation
```python
# Get images by label
images = document_analyzer.get_by_label('document.pdf', 'Picture')

# Get images by class ID
images = document_analyzer.get_by_id('document.pdf', class_id)

# View available labels
labels = segmentation_model.labels()
```

## Architecture

PDFlow uses a modular architecture with the following key interfaces:

- **SegmentationModel**: Interface for document segmentation models
- **TextExtractionModel**: Interface for text extraction models
- **OCRModel**: Interface for OCR operations
- **ImagePreprocessor**: Interface for image preprocessing operations

You can implement these interfaces to use your preferred models or services.

## Response Format

The field extraction returns structured JSON data:

```json
{
  "field1": "value",
  "field2": [
    {
      "value": "row1",
      "details": {
        "additional_info": "metadata"
      }
    }
  ]
}
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/JErnestoMtz/pdflow.git
cd pdflow
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install development dependencies:
```bash
python -m pip install -e ".[dev]"
```

## Documentation

To build the documentation:

```bash
cd docs
sphinx-apidoc -o source/api ../src/pdflow
make html
```

The documentation will be available in `docs/build/html/index.html`.

## Running Tests

```bash
pytest
```

## License

[MIT License](LICENSE)