# PDFlow

PDFlow is a Python library for document processing and analysis, featuring QR code extraction, document segmentation, and multi-format file handling.

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
  - YOLO-based document segmentation
  - Extensible model architecture
  - Custom model support
  - Pre-trained models included

## Installation

```bash
uv pip install pdflow
```

## Quick Start

```python
from pdflow import extract_qrs_decoded, get_segmentation_model

# Extract and decode QR codes from a document
qr_contents = extract_qrs_decoded("document.pdf")
print(f"Found QR codes: {qr_contents}")

# Perform document segmentation
model = get_segmentation_model()  # Uses default YOLO model
results = model.predict("document.jpg")
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdflow.git
cd pdflow
```

2. Create a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install development dependencies:
```bash
uv pip install -e ".[dev]"
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

[Add your license information here]