from typing import List, Dict, Tuple, Optional, Union, Any, BinaryIO
from PIL import Image
from pdflow.base_funcs import file_to_images, crop_boxes
import fitz  # PyMuPDF
import asyncio
import os

from .base import SegmentationModel, TextExtractionModel, ImagePreprocessor
from .default_models import DefaultImagePreprocessor, DEFAULT_MODEL_SETTINGS

class DocumentAnalyzer:
    def __init__(self, 
                 segmentation_model: SegmentationModel, 
                 extraction_model: TextExtractionModel,
                 model_settings: Optional[Dict[str, Any]] = None):
        """Initialize DocumentAnalyzer.
        
        Args:
            segmentation_model: Model for segmenting document regions
            extraction_model: Model for text extraction and field extraction
            model_settings: Settings for the language model (temperature, etc.)
        """
        self.segmentation_model = segmentation_model
        self.extraction_model = extraction_model
        
        # If the extraction model supports model parameters, update them
        if hasattr(extraction_model, 'model_settings'):
            extraction_model.model_settings = model_settings or DEFAULT_MODEL_SETTINGS

    def _pdf_contains_text(self, file_input: Union[str, BinaryIO, bytes]) -> bool:
        """Check if the PDF contains actual text (not scanned)."""
        if isinstance(file_input, str):
            doc = fitz.open(file_input)
        elif isinstance(file_input, bytes):
            doc = fitz.open(stream=file_input, filetype="pdf")
        else:
            file_input.seek(0)
            doc = fitz.open(stream=file_input.read(), filetype="pdf")
            
        for page in doc:
            if page.get_text():  # If any page has text, return True
                doc.close()
                return True
        doc.close()
        return False

    def _extract_text_from_pdf(self, file_input: Union[str, BinaryIO, bytes]) -> str:
        """Extract text directly from a PDF with actual text."""
        if isinstance(file_input, str):
            doc = fitz.open(file_input)
        elif isinstance(file_input, bytes):
            doc = fitz.open(stream=file_input, filetype="pdf")
        else:
            file_input.seek(0)
            doc = fitz.open(stream=file_input.read(), filetype="pdf")
            
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def _read_text_file(self, file_input: Union[str, BinaryIO, bytes]) -> str:
        """Read text directly from text-based files."""
        if isinstance(file_input, str):
            with open(file_input, 'r') as f:
                return f.read()
        elif isinstance(file_input, bytes):
            return file_input.decode('utf-8')
        else:
            file_input.seek(0)
            return file_input.read().decode('utf-8')

    def _get_file_type(self, file_input: Union[str, BinaryIO, bytes]) -> str:
        """Determine the type of file based on content or extension."""
        if isinstance(file_input, str):
            ext = os.path.splitext(file_input)[1].lower()
            return ext[1:] if ext else 'unknown'
        elif isinstance(file_input, bytes):
            if file_input.startswith(b'%PDF'):
                return 'pdf'
            # Try to decode as text
            try:
                file_input.decode('utf-8')
                return 'text'
            except UnicodeDecodeError:
                return 'binary'
        else:
            file_input.seek(0)
            header = file_input.read(4)
            file_input.seek(0)
            if header.startswith(b'%PDF'):
                return 'pdf'
            # Try to decode as text
            try:
                file_input.seek(0)
                file_input.read().decode('utf-8')
                return 'text'
            except UnicodeDecodeError:
                return 'binary'
            
    async def extract_fields_from_text(self, text: str, fields: List[str]) -> Dict[str, Optional[str]]:
        """Extract fields directly from a text string."""
        return await self.extraction_model.extract_fields(text, fields)

    async def extract_fields(self, 
                           file_input: Union[str, BinaryIO, bytes], 
                           fields: List[str],
                           image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> Dict[str, Optional[str]]:
        """Extract fields from the document.
        
        Args:
            file_input: File input as path string, file object, or bytes
            fields: List of fields to extract
            image_processor: Optional image preprocessor for image-based documents
            
        Returns:
            Dictionary of extracted fields
        """
        file_type = self._get_file_type(file_input)
        
        if file_type == 'pdf' and self._pdf_contains_text(file_input):
            # If the PDF contains text, extract it directly
            text_extraction = self._extract_text_from_pdf(file_input)
        elif file_type in ['txt', 'csv', 'text']:
            # For text-based files, read directly
            text_extraction = self._read_text_file(file_input)
        else:
            # For images, scanned PDFs, or other binary files, use OCR
            images = file_to_images(file_input)
            preprocessed_images = image_processor.preprocess_images(images)
            
            # Extract text from all images concurrently
            text_parts = await asyncio.gather(
                *[self.extraction_model.extract_text([image]) for image in preprocessed_images]
            )
            text_extraction = '\n'.join(text_parts)
        
        # Extract fields from text
        return await self.extraction_model.extract_fields(text_extraction, fields)

    def segment_document(self, 
                        file_input: Union[str, BinaryIO, bytes], 
                        image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> List[Dict[int, List[Tuple[float, float, float, float]]]]:
        """Segment the document into regions of interest."""
        # Convert file to images and preprocess
        images = file_to_images(file_input)
        preprocessed_images = image_processor.preprocess_images(images)
        segments = [self.segmentation_model.segment(image) for image in preprocessed_images]
        return segments

    def get_by_id(self, 
                  file_input: Union[str, BinaryIO, bytes], 
                  class_ids: Union[int, List[int]],
                  image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> Optional[List[Image.Image]]:
        """Get cropped images for specified class ID(s)."""
        if isinstance(class_ids, int):
            class_ids = [class_ids]
            
        pages = file_to_images(file_input)
        segments = self.segment_document(file_input, image_processor=image_processor)
        images = []
        for i, page in enumerate(pages):
            for class_id in segments[i].keys():
                if class_id in class_ids:
                    images.extend(crop_boxes(page, segments[i][class_id]))
        return images if images else None

    def get_by_label(self, 
                     file_input: Union[str, BinaryIO, bytes], 
                     labels: Union[str, List[str]],
                     image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> Optional[List[Image.Image]]:
        """Get cropped images for specified label(s)."""
        if isinstance(labels, str):
            labels = [labels]
            
        # Get the id-to-label mapping and create a reverse mapping
        label_mapping = self.segmentation_model.labels()
        label_to_id = {label: id for id, label in label_mapping.items()}
        
        # Convert labels to class IDs using the reverse mapping
        class_ids = [label_to_id[label] for label in labels if label in label_to_id]
        return self.get_by_id(file_input, class_ids, image_processor)
    
    