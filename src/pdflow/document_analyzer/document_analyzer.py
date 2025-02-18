from typing import List, Dict, Tuple, Optional, Union, Any
from PIL import Image
from pdflow.base_funcs import file_to_images, crop_boxes
import fitz  # PyMuPDF
import asyncio

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

    def _pdf_contains_text(self, path: str) -> bool:
        """Check if the PDF contains actual text (not scanned)."""
        doc = fitz.open(path)
        for page in doc:
            if page.get_text():  # If any page has text, return True
                return True
        return False

    def _extract_text_from_pdf(self, path: str) -> str:
        """Extract text directly from a PDF with actual text."""
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    async def extract_fields(self, 
                           path: str, 
                           fields: List[str],
                           image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> Dict[str, Optional[str]]:
        """Extract fields from the document, using OCR only if necessary."""
        if path.endswith(".pdf") and self._pdf_contains_text(path):
            # If the PDF contains text, extract it directly
            text_extraction = self._extract_text_from_pdf(path)
        else:
            # If the PDF is scanned or not a PDF, use OCR
            images = file_to_images(path)
            preprocessed_images = image_processor.preprocess_images(images)
            
            # Extract text from all images concurrently
            text_parts = await asyncio.gather(
                *[self.extraction_model.extract_text([image]) for image in preprocessed_images]
            )
            text_extraction = '\n'.join(text_parts)
        
        # Extract fields from text
        return await self.extraction_model.extract_fields(text_extraction, fields)

    def segment_document(self, path: str, 
                        image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> List[Dict[int, List[Tuple[float, float, float, float]]]]:
        """Segment the document into regions of interest."""
        # Convert file to images and preprocess
        images = file_to_images(path)
        preprocessed_images = image_processor.preprocess_images(images)
        segments = [self.segmentation_model.segment(image) for image in preprocessed_images]
        return segments

    def get_by_id(self, path: str, 
                  class_ids: Union[int, List[int]],
                  image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> Optional[List[Image.Image]]:
        """Get cropped images for specified class ID(s).
        
        Args:
            path: Path to the document
            class_ids: Single class ID or list of class IDs to extract
            image_processor: Optional image preprocessor
            
        Returns:
            List of cropped images or None if no matches found
        """
        if isinstance(class_ids, int):
            class_ids = [class_ids]
            
        pages = file_to_images(path)
        segments = self.segment_document(path, image_processor=image_processor)
        images = []
        for i, page in enumerate(pages):
            for class_id in segments[i].keys():
                if class_id in class_ids:
                    images.extend(crop_boxes(page, segments[i][class_id]))
        return images if images else None

    def get_by_label(self, path: str, 
                     labels: Union[str, List[str]],
                     image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> Optional[List[Image.Image]]:
        """Get cropped images for specified label(s).
        
        Args:
            path: Path to the document
            labels: Single label or list of labels to extract (e.g., 'signature', 'table')
            image_processor: Optional image preprocessor
            
        Returns:
            List of cropped images or None if no matches found
        """
        if isinstance(labels, str):
            labels = [labels]
            
        # Get the id-to-label mapping and create a reverse mapping
        label_mapping = self.segmentation_model.labels()
        label_to_id = {label: id for id, label in label_mapping.items()}
        
        # Convert labels to class IDs using the reverse mapping
        class_ids = [label_to_id[label] for label in labels if label in label_to_id]
        return self.get_by_id(path, class_ids, image_processor)
    
    