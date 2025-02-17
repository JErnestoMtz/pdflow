from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Dict, Tuple, Optional
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pdflow.base_funcs import file_to_images, crop_boxes
import json
import fitz  # PyMuPDF

class SegmentationModel(ABC):
    @abstractmethod
    def labels(self) -> Dict[int, str]:
        """Return a dictionary of labels for detected objects."""
        pass
    
    @abstractmethod
    def segment(self, image: Image.Image) -> Optional[Dict[int, List[Tuple[float, float, float, float]]]]:
        """Return a dictionary of bounding boxes for detected objects with normalized coordinates."""
        pass

class OCRModel(ABC):
    @abstractmethod
    def extract_text(self, images: List[Image.Image]) -> str:
        """Extract text from image segments."""
        pass

class ExtractionMessage(ABC):
    @abstractmethod
    def __call__(self, fields: List[str], content: str) -> ModelMessage:
        pass

class DefaultExtractionMessage(ExtractionMessage):
    def __call__(self, fields: List[str], content: str) -> ModelMessage:
        """Create structured prompt for field extraction"""
        format_example = json.dumps({field: f"<{field}>" for field in fields}, indent=2)
        
        system_prompt = f"""You are a professional data extraction system. 
        Extract the following fields from the document: {', '.join(fields)}.
        Return ONLY valid JSON format with the extracted values. 
        Use null for missing fields.
        Example format:
        {format_example}"""
        
        user_prompt = f"DOCUMENT CONTENT:\n{content}"
        
        return ModelMessage(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

class ImagePreprocessor(ABC):
    @abstractmethod
    def preprocess_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Preprocess images for analysis."""
        pass

class DefaultImagePreprocessor(ImagePreprocessor):
    def preprocess_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Basic preprocessing that returns original images."""
        return images

class DocumentAnalyzer:
    def __init__(self, segmentation_model: SegmentationModel, ocr_model: OCRModel, language_model: Model):
        self.segmentation_model = segmentation_model
        self.ocr_model = ocr_model
        self.language_model = language_model

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

    def extract_fields(self, 
                       path: str, 
                       fields: List[str], 
                       extraction_message: ExtractionMessage = DefaultExtractionMessage(),
                       image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> Dict[str, Optional[str]]:
        """Extract fields from the document, using OCR only if necessary."""
        if path.endswith(".pdf") and self._pdf_contains_text(path):
            # If the PDF contains text, extract it directly
            text_extraction = self._extract_text_from_pdf(path)
        else:
            # If the PDF is scanned or not a PDF, use OCR
            images = file_to_images(path)
            preprocessed_images = image_processor.preprocess_images(images)
            text_extraction = '\n'.join([self.ocr_model.extract_text([image]) for image in preprocessed_images])
        
        # Generate and execute extraction prompt
        prompt = extraction_message(fields, text_extraction)
        response = self.language_model.request(prompt)
        
        # Parse and validate response
        try:
            extracted_data = json.loads(response.content)
            return {field: extracted_data.get(field, None) for field in fields}
        except json.JSONDecodeError:
            return {field: None for field in fields}
        
    def segment_document(self, path: str, 
                         image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> List[Dict[int, List[Tuple[float, float, float, float]]]]:
        """Segment the document into regions of interest."""
        # Convert file to images and preprocess
        images = file_to_images(path)
        preprocessed_images = image_processor.preprocess_images(images)
        segments = [self.segmentation_model.segment(image) for image in preprocessed_images]
        return segments

    def get_image_ids(self, path: str, 
                            class_ids: List[int],
                            image_processor: ImagePreprocessor = DefaultImagePreprocessor()) -> Optional[List[Image.Image]]:
        """Get cropped images for specified class IDs."""
        pages = file_to_images(path)
        segments = self.segment_document(path, image_processor=image_processor)
        images = []
        for i, page in enumerate(pages):
            for class_id in segments[i].keys():  # Corrected: Use .keys() method
                if class_id in class_ids:
                    images.extend(crop_boxes(page, segments[i][class_id]))  # Corrected: Pass page and boxes
        return images if images else None