from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Dict, Tuple, Optional

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
    async def extract_text(self, images: List[Image.Image]) -> str:
        """Extract text from image segments."""
        pass

class ExtractionMessage(ABC):
    @abstractmethod
    def __call__(self, fields: List[str], content: Optional[str] = None) -> List[Dict[str, str]]:
        """Create a message for field extraction.
        
        Args:
            fields: List of fields to extract
            content: Optional document content. If None, uses a default prompt
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        pass

class ImagePreprocessor(ABC):
    @abstractmethod
    def preprocess_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Preprocess images for analysis."""
        pass

class TextExtractionModel(ABC):
    @abstractmethod
    async def extract_text(self, images: List[Image.Image]) -> str:
        """Extract text from image segments."""
        pass

    @abstractmethod
    async def extract_fields(self, text: str, fields: List[str]) -> Dict[str, Optional[str]]:
        """Extract structured fields from text."""
        pass 