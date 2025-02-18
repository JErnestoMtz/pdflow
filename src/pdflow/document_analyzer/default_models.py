from typing import List, Dict, Optional, Union, Tuple, Any
from PIL import Image
import io
import json
import asyncio
from functools import wraps
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from azure.ai.formrecognizer import DocumentAnalysisClient
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .base import ExtractionMessage, ImagePreprocessor, TextExtractionModel, OCRModel, SegmentationModel

# Default model parameters optimized for extraction tasks
DEFAULT_MODEL_SETTINGS = {
    "temperature": 0.0,  # Zero temperature for deterministic outputs
    "top_p": 1.0,       # No nucleus sampling
    "max_tokens": 1000,  # Reasonable limit for JSON responses
}

class DefaultExtractionMessage(ExtractionMessage):
    def __call__(self, fields: List[str], content: Optional[str] = None) -> List[Dict[str, str]]:
        """Create structured prompt for field extraction"""
        format_example = {
            "single_value_field": "<value>",
            "list_or_table_field": [
                {"value": "<value1>", "details": {"percentage": "60%", "date": "2024-01-01"}},
                {"value": "<value2>", "details": {"percentage": "40%", "date": "2024-01-02"}}
            ]
        }
        
        system_prompt = f"""You are a professional data extraction system. 
        Extract the following fields from the document: {', '.join(fields)}.
        Return ONLY valid JSON format with the extracted values. 
        Use null for missing fields.
        
        Special instructions:
        1. If a field contains multiple values (like in a table or list), return it as a list of objects
        2. For table data, include relevant details like percentages, dates in a 'details' object
        3. For simple single values, return them directly
        
        Example format:
        {json.dumps(format_example, indent=2)}"""
        
        user_prompt = (
            f"DOCUMENT CONTENT:\n{content}" if content is not None 
            else "Please provide a JSON template with null values for the requested fields."
        )
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

class DefaultImagePreprocessor(ImagePreprocessor):
    def preprocess_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Basic preprocessing that returns original images."""
        return images

class MultiModalModel(TextExtractionModel):
    def __init__(self, openai_client: Any):
        """Initialize with OpenAI client for multimodal capabilities."""
        model = OpenAIModel('gpt-4', openai_client=openai_client)
        self.agent = Agent(model)

    async def extract_text(self, images: List[Image.Image]) -> str:
        # Implementation for multimodal OCR
        pass

    async def extract_fields(self, text: str, fields: List[str]) -> Dict[str, Optional[str]]:
        # Implementation for field extraction using the agent
        pass

class TwoStageExtractor(TextExtractionModel):
    """A two-stage extraction model that separates OCR and field extraction.
    
    This model uses:
    1. An OCR model (or Azure Document Intelligence) to extract text.
    2. A pre-configured PydanticAI Agent to extract structured fields from the text.
    """
    
    def __init__(self, 
                 ocr_model: Union[OCRModel, DocumentAnalysisClient],
                 agent: Agent,
                 model_settings: Optional[Dict[str, Any]] = None):
        """Initialize with a separate OCR model and a pre-configured PydanticAI agent.
        
        Args:
            ocr_model: Either an OCRModel implementation or an Azure DocumentAnalysisClient.
            agent: A pre-configured PydanticAI Agent instance for field extraction.
            model_settings: Settings for the language model (e.g., temperature, max_tokens).
        """
        self.ocr_model = ocr_model
        self.agent = agent
        self.extraction_message = DefaultExtractionMessage()
        self.model_settings = model_settings or DEFAULT_MODEL_SETTINGS.copy()

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'PNG')
        return img_byte_arr.getvalue()

    def extract_text(self, images: List[Image.Image]) -> str:
        """Extract text using either an OCRModel or the Azure client."""
        if isinstance(self.ocr_model, DocumentAnalysisClient):
            # Use Azure client directly
            all_text = []
            for image in images:
                image_bytes = self._image_to_bytes(image)
                result = self.ocr_model.begin_analyze_document(
                    "prebuilt-document",
                    document=image_bytes
                ).result()
                
                page_text = []
                for page in result.pages:
                    for line in page.lines:
                        page_text.append(line.content)
                
                all_text.append(" ".join(page_text))
            return "\n".join(all_text)
        else:
            # Use OCRModel implementation
            return self.ocr_model.extract_text(images)

    async def extract_fields(self, text: str, fields: List[str]) -> Dict[str, Optional[Union[str, List[Dict[str, Any]]]]]:
        """Extract structured fields from text using the PydanticAI Agent."""
        messages = self.extraction_message(fields, text)
        print("\nPrompt Messages:")
        for msg in messages:
            print(f"\n{msg['role'].upper()} MESSAGE:")
            print(msg['content'])
        
        try:
            # Combine system and user messages into a single prompt
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            
            # Use the agent to run the extraction, applying the model settings if provided
            result = await self.agent.run(prompt, model_settings=self.model_settings)
            print("\nModel Response:", result.data)
            
            # Parse the response
            if isinstance(result.data, dict):
                extracted_data = result.data
            else:
                # Clean up the response if it contains markdown code blocks
                response_text = result.data
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                try:
                    extracted_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {str(e)}")
                    print(f"Attempted to parse: {response_text}")
                    return {field: None for field in fields}
            
            # Process each field, handling both single values and lists
            processed_data = {}
            for field in fields:
                value = extracted_data.get(field)
                if isinstance(value, list):
                    # For list values, ensure each item has a proper structure
                    processed_items = []
                    for item in value:
                        if isinstance(item, dict):
                            processed_items.append(item)
                        else:
                            # If it's a simple value, wrap it in a dict
                            processed_items.append({"value": item})
                    processed_data[field] = processed_items
                else:
                    processed_data[field] = value
                    
            return processed_data
        except Exception as e:
            print(f"Error extracting fields: {str(e)}")
            return {field: None for field in fields}

if YOLO_AVAILABLE:
    class YOLOSegmentationAdapter(SegmentationModel):
        """Adapter to make YOLO models compatible with SegmentationModel interface."""
        def __init__(self, model: YOLO):
            self.model = model
            self._labels = None

        def labels(self) -> Dict[int, str]:
            """Return a dictionary of labels for detected objects."""
            if self._labels is None:
                # Cache the labels
                self._labels = {i: name for i, name in enumerate(self.model.names.values())}
            return self._labels

        def segment(self, image: Image.Image) -> Optional[Dict[int, List[Tuple[float, float, float, float]]]]:
            """Return a dictionary of bounding boxes for detected objects."""
            results = self.model(image, verbose=False)
            if not results or not results[0].boxes:
                return None

            boxes_by_class = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                # YOLO boxes are already normalized
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                if class_id not in boxes_by_class:
                    boxes_by_class[class_id] = []
                boxes_by_class[class_id].append((x1, y1, x2, y2))

            return boxes_by_class 