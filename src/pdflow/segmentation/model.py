from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, Dict
from urllib.request import urlretrieve
import hashlib

from ultralytics import YOLO

class SegmentationModelProtocol(Protocol):
    """Protocol defining the interface for segmentation models."""
    def predict(self, image: Any) -> Any: ...

class BaseSegmentationModel(ABC):
    """Abstract base class for segmentation models."""
    class_dict: Dict[int, str]
    @abstractmethod
    def predict(self, image: Any) -> Any:
        """Predict segmentation for an image."""
        pass


class YOLOSegmentation(BaseSegmentationModel):
    """Default YOLO-based segmentation model."""
    
    def __init__(self) -> None:
        # Get the absolute path to the model file in the package
        pkg_path = resources.files('pdflow')
        self.model_path = pkg_path / "segmentation" / "models" / "yolov11l_best.pt"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.class_dict = self.model.names

    def predict(self, image: Any) -> Any:
        return self.model(image)

def get_segmentation_model(model: str | BaseSegmentationModel | None = None) -> SegmentationModelProtocol:
    """
    Get a segmentation model instance.
    
    Args:
        model: Can be either:
            - None: Returns default YOLO model
            - str: Identifier for a specific model type
            - BaseSegmentationModel: Custom model instance
    
    Returns:
        A segmentation model instance
    """
    if model is None:
        return YOLOSegmentation()
    
    if isinstance(model, BaseSegmentationModel):
        return model
    
    if isinstance(model, str):
        # Add more model types here as needed
        models = {
            "yolo": YOLOSegmentation,
            # "other_model": OtherModelClass,
        }
        if model not in models:
            raise ValueError(f"Unknown model type: {model}. Available models: {list(models.keys())}")
        return models[model]()
    