from typing import List
import fitz  # PyMuPDF
import cv2
from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np



def crop_boxes(image: Image.Image, boxes: List[List[float]]) -> List[Image.Image]:
    """
    Crops multiple regions from an image using normalized coordinates (xyxyn format).
    
    Args:
        image: Input PIL Image
        boxes: List of bounding boxes in normalized xyxyn format [x1, y1, x2, y2]
               where coordinates are between 0 and 1
    
    Returns:
        List of cropped PIL Images
    """
    cropped_images = []
    img_width, img_height = image.size
    
    for box in boxes:
        # Convert normalized coordinates to absolute pixel values
        x1 = box[0] * img_width
        y1 = box[1] * img_height
        x2 = box[2] * img_width
        y2 = box[3] * img_height
        
        # Ensure coordinates are within image bounds
        left = max(0, min(x1, img_width))
        upper = max(0, min(y1, img_height))
        right = max(left, min(x2, img_width))
        lower = max(upper, min(y2, img_height))
        
        # Crop and add to results
        cropped_img = image.crop((left, upper, right, lower))
        cropped_images.append(cropped_img)
    
    return cropped_images