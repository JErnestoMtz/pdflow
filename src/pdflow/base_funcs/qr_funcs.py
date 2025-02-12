from pyzbar.pyzbar import decode
from PIL import Image
import numpy as np
from typing import Optional, List
import cv2
def read_qr_code(image: Image.Image) -> Optional[str]:
    """
    Reads a QR code from a Pillow image and returns its content as a URL.
    
    Args:
        image (PIL.Image): Input Pillow image containing a QR code.
    
    Returns:
        str: Decoded URL from QR code, or None if not found.
    """
    decoded_objects = decode(image)
    
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8')
    return None


def detect_and_crop_qr(image: Image.Image) -> Optional[List[Image.Image]]:
    """
    Detects QR codes in a Pillow image and returns cropped QR code images.
    
    Args:
        image (PIL.Image): Input Pillow image.
    
    Returns:
        list: List of cropped Pillow Image objects (one per detected QR code).
    """
    # Convert Pillow image to OpenCV format (numpy array)
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Pillow uses RGB, OpenCV uses BGR
    
    decoded_objects = decode(img_cv)
    cropped_images = []
    
    for obj in decoded_objects:
        # Get bounding box coordinates
        points = obj.polygon
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Crop the image
        cropped_cv = img_cv[y_min:y_max, x_min:x_max]
        
        # Convert back to Pillow Image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
        cropped_images.append(cropped_pil)
    
    return cropped_images

