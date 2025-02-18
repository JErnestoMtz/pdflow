from PIL import Image
from typing import List, Optional
from .base_funcs import file_to_images, read_qr_code, detect_and_crop_qr

def extract_qrs(file_path: str) -> List[Optional[Image.Image]]:
    images = file_to_images(file_path)
    qrs = []
    for image in images:
        cropped_qrs = detect_and_crop_qr(image)  # This returns a list
        qrs.extend(cropped_qrs)  # Flatten the list
    return qrs

def extract_qrs_decoded(file_path: str) -> List[Optional[str]]:
    qrs = extract_qrs(file_path)
    decoded_qrs = [read_qr_code(qr) for qr in qrs if qr is not None]
    return decoded_qrs



