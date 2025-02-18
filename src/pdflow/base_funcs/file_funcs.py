import fitz  # PyMuPDF
from PIL import Image
import os
from typing import Union, BinaryIO, List
import io

def file_to_images(file_input: Union[str, BinaryIO, bytes]) -> List[Image.Image]:
    """
    Converts a file (PDF or image) to a list of Pillow Images.
    Handles multi-page files (PDF, TIFF, GIF) and single-page files (PNG, JPG).
    Accepts a file path, a file-like object opened in binary mode (rb), or bytes.

    Args:
        file_input: Path to the input file (PDF, PNG, JPG, TIFF, GIF, etc.), 
                   a file object opened in binary mode, or bytes content.

    Returns:
        List of PIL Images (one for each page/frame)
    """
    images = []
    
    # Handle bytes input by converting to BytesIO
    if isinstance(file_input, bytes):
        file_input = io.BytesIO(file_input)
    
    # Check if file_input is a string (file path) or a file-like object
    if isinstance(file_input, str):
        ext = os.path.splitext(file_input)[1].lower()
        if ext == '.pdf':
            # Process PDF files with PyMuPDF
            pdf_document = fitz.open(file_input)
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                matrix = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)
            pdf_document.close()
        else:
            # Process image files with Pillow
            with Image.open(file_input) as img:
                while True:
                    try:
                        images.append(img.copy())
                        img.seek(img.tell() + 1)
                    except (EOFError, ValueError):
                        break
    else:
        # file_input is a file-like object (opened in binary mode)
        file_input.seek(0)
        header = file_input.read(4)
        file_input.seek(0)
        if header.startswith(b"%PDF"):
            # Process PDF from binary stream
            file_bytes = file_input.read()
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                matrix = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)
            pdf_document.close()
        else:
            # Process image file from binary stream using Pillow
            with Image.open(file_input) as img:
                while True:
                    try:
                        images.append(img.copy())
                        img.seek(img.tell() + 1)
                    except (EOFError, ValueError):
                        break

    return images