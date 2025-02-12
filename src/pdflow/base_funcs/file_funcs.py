import fitz  # PyMuPDF
from PIL import Image
import os

def file_to_images(file_path: str) -> list[Image.Image]:
    """
    Converts a file (PDF or image) to a list of Pillow Images.
    Handles multi-page files (PDF, TIFF, GIF) and single-page files (PNG, JPG).
    
    Args:
        file_path: Path to the input file (PDF, PNG, JPG, TIFF, GIF, etc.)
        
    Returns:
        List of PIL Images (one for each page/frame)
    """
    images = []
    
    if os.path.splitext(file_path)[1].lower() == '.pdf':
        # Process PDF files with PyMuPDF
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            matrix = fitz.Matrix(300/72, 300/72)  # 300 DPI
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
        pdf_document.close()
    else:
        # Process image files with Pillow
        with Image.open(file_path) as img:
            while True:
                try:
                    images.append(img.copy())
                    # Try to seek to next frame/page
                    img.seek(img.tell() + 1)
                except EOFError:
                    break  # End of frames/pages
                except ValueError:
                    break  # Some formats might raise ValueError instead
    
    return images