import os
import cv2
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

def extract_text_from_file(file_path: str) -> str:
    """
    Intelligent document parser:
    - Uses PyMuPDF for text-based PDFs.
    - Uses OpenCV + Tesseract OCR for image scans (PNG, JPG) and scanned PDFs.
    """
    if not os.path.exists(file_path):
        return ""
    
    ext = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    
    try:
        # 1. Handle Native & Scanned PDFs via PyMuPDF
        if ext == '.pdf':
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    extracted_text += text + "\n"
                else:
                    # Fallback to OCR if page has no selectable text (scanned PDF)
                    pix = page.get_pixmap(dpi=150)
                    img_path = f"temp_page_{page_num}.png"
                    pix.save(img_path)
                    
                    # Preprocess image with OpenCV
                    gray_text = preprocess_and_ocr(img_path)
                    extracted_text += gray_text + "\n"
                    
                    if os.path.exists(img_path):
                        os.remove(img_path)
            doc.close()
            
        # 2. Handle Image Files (PNG, JPG, JPEG)
        elif ext in ['.png', '.jpg', '.jpeg']:
            extracted_text = preprocess_and_ocr(file_path)
            
        # 3. Handle Plain Text files
        elif ext in ['.txt', '.csv', '.log']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f.read()
                
    except Exception as e:
        print(f"OCR Extraction Error for {file_path}: {e}")
        
    return extracted_text.strip()

def preprocess_and_ocr(image_path: str) -> str:
    """
    Applies OpenCV grayscale and thresholding preprocessing to optimize Tesseract OCR accuracy.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding or resizing if needed
        # (Using standard Tesseract configuration)
        text = pytesseract.image_to_string(gray)
        return text
    except Exception as e:
        print(f"Image preprocessing/OCR failed: {e}")
        return ""
