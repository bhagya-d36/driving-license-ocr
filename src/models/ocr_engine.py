#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR engine wrappers for PaddleOCR and Tesseract.
"""

import os
import tempfile
import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
from typing import Tuple, List, Union, Dict, Any

# Set Tesseract executable path - use the same logic as in image_utils.py
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Try common installation paths
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'/usr/bin/tesseract',
        r'/usr/local/bin/tesseract'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

class OCREngine:
    """Base OCR Engine class that defines the interface."""
    
    def __init__(self):
        """Initialize the OCR engine."""
        pass
    
    def extract_text(self, image: Union[str, np.ndarray]) -> Tuple[List[Tuple[str, float]], Any]:
        """Extract text from an image."""
        raise NotImplementedError("Subclasses must implement extract_text()")

class PaddleOCREngine(OCREngine):
    """PaddleOCR engine wrapper."""
    
    def __init__(self, lang='en', use_angle_cls=True):
        """Initialize PaddleOCR engine."""
        super().__init__()
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.ocr = None  # Lazy initialization
        
    def _initialize_engine(self):
        """Lazily initialize the OCR engine."""
        if self.ocr is None:
            self.ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
    
    def extract_text(self, image: Union[str, np.ndarray]) -> Tuple[List[Tuple[str, float]], Any]:
        """
        Perform OCR on an image using PaddleOCR.
        
        Args:
            image: Either a file path or a numpy array
            
        Returns:
            Tuple of (extracted_text_with_confidence, raw_results)
        """
        self._initialize_engine()
        
        # Handle image input - could be a file path or a numpy array
        if isinstance(image, str):
            # If it's a string, assume it's a file path
            image_path = image
            # Run OCR on the file
            results = self.ocr.ocr(image_path, cls=True)
        else:
            # If it's a numpy array, save it to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_filename = tmp.name
                # Convert to RGB if it's BGR (OpenCV default)
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Save as JPEG
                    cv2.imwrite(temp_filename, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                else:
                    cv2.imwrite(temp_filename, image)
            
            # Run OCR on the temporary file
            results = self.ocr.ocr(temp_filename, cls=True)
            
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
        
        # Process results
        extracted_text = []
        if results and results[0]:  # Check if results are not empty
            for line in results[0]:
                text = line[1][0]           # Extracted text
                confidence = line[1][1]     # Confidence score
                extracted_text.append((text, confidence))
        
        # Return both the extracted text tuples and the full PaddleOCR results
        return extracted_text, results

class TesseractOCREngine(OCREngine):
    """Tesseract OCR engine wrapper."""
    
    def __init__(self, lang='eng', config=''):
        """Initialize Tesseract OCR engine."""
        super().__init__()
        self.lang = lang
        self.config = config
        
        # Validate Tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("WARNING: Tesseract is not installed or not in PATH. "
                  "Please install Tesseract OCR and make sure it's in your PATH.")
            print(f"Current Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
    
    def extract_text(self, image: Union[str, np.ndarray]) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
        """
        Perform OCR on an image using Tesseract.
        
        Args:
            image: Either a file path or a numpy array
            
        Returns:
            Tuple of (extracted_text_with_confidence, raw_results)
        """
        # Handle different image input types
        if isinstance(image, str):
            # Load the image if a file path is provided
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Convert image to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding for better results
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
        )
            
        try:
            # Get detailed OCR data
            raw_results = pytesseract.image_to_data(
                processed, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            extracted_text = []
            num_boxes = len(raw_results['text'])
            
            for i in range(num_boxes):
                if int(raw_results['conf'][i]) > 0:  # Filter out low confidence results
                    text = raw_results['text'][i]
                    confidence = float(raw_results['conf'][i]) / 100.0  # Normalize to 0-1 range
                    
                    # Skip empty text
                    if text.strip():
                        extracted_text.append((text, confidence))
                        
            return extracted_text, raw_results
        except pytesseract.TesseractNotFoundError:
            print("ERROR: Tesseract OCR is not properly installed or configured.")
            return [], {'text': [], 'conf': []}
        except Exception as e:
            print(f"Tesseract Error: {str(e)}")
            return [], {'text': [], 'conf': []}