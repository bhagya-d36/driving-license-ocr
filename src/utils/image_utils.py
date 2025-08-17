#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image processing utilities for the driving license OCR pipeline.
"""

import cv2
import numpy as np
import pytesseract
import os
from typing import Tuple, Dict, List, Optional

# Set Tesseract executable path - adjust this to your Tesseract installation path
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

def best_rotation_by_keyword(image: np.ndarray, 
                            keywords=('Department', 'Motor', 'Traffic', 'Sri', 'Lanka', 'LICENCE', 'Surname', 'Number', 'Date', 'Category'), 
                            verbose: bool=False) -> Tuple[np.ndarray, int, Dict[int, int]]:
    """
    Determine the best rotation angle for an image by detecting keywords.
    
    Args:
        image: Input image as numpy array
        keywords: List of keywords to look for in the image
        verbose: If True, print debug information
        
    Returns:
        Tuple of (rotated_image, best_angle, scores_dict)
    """
    # Check if Tesseract is properly configured
    if verbose:
        print(f"  Using Tesseract executable: {pytesseract.pytesseract.tesseract_cmd}")
    
    try:
        # Test Tesseract before proceeding
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        if verbose:
            print("  ERROR: Tesseract not found. Please install Tesseract OCR and set the correct path.")
        # Return original image as fallback
        return image, 0, {0: 0, 90: 0, 180: 0, 270: 0}
    
    # Prepare candidate images rotated by 0°, 90°, 180°, and 270° angles
    candidates = [
        (image, 0),  # original image, no rotation
        (cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 90),  # rotate 90 degrees clockwise
        (cv2.rotate(image, cv2.ROTATE_180), 180),  # rotate 180 degrees
        (cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 270)  # rotate 270 degrees clockwise
    ]

    best_score = -1         # track highest keyword match count found so far
    best_img = image        # image corresponding to the best score (start with original)
    best_angle = 0          # angle corresponding to the best score
    scores = {}             # dictionary to store score per angle

    # Loop over each rotated candidate image and its angle
    for img, angle in candidates:
        # Convert image to grayscale for better OCR performance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to convert the image to black and white
        proc = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 15)
        
        try:
            # Run OCR on the processed image to extract text
            text = pytesseract.image_to_string(proc, config="--psm 6")
            
            # Count how many of the keywords appear in the extracted text (case-insensitive)
            score = sum(1 for k in keywords if k.lower() in text.lower())
        except Exception as e:
            if verbose:
                print(f"  Tesseract error: {str(e)}")
            score = 0
        
        # Store this score with its corresponding angle
        scores[angle] = score
        
        # If verbose, print out the current trial angle and how many keywords were found
        if verbose:
            print(f"  Orientation trial angle={angle:3d} score={score} (keywords found)")
        
        # Update best score and best image if this rotation yields more keyword matches
        if score > best_score:
            best_score = score
            best_img = img
            best_angle = angle

    # After checking all rotations, optionally print the final chosen angle and score
    if verbose:
        print(f"  Selected rotation angle={best_angle} with score={best_score}")
    
    return best_img, best_angle, scores

# The rest of your functions remain the same
def locate_table_region(image: np.ndarray, verbose: bool=False) -> np.ndarray:
    """
    Locate the table region in the license image that likely contains vehicle class information.
    
    Args:
        image: Input image as numpy array
        verbose: If True, print debug information
        
    Returns:
        Cropped image containing the table region
    """
    # Calculate adaptive parameters based on image size
    h, w = image.shape[:2]
    horiz_kernel_size = max(int(w * 0.05), 15)
    vert_kernel_size = max(int(h * 0.05), 15)
    
    # Convert input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Thresholding
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 9)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 5)
    bw = cv2.bitwise_or(thresh1, thresh2)
    
    # Morphological operations
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_size, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_size))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    grid = cv2.add(horiz, vert)
    kernel_size = max(3, min(int(min(h, w) * 0.005), 7))
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)), iterations=1)

    # Find contours
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        aspect_ratio = ww / float(hh) if hh > 0 else 0
        if 0.1*w < ww < 0.98*w and 0.1*h < hh < 0.95*h and 0.5 < aspect_ratio < 5:
            candidates.append({'contour': cnt, 'rect': (x, y, ww, hh), 'area': area, 'aspect_ratio': aspect_ratio})
    candidates.sort(key=lambda c: c['area'], reverse=True)
    
    if candidates:
        x, y, ww, hh = candidates[0]['rect']
        if verbose:
            print(f"  Table ROI detected at x={x}, y={y}, w={ww}, h={hh}, area={candidates[0]['area']}")
            print(f"  Aspect ratio: {candidates[0]['aspect_ratio']:.2f}")
        return image[y:y+hh, x:x+ww]

    # Edge-based fallback
    if verbose:
        print("  No table detected with grid method, trying edge-based detection...")
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                          minLineLength=min(h, w)//4, maxLineGap=20)
    
    if lines is not None and len(lines) > 5:
        mask = np.zeros_like(gray)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, ww, hh = cv2.boundingRect(cnt)
            if 0.1*w < ww < 0.98*w and 0.1*h < hh < 0.95*h:
                if verbose:
                    print(f"  Table ROI detected with fallback method at x={x}, y={y}, w={ww}, h={hh}")
                return image[y:y+hh, x:x+ww]

    if verbose:
        print("  All detection methods failed; using full image")
    return image

def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    
    return binary