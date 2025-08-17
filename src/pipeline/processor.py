#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main processor for the driving license OCR pipeline.
"""

import os
import cv2
import pandas as pd
from typing import Optional

from src.utils.image_utils import best_rotation_by_keyword, locate_table_region
from src.models.ocr_engine import PaddleOCREngine
from src.extraction.text_extraction import extract_vehicle_classes_and_dates
from src.data.output_formatter import results_to_dataframe

def process_single_image(img_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Process a single license image and extract vehicle classes and dates.
    
    Args:
        img_path: Path to the image file
        verbose: If True, print debug information
        
    Returns:
        DataFrame with extracted information
    """
    if verbose:
        print(f"\nProcessing image: {os.path.basename(img_path)}")
    
    # Step 1: Load the image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {img_path}")
        return pd.DataFrame()
    
    # Step 2: Find the best rotation
    rotated_img, angle, scores = best_rotation_by_keyword(image, verbose=verbose)
    
    # Step 3: Locate the table region from the rotated image
    table_region = locate_table_region(rotated_img, verbose=verbose)
    
    # Step 4: Initialize OCR engine and perform OCR on the table region
    ocr_engine = PaddleOCREngine(lang='en')
    _, ocr_results = ocr_engine.extract_text(table_region)
    
    # Step 5: Extract vehicle classes and dates from OCR results
    vehicle_info = extract_vehicle_classes_and_dates(ocr_results)
    
    # Step 6: Format results as a DataFrame
    return results_to_dataframe(vehicle_info, os.path.basename(img_path))