#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch processor for multiple driving license images.
"""

import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

from src.utils.image_utils import best_rotation_by_keyword, locate_table_region
from src.models.ocr_engine import PaddleOCREngine
from src.extraction.text_extraction import extract_vehicle_classes_and_dates
from src.data.output_formatter import results_to_dataframe

def process_all_images(image_paths: List[str]) -> pd.DataFrame:
    """
    Process all images and collect results into a DataFrame.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        DataFrame with extracted information
    """
    all_results = []
    ocr_engine = PaddleOCREngine(lang='en')
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Step 1: Load the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        try:
            # Step 2: Find the best rotation
            rotated_img, angle, _ = best_rotation_by_keyword(image, verbose=False)
            
            # Step 3: Locate the table region from the rotated image
            table_region = locate_table_region(rotated_img, verbose=False)
            
            # Step 4: Perform OCR on the table region
            _, ocr_results = ocr_engine.extract_text(table_region)
            
            # Step 5: Extract vehicle classes and dates from OCR results
            vehicle_info = extract_vehicle_classes_and_dates(ocr_results)
            
            # Create a DataFrame for this image and add to results
            image_df = results_to_dataframe(vehicle_info, os.path.basename(img_path))
            
            # Add rotation angle information
            image_df['rotation_angle'] = angle
            
            # Append to all results
            all_results.append(image_df)
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            # Create an error entry
            error_df = pd.DataFrame([{
                'image_name': os.path.basename(img_path),
                'vehicle_class': 'ERROR',
                'start_date': '',
                'expiry_date': '',
                'confidence': 0,
                'rotation_angle': 0,
                'error': str(e)
            }])
            all_results.append(error_df)
    
    # Combine all results into one DataFrame
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()