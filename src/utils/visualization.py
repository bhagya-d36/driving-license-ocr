#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for displaying images and OCR results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def display_image_with_annotations(image: np.ndarray, 
                                 ocr_results: List, 
                                 title: str = "OCR Results") -> None:
    """
    Display an image with OCR annotations.
    
    Args:
        image: Image to display
        ocr_results: OCR results with text bounding boxes
        title: Title to display on the image
    """
    # Convert BGR to RGB for matplotlib
    display_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(display_img)
    plt.title(title)
    
    # Check if we have valid OCR results with bounding boxes
    if ocr_results and ocr_results[0]:
        for line in ocr_results[0]:
            # Extract bounding box and text
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]
            
            # Draw bounding box as a polygon
            points = np.array(box).astype(np.int32)
            pts = points.reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, (255, 0, 0), 2)
            
            # Calculate text position (above the box)
            text_pos = (int(points[0][0]), int(points[0][1] - 5))
            
            # Display text and confidence
            plt.text(text_pos[0], text_pos[1], f"{text} ({confidence:.2f})", 
                    color='blue', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_processing_steps(original_img: np.ndarray, 
                         rotated_img: np.ndarray, 
                         table_region: np.ndarray, 
                         rotation_angle: int) -> None:
    """
    Display the steps of image processing.
    
    Args:
        original_img: Original input image
        rotated_img: Rotated image after orientation correction
        table_region: Extracted table region
        rotation_angle: Rotation angle applied
    """
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Rotated image
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Rotated Image ({rotation_angle}°)")
    plt.axis('off')
    
    # Table region
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(table_region, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Table Region")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def display_results_table(results_df):
    """Display results in a styled table format."""
    from IPython.display import display, HTML
    import pandas as pd
    
    # Apply styling to the DataFrame
    styled_df = results_df.style.set_properties(**{
        'background-color': '#f5f5f5',
        'border-color': '#888888',
        'border-style': 'solid',
        'border-width': '1px',
        'text-align': 'center'
    })
    
    # Highlight valid vehicle classes
    styled_df = styled_df.applymap(
        lambda x: 'background-color: #c6efce' if x in ['A1', 'A', 'B1', 'B', 'C1', 'C', 'CE', 'D1', 'D', 'DE', 'G1', 'G', 'J'] else '',
        subset=['vehicle_class']
    )
    
    # Display the styled DataFrame
    display(HTML("<h3>License Extraction Results</h3>"))
    display(styled_df)