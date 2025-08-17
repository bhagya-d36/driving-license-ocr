#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading utilities for license OCR.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict

def find_image_files(folder_path: str, reverse: bool = True) -> List[str]:
    """
    Find all jpg files in a folder and return a sorted list.
    
    Args:
        folder_path: Path to the folder containing images
        reverse: If True, sort in descending order
        
    Returns:
        List of image paths
    """
    # Find all JPG files in the directory, avoiding duplicates
    # Use a set to store unique file paths by their lowercase name
    unique_files = set()
    for img_path in Path(folder_path).glob('*.[jJ][pP][gG]'):
        unique_files.add(str(img_path))
    
    # Convert back to list and sort (reverse=True for descending order)
    image_paths = sorted(list(unique_files), reverse=reverse)
    
    return image_paths

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array or None if loading fails
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if loading was successful
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    return image

def get_image_metadata(image_path: str) -> Dict:
    """
    Get metadata for an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing image metadata
    """
    # Get file info
    file_stats = os.stat(image_path)
    file_size = file_stats.st_size
    
    # Load the image
    image = load_image(image_path)
    
    if image is not None:
        # Get image dimensions
        height, width, channels = image.shape
        
        return {
            'filename': os.path.basename(image_path),
            'path': image_path,
            'size_bytes': file_size,
            'width': width,
            'height': height,
            'channels': channels,
            'aspect_ratio': width / height
        }
    else:
        return {
            'filename': os.path.basename(image_path),
            'path': image_path,
            'size_bytes': file_size,
            'error': 'Failed to load image'
        }

def batch_load_images(image_paths: List[str]) -> Dict[str, np.ndarray]:
    """
    Load multiple images from a list of paths.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        Dictionary mapping file paths to loaded images
    """
    loaded_images = {}
    
    for path in image_paths:
        image = load_image(path)
        if image is not None:
            loaded_images[path] = image
    
    return loaded_images