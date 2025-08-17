#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Driving License OCR Pipeline - Command Line Interface
Main entry point for the driving license OCR system.
"""

import os
import sys
import argparse
import pandas as pd
import pytesseract
from pathlib import Path

from src.pipeline.processor import process_single_image
from src.pipeline.batch_processor import process_all_images
from src.data.data_loader import find_image_files

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Driving License OCR Pipeline - Extract vehicle classes and validity dates from licenses'
    )
    
    parser.add_argument(
        '--input', '-i', 
        required=True,
        help='Path to input image file or directory containing images'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='data/output/license_extraction_results.csv',
        help='Path to output CSV file (default: data/output/license_extraction_results.csv)'
    )
    
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Process all images in the input directory'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    
    parser.add_argument(
        '--tesseract-path',
        default='',
        help='Path to Tesseract OCR executable (e.g., C:\\Program Files\\Tesseract-OCR\\tesseract.exe)'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set Tesseract path if provided
    if args.tesseract_path and os.path.exists(args.tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
        if args.verbose:
            print(f"Using Tesseract executable: {pytesseract.pytesseract.tesseract_cmd}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process single image or batch
    if os.path.isfile(args.input):
        # Process single image
        print(f"Processing image: {args.input}")
        results_df = process_single_image(args.input, verbose=args.verbose)
        
    elif os.path.isdir(args.input) and args.batch:
        # Process all images in directory
        print(f"Processing all images in directory: {args.input}")
        image_paths = find_image_files(args.input)
        
        if not image_paths:
            print(f"No image files found in {args.input}")
            return 1
            
        print(f"Found {len(image_paths)} images")
        results_df = process_all_images(image_paths)
        
    else:
        print("Error: Input must be a file or a directory with --batch option")
        return 1
    
    # Save results to CSV
    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Display summary
    print("\nResults Summary:")
    print(f"Total entries: {len(results_df)}")
    print(f"Vehicle classes found: {results_df['vehicle_class'].nunique()}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())