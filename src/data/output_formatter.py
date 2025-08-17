#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Output formatting utilities for license OCR results.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional

def results_to_dataframe(results: List[Dict[str, Any]], image_name: str = "") -> pd.DataFrame:
    """
    Convert OCR results to a pandas DataFrame.
    
    Args:
        results: List of dictionaries with OCR results
        image_name: Name of the processed image
        
    Returns:
        DataFrame with structured OCR results
    """
    # Prepare data for DataFrame
    results_data = []
    for info in results:
        results_data.append({
            'image_name': image_name,
            'vehicle_class': info.get('class', ''),
            'start_date': info.get('start_date', ''),
            'expiry_date': info.get('expiry_date', ''),
            'confidence': info.get('confidence', 0)
        })
    
    # If no results were found, add an empty row
    if not results_data:
        results_data = [{
            'image_name': image_name,
            'vehicle_class': 'N/A',
            'start_date': 'N/A',
            'expiry_date': 'N/A',
            'confidence': 0
        }]
    
    return pd.DataFrame(results_data)

def save_results_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame results to a CSV file.
    
    Args:
        df: DataFrame with results
        output_path: Path where CSV will be saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")

def format_html_report(df: pd.DataFrame, title: str = "Driving License OCR Results") -> str:
    """
    Format OCR results as an HTML report.
    
    Args:
        df: DataFrame with results
        title: Report title
        
    Returns:
        HTML string with formatted report
    """
    # Apply styling
    styled_df = df.style.set_properties(**{
        'background-color': '#f5f5f5',
        'border-color': '#888888',
        'border-style': 'solid',
        'border-width': '1px',
        'text-align': 'center'
    })
    
    # Highlight valid vehicle classes
    valid_classes = ['A1', 'A', 'B1', 'B', 'C1', 'C', 'CE', 'D1', 'D', 'DE', 'G1', 'G', 'J']
    styled_df = styled_df.applymap(
        lambda x: 'background-color: #c6efce' if x in valid_classes else '',
        subset=['vehicle_class']
    )
    
    # Add header and styling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333366; }}
            .summary {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="summary">
            <p>Total images processed: {df['image_name'].nunique()}</p>
            <p>Total vehicle classes detected: {df['vehicle_class'].nunique()}</p>
        </div>
        {styled_df.to_html()}
    </body>
    </html>
    """
    
    return html

def save_html_report(df: pd.DataFrame, output_path: str, title: str = "Driving License OCR Results") -> None:
    """
    Save OCR results as an HTML report.
    
    Args:
        df: DataFrame with results
        output_path: Path where HTML report will be saved
        title: Report title
    """
    html = format_html_report(df, title)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to {output_path}")