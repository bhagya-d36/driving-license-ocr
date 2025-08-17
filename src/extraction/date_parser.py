#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Date parsing and validation utilities for license OCR.
"""

import re
from datetime import datetime
from typing import Optional, Tuple, List

def parse_date(date_string: str) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.
    
    Args:
        date_string: Date string to parse
        
    Returns:
        datetime object if successful, None otherwise
    """
    # Clean the input string
    cleaned = date_string.strip()
    
    # Try various date formats
    formats = [
        "%d.%m.%Y",  # 01.01.2020
        "%d-%m-%Y",  # 01-01-2020
        "%d/%m/%Y",  # 01/01/2020
        "%Y.%m.%d",  # 2020.01.01
        "%Y-%m-%d",  # 2020-01-01
        "%Y/%m/%d",  # 2020/01/01
        "%d.%m.%y",  # 01.01.20
        "%d-%m-%y",  # 01-01-20
        "%d/%m/%y",  # 01/01/20
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    
    # Try to extract date components if standard formats fail
    # Pattern for DD.MM.YYYY with flexible separators
    pattern = r"(\d{1,2})[./\-\s](\d{1,2})[./\-\s](\d{2,4})"
    match = re.match(pattern, cleaned)
    if match:
        day, month, year = match.groups()
        
        # Handle two-digit years
        if len(year) == 2:
            current_century = datetime.now().year // 100 * 100
            year_num = int(year)
            if year_num > 50:  # Assume years > 50 are from previous century
                year = str(year_num + (current_century - 100))
            else:
                year = str(year_num + current_century)
        
        try:
            return datetime(int(year), int(month), int(day))
        except ValueError:
            pass
            
    return None

def extract_dates_from_text(text: str) -> List[datetime]:
    """
    Extract all date-like patterns from text.
    
    Args:
        text: Text to search for dates
        
    Returns:
        List of datetime objects
    """
    # List of date patterns to search for
    patterns = [
        r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b",  # DD/MM/YYYY or DD/MM/YY
        r"\b\d{2,4}[./\-]\d{1,2}[./\-]\d{1,2}\b",  # YYYY/MM/DD or YY/MM/DD
        r"\b\d{8}\b"  # DDMMYYYY or YYYYMMDD
    ]
    
    # Find all potential date strings
    potential_dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        potential_dates.extend(matches)
        
    # Try to parse each potential date
    valid_dates = []
    for date_str in potential_dates:
        date_obj = parse_date(date_str)
        if date_obj:
            valid_dates.append(date_obj)
            
    return valid_dates

def validate_date_range(start_date: Optional[datetime], 
                       end_date: Optional[datetime],
                       min_year: int = 1990,
                       max_year: int = 2040) -> bool:
    """
    Validate a date range to ensure it's reasonable for a license.
    
    Args:
        start_date: Start date
        end_date: End date
        min_year: Minimum valid year
        max_year: Maximum valid year
        
    Returns:
        True if the date range is valid, False otherwise
    """
    if not start_date or not end_date:
        return False
        
    # Check year ranges
    if start_date.year < min_year or start_date.year > max_year:
        return False
        
    if end_date.year < min_year or end_date.year > max_year:
        return False
        
    # End date should be after start date
    if end_date <= start_date:
        return False
        
    # License validity is typically between 5-10 years
    years_diff = (end_date.year - start_date.year) + \
                 (end_date.month - start_date.month) / 12
                 
    if years_diff < 3 or years_diff > 12:
        return False
        
    return True

def format_date(date_obj: Optional[datetime], format_str: str = "%d.%m.%Y") -> Optional[str]:
    """
    Format a datetime object into a string.
    
    Args:
        date_obj: datetime object to format
        format_str: Format string to use
        
    Returns:
        Formatted date string or None if input is None
    """
    if date_obj:
        return date_obj.strftime(format_str)
    return None