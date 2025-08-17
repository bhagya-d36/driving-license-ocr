#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text extraction logic for driving license OCR.
"""

import re
from datetime import datetime
from typing import List, Dict, Union, Optional, Any

def extract_vehicle_classes_and_dates(ocr_results):
    """
    Extracts vehicle classes and their associated dates from OCR results.
    
    Args:
        ocr_results: OCR results from PaddleOCR
        
    Returns:
        List of dictionaries containing vehicle classes and dates
    """
    # Define valid vehicle classes
    valid_vehicle_classes = [
        "A1", "A", "B1", "B", "C1", "C", "CE", "D1", "D", "DE", "G1", "G", "J"
    ]
    
    # Standard progression of classes for assumption #2
    standard_classes = ["A1", "A", "B1", "B", "C1", "C", "CE", "D1", "D", "DE", "G1", "G", "J"]

    # OCR misrecognition mapping dictionary
    ocr_corrections = {
        "T": "1", "|" : "1", "l": "1", "I": "1", "O": "0", "o": "0",
    }
    
    # Common OCR misrecognitions for vehicle classes
    class_corrections = {
        "81": "B1", "8": "B", "C|": "C1", "CI": "C1", "Cl": "C1",
        "DI": "D1", "Dl": "D1", "D|": "D1", "GI": "G1", "Gl": "G1", "G|": "G1",
        "AI": "A1", "Al": "A1", "A|": "A1",
    }

    def correct_ocr_text(text):
        """Correct commonly misrecognized characters in OCR text"""
        # Special case for specific whole strings
        if text in class_corrections:
            return class_corrections[text]
        elif text in ocr_corrections and len(text) > 1:
            return ocr_corrections[text]
            
        return "".join(ocr_corrections.get(char, char) for char in text)
    
    def convert_lone_number_to_class(text):
        """Convert lone numbers to potential vehicle classes if possible"""
        # Check if this might be a class incorrectly recognized
        text = text.strip().upper()
        if text in class_corrections:
            return class_corrections[text]
            
        # Try to match patterns like "81" to "B1" or similar conversions
        common_replacements = {
            "81": "B1", "A1": "A1", "8": "B", "A": "A", 
            "C1": "C1", "C": "C", "D1": "D1", "D": "D",
            "G1": "G1", "G": "G", "J": "J"
        }
        
        if text in common_replacements:
            return common_replacements[text]
            
        return None  # Not convertible to a valid class

    def normalize_date_string(s):
        """Convert various date formats to standard DD.MM.YYYY format"""
        orig = s
        s = s.strip().replace(" ", "")
        
        # Already in correct format
        if re.match(r"\d{1,2}\.\d{1,2}\.\d{4}$", s):
            return s
            
        # Replace various separators with dots
        s2 = re.sub(r"[/:,-]", ".", s)
        if re.match(r"\d{1,2}\.\d{1,2}\.\d{4}$", s2):
            return s2
            
        # Handle 8-digit format (DDMMYYYY)
        if re.match(r"^\d{8}$", s2):
            return f"{s2[:2]}.{s2[2:4]}.{s2[4:]}"
            
        # Handle formats with various spacing
        m = re.match(r"^(\d{2})[. ]?(\d{2,4})[. ]?(\d{4})$", s)
        if m:
            day = m.group(1)
            month = m.group(2)
            year = m.group(3)
            if len(month) == 2:
                return f"{day}.{month}.{year}"
            elif len(month) == 4:
                return f"{day}.{month[:2]}.{month[2:]}"
                
        # Try to extract just the numbers and format them
        nums = re.findall(r"\d+", s)
        if len(nums) == 3:
            return f"{nums[0]}.{nums[1]}.{nums[2]}"
            
        return orig
    
    def add_expiry_date(start_date_str):
        """Calculate an expiry date 8 years from the start date if after 2012"""
        if not start_date_str:
            return None
            
        try:
            start_date = datetime.strptime(start_date_str, "%d.%m.%Y")
            # Check if year is after 2012
            if start_date.year > 2012:
                expiry_date = datetime(start_date.year + 8, start_date.month, start_date.day)
                return expiry_date.strftime("%d.%m.%Y")
        except Exception:
            pass
            
        return None
        
    def is_valid_date_string(s):
        """Check if string can be parsed as a valid date"""
        try:
            datetime.strptime(s, "%d.%m.%Y")
            return True
        except Exception:
            return False
            
    def validate_date(date_str):
        """Ensure dates don't exceed the current year (2024)"""
        if not date_str:
            return None
            
        try:
            date_obj = datetime.strptime(date_str, "%d.%m.%Y")
            if date_obj.year > 2024:  # Check if date exceeds 2024
                return None
            return date_str
        except Exception:
            return None

    # Check if results are empty
    if not ocr_results or not ocr_results[0]:
        # Return all classes with empty dates if no OCR results
        return [{
            "class": cls, 
            "start_date": None, 
            "expiry_date": None, 
            "confidence": 0.0
        } for cls in standard_classes]

    # Extract text and confidence from OCR results
    extracted_texts = [(line[1][0], line[1][1]) for line in ocr_results[0]]
    
    # First pass - look for recognized vehicle classes and collect standalone dates
    results = []
    classes_found = set()
    standalone_dates = []
    found_dates = set()
    
    # Process the extracted text to find vehicle classes and dates
    for idx, (text, conf) in enumerate(extracted_texts):
        corrected_text = correct_ocr_text(text)
        
        # If we find a vehicle class
        if corrected_text in valid_vehicle_classes:
            classes_found.add(corrected_text)
            entry = {"class": corrected_text, "start_date": None, "expiry_date": None, "confidence": conf}
            dates_found = []
            
            # Look ahead for dates associated with this vehicle class
            scan_idx = idx + 1
            while scan_idx < len(extracted_texts):
                next_text = extracted_texts[scan_idx][0]
                next_corrected = correct_ocr_text(next_text)
                
                # Stop if we hit another vehicle class
                if next_corrected in valid_vehicle_classes:
                    break
                    
                norm_date = normalize_date_string(next_corrected)
                # Try to parse normalized date
                if is_valid_date_string(norm_date):
                    dates_found.append(norm_date)
                    found_dates.add(norm_date)  # Track all found dates
                    
                scan_idx += 1
                
            # Assign dates found to start and expiry fields
            if len(dates_found) >= 2:
                # Sort dates chronologically
                date_objs = [datetime.strptime(d, "%d.%m.%Y") for d in dates_found[:2]]
                if date_objs[0] < date_objs[1]:
                    entry["start_date"] = validate_date(dates_found[0])
                    entry["expiry_date"] = dates_found[1]
                else:
                    entry["start_date"] = validate_date(dates_found[1])
                    entry["expiry_date"] = dates_found[0]
            elif len(dates_found) == 1:
                entry["start_date"] = validate_date(dates_found[0])
                # Calculate expiry date if start date > 2012
                entry["expiry_date"] = add_expiry_date(entry["start_date"])
                
            results.append(entry)
        else:
            # Check if this might be a standalone date
            norm_date = normalize_date_string(corrected_text)
            if is_valid_date_string(norm_date) and norm_date not in found_dates:
                standalone_dates.append((norm_date, conf))
                found_dates.add(norm_date)
    
    # Second pass - try to convert lone numbers to classes
    for idx, (text, conf) in enumerate(extracted_texts):
        # Only process if it's not already a recognized class
        potential_class = convert_lone_number_to_class(text)
        if potential_class and potential_class not in classes_found and potential_class in valid_vehicle_classes:
            classes_found.add(potential_class)
            entry = {"class": potential_class, "start_date": None, "expiry_date": None, "confidence": conf}
            dates_found = []
            
            # Look ahead for dates associated with this potential class
            scan_idx = idx + 1
            while scan_idx < len(extracted_texts):
                next_text = extracted_texts[scan_idx][0]
                next_corrected = correct_ocr_text(next_text)
                
                # Stop if we hit another vehicle class
                if next_corrected in valid_vehicle_classes or convert_lone_number_to_class(next_corrected) in valid_vehicle_classes:
                    break
                    
                norm_date = normalize_date_string(next_corrected)
                # Try to parse normalized date
                if is_valid_date_string(norm_date):
                    dates_found.append(norm_date)
                    
                scan_idx += 1
                
            # Assign dates found to start and expiry fields
            if len(dates_found) >= 2:
                # Sort dates chronologically
                date_objs = [datetime.strptime(d, "%d.%m.%Y") for d in dates_found[:2]]
                if date_objs[0] < date_objs[1]:
                    entry["start_date"] = validate_date(dates_found[0])
                    entry["expiry_date"] = dates_found[1]
                else:
                    entry["start_date"] = validate_date(dates_found[1])
                    entry["expiry_date"] = dates_found[0]
            elif len(dates_found) == 1:
                entry["start_date"] = validate_date(dates_found[0])
                # Calculate expiry date if start date > 2012
                entry["expiry_date"] = add_expiry_date(entry["start_date"])
                
            results.append(entry)
    
    # Third pass - intelligent date pairing and assignment to unrecognized classes
    if standalone_dates:
        # Apply intelligent date pairing based on realistic license periods
        paired_dates = []
        i = 0
        while i < len(standalone_dates):
            if i + 1 < len(standalone_dates):
                date1 = datetime.strptime(standalone_dates[i][0], "%d.%m.%Y")
                date2 = datetime.strptime(standalone_dates[i+1][0], "%d.%m.%Y")
                
                # If dates are within reasonable license timeframe (6-9 years)
                year_diff = abs(date2.year - date1.year)
                if 6 <= year_diff <= 9:
                    # These are likely start/expiry pair
                    if date1 < date2:
                        start_date = validate_date(standalone_dates[i][0])
                        paired_dates.append(((start_date, standalone_dates[i+1][0]), 
                                            (standalone_dates[i][1] + standalone_dates[i+1][1])/2))
                    else:
                        start_date = validate_date(standalone_dates[i+1][0])
                        paired_dates.append(((start_date, standalone_dates[i][0]), 
                                            (standalone_dates[i][1] + standalone_dates[i+1][1])/2))
                    i += 2
                    continue
            
            # Single date
            start_date = validate_date(standalone_dates[i][0])
            paired_dates.append(((start_date, add_expiry_date(start_date)), 
                                standalone_dates[i][1]))
            i += 1
            
        # Assign standalone date pairs to vehicle classes that haven't been found yet
        unassigned_classes = [cls for cls in standard_classes if cls not in classes_found]
        
        for i, ((start_date, expiry_date), conf) in enumerate(paired_dates):
            if i < len(unassigned_classes) and start_date is not None:
                results.append({
                    "class": unassigned_classes[i],
                    "start_date": start_date,
                    "expiry_date": expiry_date,
                    "confidence": conf
                })
    
    # Filter out duplicate vehicle classes, preferring entries with dates
    def filter_duplicate_classes(entries):
        """Remove duplicate vehicle classes, keeping the entry with most date information"""
        class_groups = {}
        # Group entries by vehicle class
        for entry in entries:
            vehicle_class = entry["class"]
            if vehicle_class not in class_groups:
                class_groups[vehicle_class] = []
            class_groups[vehicle_class].append(entry)
        
        # For each group, select the best entry
        filtered_results = []
        for vehicle_class, entries in class_groups.items():
            if len(entries) == 1:
                filtered_results.append(entries[0])
            else:
                # Score entries by date information
                best_entry = None
                best_score = -1
                
                for entry in entries:
                    # Score based on date information - having both dates is better than one date
                    score = 0
                    if entry["start_date"]:
                        score += 1
                    if entry["expiry_date"]:
                        score += 1
                    # Use confidence as a tiebreaker
                    if score > best_score or (score == best_score and entry["confidence"] > best_entry["confidence"]):
                        best_score = score
                        best_entry = entry
                
                filtered_results.append(best_entry)
        
        return filtered_results

    # Apply the filter before returning
    filtered_results = filter_duplicate_classes(results)
    
    # Ensure all standard classes are present in the output
    final_results = []
    classes_in_results = {entry["class"] for entry in filtered_results}
    
    # Add all filtered results to final results
    final_results.extend(filtered_results)
    
    # Add any missing classes with empty dates
    for cls in standard_classes:
        if cls not in classes_in_results:
            final_results.append({
                "class": cls,
                "start_date": None,
                "expiry_date": None,
                "confidence": 0.0  # Default confidence for undetected classes
            })
    
    # Sort results to ensure the order of vehicle classes follows the standard progression
    # This ensures A1, A, B1, B appear in the correct order
    class_order = {cls: idx for idx, cls in enumerate(standard_classes)}
    final_results.sort(key=lambda x: class_order.get(x["class"], float('inf')))
    
    return final_results