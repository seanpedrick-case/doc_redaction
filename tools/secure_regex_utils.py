"""
Secure regex utilities to prevent ReDoS (Regular Expression Denial of Service) attacks.

This module provides safe alternatives to common regex patterns that can cause
catastrophic backtracking and performance issues.
"""

import re
from typing import List, Optional


def safe_extract_numbers_with_seconds(text: str) -> List[float]:
    """
    Safely extract numbers before 'seconds' from text without ReDoS vulnerability.

    Args:
        text: The text to search for numbers followed by 'seconds'

    Returns:
        List of float numbers found before 'seconds'
    """
    if not text or not isinstance(text, str):
        return []

    # Use a more specific pattern that avoids catastrophic backtracking
    # Look for digits, optional decimal part, optional whitespace, then 'seconds'
    pattern = r"\b(\d+(?:\.\d+)?)\s*seconds\b"

    matches = re.findall(pattern, text)
    try:
        return [float(match) for match in matches]
    except (ValueError, TypeError):
        return []


def safe_extract_numbers(text: str) -> List[float]:
    """
    Safely extract all numbers from text without ReDoS vulnerability.

    Args:
        text: The text to extract numbers from

    Returns:
        List of float numbers found in the text
    """
    if not text or not isinstance(text, str):
        return []

    # Use a simple, safe pattern that doesn't cause backtracking
    # Match digits, optional decimal point and more digits
    pattern = r"\b\d+(?:\.\d+)?\b"

    matches = re.findall(pattern, text)
    try:
        return [float(match) for match in matches]
    except (ValueError, TypeError):
        return []


def safe_extract_page_number_from_filename(filename: str) -> Optional[int]:
    """
    Safely extract page number from filename ending with .png.

    Args:
        filename: The filename to extract page number from

    Returns:
        Page number if found, None otherwise
    """
    if not filename or not isinstance(filename, str):
        return None

    # Use a simple, safe pattern
    pattern = r"(\d+)\.png$"
    match = re.search(pattern, filename)

    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            return None

    return None


def safe_extract_page_number_from_path(path: str) -> Optional[int]:
    """
    Safely extract page number from path containing _(\d+).png pattern.

    Args:
        path: The path to extract page number from

    Returns:
        Page number if found, None otherwise
    """
    if not path or not isinstance(path, str):
        return None

    # Use a simple, safe pattern
    pattern = r"_(\d+)\.png$"
    match = re.search(pattern, path)

    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            return None

    return None


def safe_clean_text(text: str, remove_html: bool = True) -> str:
    """
    Safely clean text without ReDoS vulnerability.

    Args:
        text: The text to clean
        remove_html: Whether to remove HTML tags

    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = text

    if remove_html:
        # Use a simple pattern that doesn't cause backtracking
        cleaned = re.sub(r"<[^>]*>", "", cleaned)

    # Clean up whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def safe_extract_rgb_values(text: str) -> Optional[tuple]:
    """
    Safely extract RGB values from text like "(255, 255, 255)".

    Args:
        text: The text to extract RGB values from

    Returns:
        Tuple of (r, g, b) values if found, None otherwise
    """
    if not text or not isinstance(text, str):
        return None

    # Use a simple, safe pattern
    pattern = r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)"
    match = re.match(pattern, text.strip())

    if match:
        try:
            r = int(match.group(1))
            g = int(match.group(2))
            b = int(match.group(3))

            # Validate RGB values
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return (r, g, b)
        except (ValueError, TypeError):
            pass

    return None


def safe_split_filename(filename: str, delimiters: List[str]) -> List[str]:
    """
    Safely split filename by delimiters without ReDoS vulnerability.

    Args:
        filename: The filename to split
        delimiters: List of delimiter patterns to split on

    Returns:
        List of filename parts
    """
    if not filename or not isinstance(filename, str):
        return []

    if not delimiters:
        return [filename]

    # Escape special regex characters in delimiters
    escaped_delimiters = [re.escape(delim) for delim in delimiters]

    # Create a safe pattern
    pattern = "|".join(escaped_delimiters)

    try:
        return re.split(pattern, filename)
    except re.error:
        # Fallback to simple string operations if regex fails
        result = [filename]
        for delim in delimiters:
            new_result = []
            for part in result:
                new_result.extend(part.split(delim))
            result = new_result
        return result


def safe_remove_leading_newlines(text: str) -> str:
    """
    Safely remove leading newlines without ReDoS vulnerability.

    Args:
        text: The text to clean

    Returns:
        Text with leading newlines removed
    """
    if not text or not isinstance(text, str):
        return ""

    # Use a simple pattern
    return re.sub(r"^\n+", "", text).strip()


def safe_remove_non_ascii(text: str) -> str:
    """
    Safely remove non-ASCII characters without ReDoS vulnerability.

    Args:
        text: The text to clean

    Returns:
        Text with non-ASCII characters removed
    """
    if not text or not isinstance(text, str):
        return ""

    # Use a simple pattern
    return re.sub(r"[^\x00-\x7F]", "", text)


def safe_extract_latest_number_from_filename(filename: str) -> Optional[int]:
    """
    Safely extract the latest/largest number from filename without ReDoS vulnerability.

    Args:
        filename: The filename to extract number from

    Returns:
        The largest number found, or None if no numbers found
    """
    if not filename or not isinstance(filename, str):
        return None

    # Use a simple pattern to find all numbers
    pattern = r"\d+"
    matches = re.findall(pattern, filename)

    if not matches:
        return None

    try:
        # Convert to integers and return the maximum
        numbers = [int(match) for match in matches]
        return max(numbers)
    except (ValueError, TypeError):
        return None


def safe_sanitize_text(text: str, replacement: str = "_") -> str:
    """
    Safely sanitize text by removing dangerous characters without ReDoS vulnerability.

    Args:
        text: The text to sanitize
        replacement: Character to replace dangerous characters with

    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""

    # Use a simple pattern for dangerous characters
    dangerous_chars = r'[<>:"|?*\\/\x00-\x1f\x7f-\x9f]'
    sanitized = re.sub(dangerous_chars, replacement, text)

    # Remove multiple consecutive replacements
    sanitized = re.sub(f"{re.escape(replacement)}+", replacement, sanitized)

    # Remove leading/trailing replacements
    sanitized = sanitized.strip(replacement)

    return sanitized
