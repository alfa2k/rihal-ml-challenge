"""Regex-based vendor/date/total extraction from OCR text."""

from __future__ import annotations

import re


def extract_vendor(text: str) -> str | None:
    """Extract vendor name: first non-empty, non-numeric line."""
    if not text:
        return None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip lines that are purely numeric/symbols
        if re.match(r'^[\d\s\.\,\-\+\$\%\#\*\=\/]+$', line):
            continue
        # Skip very short lines
        if len(line) < 2:
            continue
        return line
    return None


def extract_date(text: str) -> str | None:
    """Extract date from text, supporting ISO/US/EU formats."""
    if not text:
        return None

    patterns = [
        # ISO: 2024-01-15
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        # US: 01/15/2024 or 01-15-2024
        (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
        # EU: 15.01.2024
        (r'(\d{1,2})\.(\d{1,2})\.(\d{4})', lambda m: f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"),
        # Short year: 01/15/24
        (r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})(?!\d)', lambda m: f"20{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
    ]
    for pattern, formatter in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                result = formatter(match)
                # Basic validation
                parts = result.split("-")
                y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
                if 1900 <= y <= 2099 and 1 <= mo <= 12 and 1 <= d <= 31:
                    return result
            except (ValueError, IndexError):
                continue
    return None


def extract_total(text: str) -> str | None:
    """Extract total amount, preferring values near 'TOTAL' keyword."""
    if not text:
        return None

    # Look for TOTAL keyword line first
    total_pattern = re.compile(
        r'(?:total|amount\s*due|grand\s*total|balance\s*due)\s*[:\s]*[\$]?\s*([\d,]+\.?\d*)',
        re.IGNORECASE
    )
    match = total_pattern.search(text)
    if match:
        val = match.group(1).replace(",", "")
        try:
            return f"{float(val):.2f}"
        except ValueError:
            pass

    # Fallback: find all monetary values and take the largest
    money_pattern = re.compile(r'[\$]?\s*(\d{1,}[,\d]*\.\d{2})\b')
    amounts = []
    for m in money_pattern.finditer(text):
        try:
            val = float(m.group(1).replace(",", ""))
            amounts.append(val)
        except ValueError:
            continue
    if amounts:
        return f"{max(amounts):.2f}"

    return None


def extract_fields_from_text(text: str) -> dict:
    """Extract all fields from OCR text."""
    return {
        "vendor": extract_vendor(text),
        "date": extract_date(text),
        "total": extract_total(text),
    }


def extract_fields_with_fallback(ocr_text: str, jsonl_fields: dict | None) -> dict:
    """Extract fields from OCR, falling back to JSONL fields."""
    ocr_fields = extract_fields_from_text(ocr_text)
    if jsonl_fields is None:
        return ocr_fields

    result = {}
    for key in ("vendor", "date", "total"):
        ocr_val = ocr_fields.get(key)
        jsonl_val = jsonl_fields.get(key) if jsonl_fields else None
        result[key] = ocr_val if ocr_val is not None else jsonl_val
    return result
