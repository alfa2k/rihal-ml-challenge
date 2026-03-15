"""JSONL I/O and image loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image


def load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    """Write records to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_image(image_path: str) -> Image.Image | None:
    """Load an image, returning None on failure."""
    try:
        img = Image.open(image_path)
        img.load()
        return img
    except Exception:
        return None


def resolve_image_path(data_dir: str, record: dict) -> str:
    """Resolve the full image path from a data directory and record."""
    rel = record.get("image_path", "")
    return os.path.join(data_dir, rel)


def is_dummy_image(image_path: str, size_threshold: int = 500) -> bool:
    """Check if an image is a placeholder/dummy (tiny file or near-zero variance)."""
    try:
        file_size = os.path.getsize(image_path)
        if file_size <= size_threshold:
            return True
        img = Image.open(image_path).convert("L")
        import numpy as np
        pixels = np.array(img, dtype=np.float32)
        if pixels.var() < 1.0:
            return True
    except Exception:
        return True
    return False


def get_file_size(path: str) -> int:
    """Get file size in bytes, 0 on error."""
    try:
        return os.path.getsize(path)
    except Exception:
        return 0
