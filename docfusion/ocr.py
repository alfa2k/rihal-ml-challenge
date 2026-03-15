"""OCR via pytesseract with preprocessing and confidence extraction."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .utils import is_dummy_image

_MAX_DIM = 1024


def _resize_for_ocr(img: Image.Image) -> Image.Image:
    """Resize image so max dimension is _MAX_DIM, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= _MAX_DIM:
        return img
    scale = _MAX_DIM / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Adaptive preprocessing: denoise + sharpen for better OCR accuracy."""
    try:
        gray = img.convert("L")
        # Median filter for salt-and-pepper noise removal
        denoised = gray.filter(ImageFilter.MedianFilter(size=3))
        # Box blur for adaptive threshold approximation
        blurred = denoised.filter(ImageFilter.BoxBlur(radius=15))
        arr = np.array(denoised, dtype=np.float32)
        blur_arr = np.array(blurred, dtype=np.float32)
        # Adaptive threshold: pixel > local_mean - offset
        offset = 10.0
        binary = ((arr > (blur_arr - offset)) * 255).astype(np.uint8)
        return Image.fromarray(binary).convert("RGB")
    except Exception:
        return img


def run_ocr(image_path: str) -> str:
    """Run OCR on an image, returning extracted text. Returns '' for dummy images."""
    if is_dummy_image(image_path):
        return ""
    try:
        import pytesseract
        img = Image.open(image_path).convert("RGB")
        img = _resize_for_ocr(img)
        preprocessed = _preprocess_for_ocr(img)
        text = pytesseract.image_to_string(
            preprocessed, config="--oem 3 --psm 6", timeout=10
        )
        return text.strip()
    except Exception:
        return ""


def run_ocr_detailed(image_path: str) -> dict:
    """Run OCR returning text + confidence statistics.

    Returns dict with keys: text, mean_confidence, word_count, char_count,
    line_count, low_conf_ratio, confidences (list).
    """
    defaults = {
        "text": "",
        "mean_confidence": 0.0,
        "word_count": 0,
        "char_count": 0,
        "line_count": 0,
        "low_conf_ratio": 0.0,
        "confidences": [],
    }
    if is_dummy_image(image_path):
        return defaults
    try:
        import pytesseract
        img = Image.open(image_path).convert("RGB")
        img = _resize_for_ocr(img)
        preprocessed = _preprocess_for_ocr(img)

        # Get word-level data with confidences
        data = pytesseract.image_to_data(
            preprocessed, config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT, timeout=15,
        )

        confidences = []
        words = []
        for i, conf in enumerate(data.get("conf", [])):
            try:
                c = int(conf)
            except (ValueError, TypeError):
                continue
            if c < 0:
                continue
            text_val = data["text"][i].strip()
            if text_val:
                confidences.append(c)
                words.append(text_val)

        full_text = " ".join(words)
        num_words = len(words)
        num_chars = sum(len(w) for w in words)
        num_lines = len(set(
            data["line_num"][i]
            for i in range(len(data.get("line_num", [])))
            if i < len(data.get("text", [])) and data["text"][i].strip()
        ))

        if confidences:
            mean_conf = float(np.mean(confidences))
            low_conf = sum(1 for c in confidences if c < 50) / len(confidences)
        else:
            mean_conf = 0.0
            low_conf = 0.0

        return {
            "text": full_text,
            "mean_confidence": mean_conf,
            "word_count": num_words,
            "char_count": num_chars,
            "line_count": num_lines,
            "low_conf_ratio": low_conf,
            "confidences": confidences,
        }
    except Exception:
        return defaults
