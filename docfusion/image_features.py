"""Image-based feature extraction: ELA, pixel stats, edge density, texture, color, JPEG artifacts."""
from __future__ import annotations

import io
import os
import math

import numpy as np
from PIL import Image, ImageFilter

from .utils import is_dummy_image, get_file_size


def compute_ela(img: Image.Image, quality: int = 90) -> np.ndarray:
    """Compute Error Level Analysis image."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    original = img.convert("RGB")

    ela = np.abs(
        np.array(original, dtype=np.float32) - np.array(recompressed, dtype=np.float32)
    )
    return ela


def _compute_glcm_features(gray_arr: np.ndarray) -> dict:
    """Compute GLCM texture features using pure numpy (16-level quantized)."""
    features = {
        "glcm_contrast": 0.0,
        "glcm_energy": 0.0,
        "glcm_homogeneity": 0.0,
        "glcm_correlation": 0.0,
    }
    try:
        # Quantize to 16 levels
        levels = 16
        q = (gray_arr / 256.0 * levels).astype(np.int32).clip(0, levels - 1)
        # Subsample for speed
        h, w = q.shape
        step = max(1, min(h, w) // 128)
        q = q[::step, ::step]
        h, w = q.shape
        if h < 2 or w < 2:
            return features
        # Build co-occurrence matrix (horizontal offset=1)
        glcm = np.zeros((levels, levels), dtype=np.float64)
        left = q[:, :-1].ravel()
        right = q[:, 1:].ravel()
        for i, j in zip(left, right):
            glcm[i, j] += 1
            glcm[j, i] += 1
        total = glcm.sum()
        if total == 0:
            return features
        glcm /= total

        # Compute features
        ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        diff_sq = (ii - jj).astype(np.float64) ** 2
        features["glcm_contrast"] = float(np.sum(diff_sq * glcm))
        features["glcm_energy"] = float(np.sum(glcm ** 2))
        features["glcm_homogeneity"] = float(np.sum(glcm / (1.0 + np.abs(ii - jj))))
        # Correlation
        mu_i = float(np.sum(ii * glcm))
        mu_j = float(np.sum(jj * glcm))
        sigma_i = float(np.sqrt(np.sum(((ii - mu_i) ** 2) * glcm)))
        sigma_j = float(np.sqrt(np.sum(((jj - mu_j) ** 2) * glcm)))
        if sigma_i > 1e-10 and sigma_j > 1e-10:
            features["glcm_correlation"] = float(
                np.sum((ii - mu_i) * (jj - mu_j) * glcm) / (sigma_i * sigma_j)
            )
    except Exception:
        pass
    return features


def _compute_color_histogram_features(pixels: np.ndarray) -> dict:
    """Compute color histogram statistics per channel: skewness, kurtosis, IQR + uniformity."""
    features = {}
    channel_names = ["r", "g", "b"]
    all_stds = []
    for i, ch in enumerate(channel_names):
        try:
            channel = pixels[:, :, i].ravel()
            mean = channel.mean()
            std = channel.std()
            all_stds.append(std)
            if std > 1e-10:
                skew = float(np.mean(((channel - mean) / std) ** 3))
                kurt = float(np.mean(((channel - mean) / std) ** 4) - 3.0)
            else:
                skew = 0.0
                kurt = 0.0
            q75 = float(np.percentile(channel, 75))
            q25 = float(np.percentile(channel, 25))
            features[f"hist_skewness_{ch}"] = skew
            features[f"hist_kurtosis_{ch}"] = kurt
            features[f"hist_iqr_{ch}"] = q75 - q25
        except Exception:
            features[f"hist_skewness_{ch}"] = 0.0
            features[f"hist_kurtosis_{ch}"] = 0.0
            features[f"hist_iqr_{ch}"] = 0.0
    # Color uniformity: mean of std deviations across channels
    features["color_uniformity"] = float(np.mean(all_stds)) if all_stds else 0.0
    return features


def _compute_jpeg_grid_features(gray_arr: np.ndarray) -> dict:
    """Detect 8x8 JPEG block boundary artifacts."""
    features = {
        "jpeg_grid_h": 0.0,
        "jpeg_grid_v": 0.0,
        "jpeg_grid_ratio": 0.0,
    }
    try:
        h, w = gray_arr.shape
        if h < 16 or w < 16:
            return features
        arr = gray_arr.astype(np.float64)
        # Horizontal gradient at block boundaries vs non-boundaries
        h_grad = np.abs(np.diff(arr, axis=1))
        boundary_cols = list(range(7, w - 1, 8))
        non_boundary_cols = [c for c in range(w - 1) if c not in boundary_cols]
        if boundary_cols and non_boundary_cols:
            boundary_strength = float(h_grad[:, boundary_cols].mean())
            non_boundary_strength = float(h_grad[:, non_boundary_cols].mean())
            features["jpeg_grid_h"] = boundary_strength
            if non_boundary_strength > 1e-10:
                features["jpeg_grid_ratio"] = boundary_strength / non_boundary_strength
        # Vertical gradient at block boundaries
        v_grad = np.abs(np.diff(arr, axis=0))
        boundary_rows = list(range(7, h - 1, 8))
        if boundary_rows:
            features["jpeg_grid_v"] = float(v_grad[boundary_rows, :].mean())
    except Exception:
        pass
    return features


def _compute_multi_ela_features(img: Image.Image) -> dict:
    """ELA at multiple quality levels for manipulation detection."""
    features = {
        "ela_q75_mean": 0.0,
        "ela_q95_mean": 0.0,
        "ela_diff_75_95": 0.0,
        "ela_ratio_75_95": 0.0,
        "ela_std_slope": 0.0,
    }
    try:
        ela_75 = compute_ela(img, quality=75)
        ela_95 = compute_ela(img, quality=95)
        m75 = float(ela_75.mean())
        m95 = float(ela_95.mean())
        features["ela_q75_mean"] = m75
        features["ela_q95_mean"] = m95
        features["ela_diff_75_95"] = m75 - m95
        if m95 > 1e-10:
            features["ela_ratio_75_95"] = m75 / m95
        # Std slope across qualities
        ela_90 = compute_ela(img, quality=90)
        stds = [float(ela_75.std()), float(ela_90.std()), float(ela_95.std())]
        if len(stds) >= 2:
            features["ela_std_slope"] = stds[0] - stds[-1]
    except Exception:
        pass
    return features


def _compute_ocr_features(image_path: str) -> dict:
    """Extract OCR-based features (confidence, counts, density)."""
    defaults = {
        "ocr_mean_confidence": 0.0,
        "ocr_word_count": 0,
        "ocr_char_count": 0,
        "ocr_line_count": 0,
        "ocr_low_conf_ratio": 0.0,
        "ocr_text_density": 0.0,
    }
    try:
        from .ocr import run_ocr_detailed
        result = run_ocr_detailed(image_path)
        defaults["ocr_mean_confidence"] = result.get("mean_confidence", 0.0)
        defaults["ocr_word_count"] = result.get("word_count", 0)
        defaults["ocr_char_count"] = result.get("char_count", 0)
        defaults["ocr_line_count"] = result.get("line_count", 0)
        defaults["ocr_low_conf_ratio"] = result.get("low_conf_ratio", 0.0)
        # Text density: chars per 1000 pixels
        try:
            img = Image.open(image_path)
            w, h = img.size
            area = w * h
            if area > 0:
                defaults["ocr_text_density"] = result.get("char_count", 0) / (area / 1000.0)
        except Exception:
            pass
    except Exception:
        pass
    return defaults


def extract_image_features(image_path: str) -> dict:
    """Extract a comprehensive feature dict from an image file."""
    features = _default_features()

    if is_dummy_image(image_path):
        features["is_dummy"] = 1
        features["file_size"] = get_file_size(image_path)
        return features

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        features["is_dummy"] = 1
        return features

    pixels = np.array(img, dtype=np.float32)
    w, h = img.size

    features["file_size"] = get_file_size(image_path)
    features["width"] = w
    features["height"] = h
    features["is_dummy"] = 0

    # Aspect ratio, pixel count, compression ratio
    features["aspect_ratio"] = float(w) / max(h, 1)
    features["pixel_count"] = w * h
    file_size = get_file_size(image_path)
    if w * h > 0:
        features["compression_ratio"] = file_size / (w * h * 3)
    else:
        features["compression_ratio"] = 0.0

    # Per-channel pixel stats
    for i, ch in enumerate(["r", "g", "b"]):
        features[f"pixel_mean_{ch}"] = float(pixels[:, :, i].mean())
        features[f"pixel_std_{ch}"] = float(pixels[:, :, i].std())

    # ELA features (Q90 baseline)
    try:
        ela = compute_ela(img)
        features["ela_mean"] = float(ela.mean())
        features["ela_std"] = float(ela.std())
        features["ela_max"] = float(ela.max())
    except Exception:
        pass

    # Multi-quality ELA
    try:
        multi_ela = _compute_multi_ela_features(img)
        features.update(multi_ela)
    except Exception:
        pass

    # Edge density
    try:
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges, dtype=np.float32)
        features["edge_density"] = float((edge_arr > 30).mean())
    except Exception:
        pass

    # Noise level
    try:
        gray = img.convert("L")
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
        noise = np.array(gray, dtype=np.float32) - np.array(blurred, dtype=np.float32)
        features["noise_level"] = float(noise.std())
    except Exception:
        pass

    # GLCM texture features
    try:
        gray_arr = np.array(img.convert("L"), dtype=np.float32)
        glcm_feats = _compute_glcm_features(gray_arr)
        features.update(glcm_feats)
    except Exception:
        pass

    # Color histogram features
    try:
        color_feats = _compute_color_histogram_features(pixels)
        features.update(color_feats)
    except Exception:
        pass

    # JPEG grid detection
    try:
        gray_arr = np.array(img.convert("L"), dtype=np.float32)
        jpeg_feats = _compute_jpeg_grid_features(gray_arr)
        features.update(jpeg_feats)
    except Exception:
        pass

    # OCR confidence features
    try:
        ocr_feats = _compute_ocr_features(image_path)
        features.update(ocr_feats)
    except Exception:
        pass

    return features


def _default_features() -> dict:
    """Return a dict of default (zero) feature values."""
    return {
        # Basic
        "file_size": 0,
        "width": 0,
        "height": 0,
        "aspect_ratio": 0.0,
        "pixel_count": 0,
        "compression_ratio": 0.0,
        # Pixel stats
        "pixel_mean_r": 0.0,
        "pixel_mean_g": 0.0,
        "pixel_mean_b": 0.0,
        "pixel_std_r": 0.0,
        "pixel_std_g": 0.0,
        "pixel_std_b": 0.0,
        # ELA baseline
        "ela_mean": 0.0,
        "ela_std": 0.0,
        "ela_max": 0.0,
        # Multi-quality ELA
        "ela_q75_mean": 0.0,
        "ela_q95_mean": 0.0,
        "ela_diff_75_95": 0.0,
        "ela_ratio_75_95": 0.0,
        "ela_std_slope": 0.0,
        # Edge & noise
        "edge_density": 0.0,
        "noise_level": 0.0,
        # GLCM texture
        "glcm_contrast": 0.0,
        "glcm_energy": 0.0,
        "glcm_homogeneity": 0.0,
        "glcm_correlation": 0.0,
        # Color histograms
        "hist_skewness_r": 0.0,
        "hist_skewness_g": 0.0,
        "hist_skewness_b": 0.0,
        "hist_kurtosis_r": 0.0,
        "hist_kurtosis_g": 0.0,
        "hist_kurtosis_b": 0.0,
        "hist_iqr_r": 0.0,
        "hist_iqr_g": 0.0,
        "hist_iqr_b": 0.0,
        "color_uniformity": 0.0,
        # JPEG grid
        "jpeg_grid_h": 0.0,
        "jpeg_grid_v": 0.0,
        "jpeg_grid_ratio": 0.0,
        # OCR confidence
        "ocr_mean_confidence": 0.0,
        "ocr_word_count": 0,
        "ocr_char_count": 0,
        "ocr_line_count": 0,
        "ocr_low_conf_ratio": 0.0,
        "ocr_text_density": 0.0,
        # Dummy flag
        "is_dummy": 0,
    }


IMAGE_FEATURE_NAMES = list(_default_features().keys())
