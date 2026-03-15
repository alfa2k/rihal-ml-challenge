"""Field-based feature extraction: total z-scores, vendor encoding, date features, Benford's Law, cents anomaly."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import datetime

import numpy as np


def compute_vendor_stats(records: list[dict]) -> dict:
    """Compute per-vendor total statistics and Benford distributions.

    Returns {vendor: {"mean": float, "std": float, "count": int, "benford": list[float]}}
    """
    vendor_totals = defaultdict(list)
    for rec in records:
        fields = rec.get("fields", {})
        vendor = fields.get("vendor")
        total_str = fields.get("total")
        if vendor and total_str:
            try:
                vendor_totals[vendor].append(float(total_str))
            except (ValueError, TypeError):
                continue

    stats = {}
    for vendor, totals in vendor_totals.items():
        arr = np.array(totals)
        # Benford distribution for this vendor
        benford = _compute_benford_distribution(totals)
        stats[vendor] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()) if len(arr) > 1 else 50.0,
            "count": len(arr),
            "benford": benford,
        }
    return stats


def _compute_benford_distribution(values: list[float]) -> list[float]:
    """Compute leading digit distribution (digits 1-9)."""
    counts = [0] * 10  # index 0 unused, 1-9
    for v in values:
        try:
            leading = int(str(abs(v)).lstrip("0").lstrip(".")[0])
            if 1 <= leading <= 9:
                counts[leading] += 1
        except (IndexError, ValueError):
            continue
    total = sum(counts[1:])
    if total == 0:
        return [1.0 / 9.0] * 9  # uniform
    return [counts[d] / total for d in range(1, 10)]


# Theoretical Benford distribution
_BENFORD_EXPECTED = [math.log10(1 + 1.0 / d) for d in range(1, 10)]


def compute_global_stats(records: list[dict]) -> dict:
    """Compute global total statistics."""
    totals = []
    for rec in records:
        fields = rec.get("fields", {})
        total_str = fields.get("total")
        if total_str:
            try:
                totals.append(float(total_str))
            except (ValueError, TypeError):
                continue
    arr = np.array(totals) if totals else np.array([0.0])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()) if len(arr) > 1 else 50.0,
    }


def build_vendor_encoding(records: list[dict]) -> dict:
    """Build vendor -> integer encoding from training records."""
    vendors = set()
    for rec in records:
        v = rec.get("fields", {}).get("vendor")
        if v:
            vendors.add(v)
    return {v: i for i, v in enumerate(sorted(vendors))}


def extract_field_features(
    fields: dict,
    vendor_stats: dict,
    global_stats: dict,
    vendor_encoding: dict,
) -> dict:
    """Extract features from a record's fields."""
    features = {}

    # Total features
    total_str = fields.get("total")
    total_float = _parse_total(total_str)
    features["total_float"] = total_float if total_float is not None else 0.0
    features["total_is_null"] = 1 if total_float is None else 0

    # Z-scores
    vendor = fields.get("vendor")
    if total_float is not None and vendor and vendor in vendor_stats:
        vs = vendor_stats[vendor]
        std = vs["std"] if vs["std"] > 0 else 1.0
        features["total_zscore_vendor"] = (total_float - vs["mean"]) / std
    else:
        features["total_zscore_vendor"] = 0.0

    if total_float is not None:
        gs = global_stats.get("std", 50.0)
        gs = gs if gs > 0 else 1.0
        features["total_zscore_global"] = (total_float - global_stats.get("mean", 0)) / gs
    else:
        features["total_zscore_global"] = 0.0

    # Vendor features
    features["vendor_code"] = vendor_encoding.get(vendor, -1)
    features["vendor_known"] = 1 if vendor and vendor in vendor_encoding else 0

    # Date features
    date_str = fields.get("date")
    date_feats = _extract_date_features(date_str)
    features.update(date_feats)

    # Null count
    null_count = sum(1 for k in ("vendor", "date", "total") if not fields.get(k))
    features["field_null_count"] = null_count

    # Total pattern features
    if total_str:
        features["total_has_cents"] = 1 if re.search(r'\.\d{2}$', total_str) else 0
        features["total_round_dollar"] = 1 if re.search(r'\.00$', total_str) else 0
        features["total_digit_count"] = len(re.findall(r'\d', total_str))
    else:
        features["total_has_cents"] = 0
        features["total_round_dollar"] = 0
        features["total_digit_count"] = 0

    # --- NEW: Benford's Law features ---
    try:
        if total_float is not None and total_float > 0:
            leading_str = str(abs(total_float)).lstrip("0").lstrip(".")
            leading_digit = int(leading_str[0]) if leading_str else 0
            features["benford_leading_digit"] = leading_digit
            # Surprise score: -log2(expected probability) for this digit
            if 1 <= leading_digit <= 9:
                expected_prob = _BENFORD_EXPECTED[leading_digit - 1]
                # Compare to vendor distribution if available
                if vendor and vendor in vendor_stats:
                    vendor_benford = vendor_stats[vendor].get("benford", _BENFORD_EXPECTED)
                    vendor_prob = vendor_benford[leading_digit - 1]
                    if vendor_prob > 1e-10:
                        features["benford_surprise"] = -math.log2(vendor_prob)
                    else:
                        features["benford_surprise"] = 10.0
                else:
                    features["benford_surprise"] = -math.log2(expected_prob)
            else:
                features["benford_surprise"] = 0.0
        else:
            features["benford_leading_digit"] = 0
            features["benford_surprise"] = 0.0
    except Exception:
        features["benford_leading_digit"] = 0
        features["benford_surprise"] = 0.0

    # --- NEW: Cents anomaly features ---
    try:
        if total_float is not None:
            cents = round((total_float % 1) * 100)
            features["cents_value"] = cents
            features["cents_is_zero"] = 1 if cents == 0 else 0
            features["cents_is_round_quarter"] = 1 if cents in (0, 25, 50, 75) else 0
            features["cents_is_repeating"] = 1 if (cents > 0 and cents // 10 == cents % 10) else 0
        else:
            features["cents_value"] = 0
            features["cents_is_zero"] = 0
            features["cents_is_round_quarter"] = 0
            features["cents_is_repeating"] = 0
    except Exception:
        features["cents_value"] = 0
        features["cents_is_zero"] = 0
        features["cents_is_round_quarter"] = 0
        features["cents_is_repeating"] = 0

    # --- NEW: Cross features ---
    try:
        vendor_code = features.get("vendor_code", -1)
        tf = features.get("total_float", 0.0)
        # vendor*total interaction
        features["vendor_total_interaction"] = float(vendor_code * tf) if vendor_code >= 0 else 0.0
        # abs z-score
        features["total_abs_zscore"] = abs(features.get("total_zscore_vendor", 0.0))
        # log(total)
        features["total_log"] = math.log1p(tf) if tf > 0 else 0.0
    except Exception:
        features["vendor_total_interaction"] = 0.0
        features["total_abs_zscore"] = 0.0
        features["total_log"] = 0.0

    return features


def _parse_total(total_str: str | None) -> float | None:
    """Parse a total string to float."""
    if not total_str:
        return None
    try:
        return float(total_str.replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def _extract_date_features(date_str: str | None) -> dict:
    """Extract date decomposition features."""
    if not date_str:
        return {
            "date_month": 0,
            "date_day_of_week": 0,
            "date_is_weekend": 0,
            "date_valid": 0,
        }
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return {
            "date_month": dt.month,
            "date_day_of_week": dt.weekday(),
            "date_is_weekend": 1 if dt.weekday() >= 5 else 0,
            "date_valid": 1,
        }
    except (ValueError, TypeError):
        return {
            "date_month": 0,
            "date_day_of_week": 0,
            "date_is_weekend": 0,
            "date_valid": 0,
        }


FIELD_FEATURE_NAMES = [
    "total_float", "total_is_null",
    "total_zscore_vendor", "total_zscore_global",
    "vendor_code", "vendor_known",
    "date_month", "date_day_of_week", "date_is_weekend", "date_valid",
    "field_null_count",
    "total_has_cents", "total_round_dollar", "total_digit_count",
    # Benford
    "benford_leading_digit", "benford_surprise",
    # Cents anomaly
    "cents_value", "cents_is_zero", "cents_is_round_quarter", "cents_is_repeating",
    # Cross features
    "vendor_total_interaction", "total_abs_zscore", "total_log",
]
