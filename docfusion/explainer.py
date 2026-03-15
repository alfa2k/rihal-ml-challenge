"""AI Forensic Report Generator - rule-based natural language explanations."""
from __future__ import annotations

import math
from typing import Optional


def generate_forensic_report(
    image_features: dict,
    field_features: dict,
    forgery_probability: float,
    threshold: float = 0.5,
) -> str:
    """Generate a natural language forensic report explaining model findings.

    Analyzes feature values against known thresholds to produce
    human-readable explanations of why a receipt was flagged or cleared.
    """
    findings = []
    risk_factors = []
    clear_factors = []

    verdict = "SUSPICIOUS" if forgery_probability >= threshold else "GENUINE"
    confidence_pct = forgery_probability * 100

    # --- ELA Analysis ---
    ela_mean = image_features.get("ela_mean", 0)
    ela_std = image_features.get("ela_std", 0)
    ela_max = image_features.get("ela_max", 0)
    ela_diff = image_features.get("ela_diff_75_95", 0)
    ela_ratio = image_features.get("ela_ratio_75_95", 0)

    if ela_mean > 15:
        risk_factors.append(
            f"**High ELA residual** (mean={ela_mean:.1f}): Significant difference between "
            f"original and recompressed image suggests regions may have been edited at "
            f"different compression levels."
        )
    elif ela_mean > 8:
        findings.append(
            f"Moderate ELA residual (mean={ela_mean:.1f}) detected. Some variation is "
            f"normal but warrants closer inspection."
        )
    else:
        clear_factors.append(
            f"ELA analysis clean (mean={ela_mean:.1f}): Uniform compression levels "
            f"consistent with an unmodified document."
        )

    if ela_std > 12:
        risk_factors.append(
            f"**High ELA variance** (std={ela_std:.1f}): Uneven error levels across the "
            f"image indicate possible localized manipulation."
        )

    if ela_max > 100:
        risk_factors.append(
            f"**ELA hotspot detected** (max={ela_max:.0f}): At least one region shows "
            f"extreme compression artifacts, possibly a pasted or edited area."
        )

    if abs(ela_diff) > 5:
        risk_factors.append(
            f"**Multi-quality ELA anomaly**: Difference between Q75 and Q95 analysis "
            f"({ela_diff:.1f}) suggests inconsistent compression history."
        )

    # --- Noise Analysis ---
    noise = image_features.get("noise_level", 0)
    if noise > 15:
        risk_factors.append(
            f"**Elevated noise level** ({noise:.1f}): High-frequency noise may indicate "
            f"image processing or filter application to conceal edits."
        )
    elif noise < 2 and not image_features.get("is_dummy", 0):
        risk_factors.append(
            f"**Suspiciously low noise** ({noise:.1f}): Extremely clean image may have "
            f"been digitally generated or heavily processed."
        )

    # --- Texture Analysis ---
    glcm_contrast = image_features.get("glcm_contrast", 0)
    glcm_energy = image_features.get("glcm_energy", 0)
    if glcm_energy > 0.3:
        findings.append(
            f"High texture energy ({glcm_energy:.3f}) suggests uniform or synthetic regions."
        )
    if glcm_contrast < 1.0 and not image_features.get("is_dummy", 0):
        findings.append(
            f"Low texture contrast ({glcm_contrast:.2f}) may indicate artificially smooth areas."
        )

    # --- JPEG Grid Analysis ---
    jpeg_ratio = image_features.get("jpeg_grid_ratio", 0)
    if jpeg_ratio > 1.3:
        risk_factors.append(
            f"**JPEG grid anomaly** (ratio={jpeg_ratio:.2f}): Block boundary artifacts are "
            f"stronger than expected, suggesting double-compression or splicing."
        )
    elif jpeg_ratio > 1.1:
        findings.append(
            f"Mild JPEG grid pattern (ratio={jpeg_ratio:.2f}): Slight block boundary "
            f"emphasis, could be from normal re-saving."
        )

    # --- Color Analysis ---
    color_uniformity = image_features.get("color_uniformity", 0)
    if color_uniformity < 10 and not image_features.get("is_dummy", 0):
        risk_factors.append(
            f"**Low color diversity** ({color_uniformity:.1f}): Unusually uniform colors "
            f"may indicate a synthetically generated receipt."
        )

    # Check for skewness anomalies
    skews = [
        image_features.get(f"hist_skewness_{c}", 0) for c in ("r", "g", "b")
    ]
    max_skew = max(abs(s) for s in skews) if skews else 0
    if max_skew > 2.0:
        findings.append(
            f"Color distribution skew ({max_skew:.1f}) detected in one or more channels."
        )

    # --- Field Analysis ---
    zscore_vendor = field_features.get("total_zscore_vendor", 0)
    zscore_global = field_features.get("total_zscore_global", 0)
    total_float = field_features.get("total_float", 0)

    if abs(zscore_vendor) > 2.5:
        direction = "above" if zscore_vendor > 0 else "below"
        risk_factors.append(
            f"**Statistical outlier**: Total amount (${total_float:.2f}) is {abs(zscore_vendor):.1f} "
            f"standard deviations {direction} the vendor average. This is unusual for this vendor."
        )
    elif abs(zscore_vendor) > 1.5:
        direction = "above" if zscore_vendor > 0 else "below"
        findings.append(
            f"Total amount is {abs(zscore_vendor):.1f}σ {direction} vendor average \u2014 "
            f"somewhat unusual but within plausible range."
        )

    # --- Benford's Law ---
    benford_surprise = field_features.get("benford_surprise", 0)
    if benford_surprise > 4.0:
        risk_factors.append(
            f"**Benford's Law violation** (surprise={benford_surprise:.1f}): The leading "
            f"digit of the total is statistically unlikely for this vendor's transaction pattern."
        )

    # --- Cents Pattern ---
    cents_zero = field_features.get("cents_is_zero", 0)
    cents_repeating = field_features.get("cents_is_repeating", 0)
    if cents_zero:
        findings.append(
            "Round dollar amount (no cents) \u2014 uncommon for genuine receipts with tax."
        )
    if cents_repeating:
        findings.append(
            "Repeating cents digits detected \u2014 sometimes seen in fabricated amounts."
        )

    # --- Null Fields ---
    null_count = field_features.get("field_null_count", 0)
    if null_count >= 2:
        risk_factors.append(
            f"**Missing fields** ({null_count}/3): Multiple required fields are absent, "
            f"which may indicate a poorly fabricated document."
        )

    # --- Vendor ---
    vendor_known = field_features.get("vendor_known", 0)
    if not vendor_known:
        findings.append(
            "Unknown vendor \u2014 not seen in training data, limited statistical comparison available."
        )

    # --- Build Report ---
    lines = []
    lines.append(f"## Forensic Analysis Report")
    lines.append("")
    lines.append(f"**Verdict: {verdict}** (Forgery probability: {confidence_pct:.1f}%)")
    lines.append("")

    if risk_factors:
        lines.append(f"### Risk Factors ({len(risk_factors)} found)")
        lines.append("")
        for i, rf in enumerate(risk_factors, 1):
            lines.append(f"{i}. {rf}")
            lines.append("")

    if findings:
        lines.append(f"### Observations")
        lines.append("")
        for f in findings:
            lines.append(f"- {f}")
        lines.append("")

    if clear_factors:
        lines.append(f"### Clear Indicators")
        lines.append("")
        for cf in clear_factors:
            lines.append(f"- {cf}")
        lines.append("")

    # Summary
    lines.append("### Summary")
    lines.append("")
    if verdict == "SUSPICIOUS":
        lines.append(
            f"This document exhibits **{len(risk_factors)} risk factor(s)** that suggest "
            f"possible tampering. The combination of image-level and field-level anomalies "
            f"increases confidence in the forgery assessment. Manual review is recommended."
        )
    else:
        lines.append(
            f"This document appears authentic. Image compression patterns are consistent, "
            f"field values fall within expected ranges, and no significant manipulation "
            f"artifacts were detected."
        )
        if findings:
            lines.append(
                f" {len(findings)} minor observation(s) noted but insufficient to indicate forgery."
            )

    return "\n".join(lines)


def compute_category_scores(image_features: dict, field_features: dict) -> dict:
    """Compute normalized scores (0-100) per forensic category for radar chart.

    Returns dict with category names as keys and 0-100 scores as values.
    Higher score = more suspicious.
    """
    scores = {}

    # 1. ELA Integrity (based on ELA stats)
    ela_mean = image_features.get("ela_mean", 0)
    ela_std = image_features.get("ela_std", 0)
    ela_score = min(100, (ela_mean / 25.0) * 50 + (ela_std / 20.0) * 50)
    scores["ELA Integrity"] = max(0, ela_score)

    # 2. Compression Artifacts (JPEG grid + multi-ELA)
    jpeg_ratio = image_features.get("jpeg_grid_ratio", 0)
    ela_diff = abs(image_features.get("ela_diff_75_95", 0))
    comp_score = min(100, ((jpeg_ratio - 1.0) / 0.5) * 60 + (ela_diff / 10.0) * 40)
    scores["Compression"] = max(0, comp_score)

    # 3. Texture Anomaly (GLCM)
    glcm_energy = image_features.get("glcm_energy", 0)
    glcm_contrast = image_features.get("glcm_contrast", 0)
    texture_score = min(100, glcm_energy * 200 + max(0, (5 - glcm_contrast)) * 10)
    scores["Texture"] = max(0, texture_score)

    # 4. Noise Pattern
    noise = image_features.get("noise_level", 0)
    if noise > 15:
        noise_score = min(100, (noise - 15) / 15 * 100)
    elif noise < 2 and not image_features.get("is_dummy", 0):
        noise_score = min(100, (2 - noise) / 2 * 80)
    else:
        noise_score = 0
    scores["Noise"] = max(0, noise_score)

    # 5. Color Consistency
    color_uni = image_features.get("color_uniformity", 50)
    max_skew = max(
        abs(image_features.get(f"hist_skewness_{c}", 0)) for c in ("r", "g", "b")
    )
    color_score = min(100, max(0, (50 - color_uni) / 50 * 50) + max_skew * 15)
    scores["Color"] = max(0, color_score)

    # 6. Statistical Anomaly (field z-scores, Benford)
    zscore = abs(field_features.get("total_zscore_vendor", 0))
    benford = field_features.get("benford_surprise", 0)
    stat_score = min(100, zscore * 20 + benford * 10)
    scores["Statistics"] = max(0, stat_score)

    # 7. Field Integrity (nulls, unknown vendor, cents patterns)
    null_count = field_features.get("field_null_count", 0)
    vendor_known = field_features.get("vendor_known", 1)
    cents_zero = field_features.get("cents_is_zero", 0)
    field_score = min(100, null_count * 25 + (1 - vendor_known) * 20 + cents_zero * 15)
    scores["Fields"] = max(0, field_score)

    return scores
