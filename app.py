"""DocFusion Web UI - Production-grade Streamlit dashboard for receipt forensics."""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docfusion.utils import load_jsonl, is_dummy_image
from docfusion.ocr import run_ocr
from docfusion.fields import extract_fields_with_fallback
from docfusion.image_features import extract_image_features, compute_ela, IMAGE_FEATURE_NAMES
from docfusion.field_features import extract_field_features, FIELD_FEATURE_NAMES
from docfusion.detector import ForgeryDetector, ALL_FEATURE_NAMES
from docfusion.explainer import generate_forensic_report, compute_category_scores

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="DocFusion - Forgery Detection", page_icon="🔍", layout="wide")

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.12) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(236,72,153,0.08) 0%, transparent 50%);
    animation: heroGlow 8s ease-in-out infinite alternate;
}
@keyframes heroGlow {
    0% { transform: translateX(-5%) translateY(-5%); }
    100% { transform: translateX(5%) translateY(5%); }
}
.hero h1 { position: relative; color: #fff; margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; }
.hero p  { position: relative; color: rgba(255,255,255,0.55); margin: 0.4rem 0 0; font-size: 0.95rem; font-weight: 400; }
.hero .badge {
    position: relative;
    display: inline-block;
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 0.6rem;
    border: 1px solid rgba(99,102,241,0.3);
}

/* ── Cards ── */
.card {
    background: rgba(30,30,47,0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.7rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.card:hover { transform: translateY(-1px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
.card .card-label {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.card .card-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1.2;
}
.card .card-sub {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.35);
    margin-top: 0.2rem;
}

/* Card accent variants */
.card-green  { border-left: 3px solid #34d399; }
.card-red    { border-left: 3px solid #f87171; }
.card-blue   { border-left: 3px solid #60a5fa; }
.card-purple { border-left: 3px solid #a78bfa; }
.card-amber  { border-left: 3px solid #fbbf24; }

/* ── Verdict Banner ── */
.verdict-banner {
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin: 0.5rem 0 1rem;
    border: 1px solid;
}
.verdict-genuine {
    background: rgba(52,211,153,0.08);
    border-color: rgba(52,211,153,0.25);
}
.verdict-suspicious {
    background: rgba(248,113,113,0.08);
    border-color: rgba(248,113,113,0.25);
}
.verdict-banner .verdict-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    opacity: 0.7;
}
.verdict-banner .verdict-text {
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0.2rem 0;
}
.verdict-genuine .verdict-label, .verdict-genuine .verdict-text { color: #34d399; }
.verdict-suspicious .verdict-label, .verdict-suspicious .verdict-text { color: #f87171; }

/* ── Confidence Bar ── */
.conf-bar-track {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin: 0.5rem 0;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.6s ease;
}

/* ── Feature Mini Bars ── */
.feat-row {
    display: flex;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.feat-row:last-child { border-bottom: none; }
.feat-name {
    flex: 0 0 140px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
}
.feat-bar-track {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border-radius: 4px;
    height: 6px;
    margin: 0 0.8rem;
    overflow: hidden;
}
.feat-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
}
.feat-val {
    flex: 0 0 60px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.6);
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* ── Section Headers ── */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.section-head .icon {
    width: 32px; height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    background: rgba(99,102,241,0.15);
}
.section-head h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.section-head p {
    margin: 0;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.35);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(30,30,47,0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.1rem;
    color: rgba(255,255,255,0.5);
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,15,30,0.95);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.sidebar-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.3);
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-online  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.status-offline { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: rgba(255,255,255,0.3);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
.empty-state h4 { color: rgba(255,255,255,0.5); margin: 0 0 0.5rem; }
.empty-state p { font-size: 0.85rem; max-width: 400px; margin: 0 auto; line-height: 1.5; }

/* ── Pipeline Viz ── */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1rem 0 1.5rem;
    flex-wrap: wrap;
}
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    background: rgba(30,30,47,0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.6);
    font-weight: 500;
}
.pipeline-step .step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: rgba(99,102,241,0.2);
    color: #a5b4fc;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
}
.pipeline-arrow {
    color: rgba(255,255,255,0.15);
    font-size: 1.2rem;
    margin: 0 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
    <h1>DocFusion</h1>
    <p>Advanced Document Forgery Detection & Forensic Analysis Platform</p>
    <div class="badge">69 FEATURES &bull; ENSEMBLE ML &bull; AI EXPLAINABILITY</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.markdown('<div class="sidebar-title">CONFIGURATION</div>', unsafe_allow_html=True)
MODEL_DIR = st.sidebar.text_input("Model directory", value="./tmp_work/model")
THRESHOLD_OVERRIDE = st.sidebar.slider(
    "Decision threshold", 0.1, 0.9, 0.5, 0.05,
    help="Probability cutoff for flagging a receipt as suspicious"
)


@st.cache_resource
def auto_train_model(train_dir: str, work_dir: str):
    """Auto-train model if not already trained (for cloud deployment)."""
    from solution import DocFusionSolution
    sol = DocFusionSolution()
    return sol.train(train_dir, work_dir)


@st.cache_resource
def load_detector(model_dir: str):
    det = ForgeryDetector()
    det.load(model_dir)
    return det


# Auto-train if no model exists but training data is available
if not os.path.exists(os.path.join(MODEL_DIR, "model.joblib")):
    train_dir = "./dummy_data/train"
    if os.path.exists(os.path.join(train_dir, "train.jsonl")):
        with st.sidebar.status("Training model...", expanded=True) as status:
            st.write("First run — training on sample data...")
            try:
                work_dir = os.path.dirname(MODEL_DIR) or "./tmp_work"
                auto_train_model(train_dir, work_dir)
                status.update(label="Model trained!", state="complete")
            except Exception as e:
                status.update(label=f"Training failed: {e}", state="error")

detector = None
if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "model.joblib")):
    try:
        detector = load_detector(MODEL_DIR)
        st.sidebar.markdown(
            '<span class="status-pill status-online">&#9679; Model Loaded</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.markdown("")
        model_type = "Stacking Ensemble" if detector._use_stacking else "GradientBoosting"
        info_items = [
            ("Type", model_type),
            ("Features", str(len(ALL_FEATURE_NAMES))),
            ("Threshold", f"{detector.threshold:.3f}"),
            ("Vendors", str(len(detector.vendor_encoding))),
        ]
        for lbl, val in info_items:
            st.sidebar.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:0.25rem 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.82rem;">'
                f'<span style="color:rgba(255,255,255,0.4)">{lbl}</span>'
                f'<span style="color:#e2e8f0;font-weight:600">{val}</span></div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")
        detector = None
else:
    st.sidebar.markdown(
        '<span class="status-pill status-offline">&#9679; No Model</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("")
    st.sidebar.code(
        "cd ML && python3 -c \"\nfrom solution import DocFusionSolution\n"
        "s = DocFusionSolution()\ns.train('./dummy_data/train', './tmp_work')\"",
        language="bash",
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.72rem;color:rgba(255,255,255,0.2);text-align:center;padding:0.5rem 0">'
    'DocFusion v2.0 &bull; Ensemble ML Pipeline</div>',
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Chart helpers
# ──────────────────────────────────────────────
_BG = "#0e1117"
_GRID = "#1e1e2f"
_TEXT = "#a0aec0"
_WHITE = "#e2e8f0"


def _style_ax(ax, title=None):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    if title:
        ax.set_title(title, color=_WHITE, fontsize=10, fontweight=600, pad=10)


def make_donut_gauge(value: float, label: str, color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2.8, 2.8), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor(_BG)
    ax.pie(
        [value, 1 - value],
        colors=[color, "#1e1e2f"],
        startangle=90,
        wedgeprops={"width": 0.28, "edgecolor": _BG, "linewidth": 2},
    )
    ax.text(0, 0.05, f"{value:.0%}", ha="center", va="center",
            fontsize=24, fontweight="bold", color="white", family="Inter")
    ax.text(0, -0.32, label, ha="center", va="center",
            fontsize=8, color=_TEXT, fontweight=600, family="Inter",
            style="italic" if label == "SUSPICIOUS" else "normal")
    plt.tight_layout(pad=0.5)
    return fig


def make_radar_chart(category_scores: dict) -> plt.Figure:
    cats = list(category_scores.keys())
    vals = list(category_scores.values())
    n = len(cats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    vals_c = vals + [vals[0]]
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"polar": True})
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    # Gradient fill
    ax.plot(angles_c, vals_c, "o-", linewidth=2, color="#f87171", markersize=5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.fill(angles_c, vals_c, alpha=0.15, color="#f87171")

    # Reference rings
    for lv in (25, 50, 75):
        ax.plot(angles_c, [lv]*(n+1), "--", color=_GRID, linewidth=0.5, alpha=0.6)

    ax.set_xticks(angles)
    ax.set_xticklabels(cats, color=_WHITE, fontsize=8, fontweight=500)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"], color=_TEXT, fontsize=6)
    ax.spines["polar"].set_color(_GRID)
    ax.grid(color=_GRID, alpha=0.3)
    plt.tight_layout(pad=1)
    return fig


def make_ela_heatmap(img: Image.Image, quality: int = 90) -> plt.Figure:
    ela = compute_ela(img, quality=quality)
    ela_gray = ela.mean(axis=2)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(_BG)
    im = ax.imshow(ela_gray, cmap="inferno", vmin=0, vmax=max(ela_gray.max() * 0.8, 1))
    ax.axis("off")
    _style_ax(ax, f"ELA Q{quality}")
    plt.tight_layout(pad=0.5)
    return fig


def make_noise_residual(img: Image.Image) -> plt.Figure:
    gray = img.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=3))
    noise = np.array(gray, dtype=np.float32) - np.array(blurred, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(_BG)
    ax.imshow(noise, cmap="RdBu_r", vmin=-30, vmax=30)
    ax.axis("off")
    _style_ax(ax, "Noise Residual")
    plt.tight_layout(pad=0.5)
    return fig


def make_suspicious_overlay(img: Image.Image) -> plt.Figure:
    ela = compute_ela(img)
    ela_gray = ela.mean(axis=2)
    thresh = ela_gray.mean() + 2 * ela_gray.std()
    mask = ela_gray > thresh
    img_arr = np.array(img).copy()
    overlay = img_arr.copy()
    overlay[mask] = [255, 50, 50]
    blended = (0.55 * img_arr + 0.45 * overlay).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(_BG)
    ax.imshow(blended)
    ax.axis("off")
    _style_ax(ax, "Suspicious Regions")
    plt.tight_layout(pad=0.5)
    return fig


def make_importance_chart(importances, feat_names, top_n=15) -> plt.Figure:
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    names = [feat_names[i] for i in sorted_idx]
    vals = [importances[i] for i in sorted_idx]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top_n + 1))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    colors = plt.cm.cool(np.linspace(0.3, 0.8, top_n))
    ax.barh(range(top_n), vals[::-1], color=colors[::-1], edgecolor=_BG, height=0.65)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], color=_WHITE, fontsize=8, fontweight=500)
    ax.set_xlabel("Importance", color=_TEXT, fontsize=8)
    ax.tick_params(colors=_TEXT, labelsize=7)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=_GRID, alpha=0.3)
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────
def get_effective_threshold():
    if detector is not None:
        return THRESHOLD_OVERRIDE if THRESHOLD_OVERRIDE else detector.threshold
    return 0.5


def analyze_single(image_path, fields, det, data_dir):
    record = {"id": "analysis", "image_path": os.path.basename(image_path), "fields": fields}
    proba = det.predict_proba([record], data_dir)
    return proba[0], int(proba[0] >= get_effective_threshold())


def extract_all_features(image_path, fields):
    img_feats = extract_image_features(image_path)
    field_feats = extract_field_features(
        fields,
        detector.vendor_stats if detector else {},
        detector.global_stats if detector else {},
        detector.vendor_encoding if detector else {},
    )
    return img_feats, field_feats


def render_card(label, value, accent="blue", sub=None):
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="card card-{accent}">'
        f'<div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_verdict(is_forged, proba):
    cls = "suspicious" if is_forged else "genuine"
    text = "SUSPICIOUS" if is_forged else "GENUINE"
    st.markdown(
        f'<div class="verdict-banner verdict-{cls}">'
        f'<div class="verdict-label">Document Verdict</div>'
        f'<div class="verdict-text">{text}</div></div>',
        unsafe_allow_html=True,
    )
    bar_color = "#f87171" if is_forged else "#34d399"
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:{_TEXT}">'
        f'<span>Genuine</span><span>Forgery probability: {proba:.1%}</span><span>Forged</span></div>'
        f'<div class="conf-bar-track">'
        f'<div class="conf-bar-fill" style="width:{proba*100:.1f}%;background:{bar_color}"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_feature_bars(feats: dict, keys: list[tuple[str, str, float]]):
    """Render feature mini-bars. keys = [(display_name, feat_key, max_val), ...]"""
    html = ""
    for name, key, maxv in keys:
        val = feats.get(key, 0)
        pct = min(100, abs(float(val)) / max(maxv, 1e-6) * 100)
        html += (
            f'<div class="feat-row">'
            f'<div class="feat-name">{name}</div>'
            f'<div class="feat-bar-track"><div class="feat-bar-fill" style="width:{pct:.0f}%"></div></div>'
            f'<div class="feat-val">{float(val):.3f}</div>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


def section_header(icon, title, desc=None):
    desc_html = f'<p>{desc}</p>' if desc else ""
    st.markdown(
        f'<div class="section-head">'
        f'<div class="icon">{icon}</div>'
        f'<div><h3>{title}</h3>{desc_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Single Analysis  ",
    "  Batch Analysis  ",
    "  Model Dashboard  ",
    "  Forensic Deep Dive  ",
    "  Test Samples  ",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 : Single Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    uploaded = st.file_uploader(
        "Upload a receipt image", type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="single_upload",
    )

    if uploaded is None:
        st.markdown(
            '<div class="empty-state"><div class="icon">📄</div>'
            '<h4>Upload a receipt to begin</h4>'
            '<p>Drag and drop or click above to upload a receipt image. '
            'The system will extract fields, run forensic analysis, and generate an AI report.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        img = Image.open(tmp_path).convert("RGB")
        is_dummy = is_dummy_image(tmp_path)
        ocr_text = run_ocr(tmp_path)
        fields = extract_fields_with_fallback(ocr_text, None)

        # ── Row 1: Image + Verdict + Fields ──
        c1, c2 = st.columns([1.3, 1])

        with c1:
            section_header("📷", "Receipt Image")
            st.image(img, use_container_width=True)

        with c2:
            if detector is not None and not is_dummy:
                proba, is_forged = analyze_single(tmp_path, fields, detector, os.path.dirname(tmp_path))
                render_verdict(is_forged, proba)
            elif detector is None:
                st.info("Load a trained model to see forgery analysis.")

            section_header("📋", "Extracted Fields")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                render_card("Vendor", fields.get("vendor") or "N/A", "purple")
            with fc2:
                render_card("Date", fields.get("date") or "N/A", "blue")
            with fc3:
                render_card("Total", f"${fields.get('total', 'N/A')}" if fields.get("total") else "N/A", "green")

        # ── Row 2: ELA + Radar + Features ──
        if not is_dummy:
            section_header("🔬", "Forensic Analysis", "Error Level Analysis, risk profiling, and feature inspection")
            r1, r2, r3 = st.columns([1, 1, 1])

            with r1:
                ela_fig = make_ela_heatmap(img)
                st.pyplot(ela_fig)
                plt.close(ela_fig)

            with r2:
                if detector is not None:
                    img_feats, field_feats = extract_all_features(tmp_path, fields)
                    cat_scores = compute_category_scores(img_feats, field_feats)
                    radar_fig = make_radar_chart(cat_scores)
                    st.pyplot(radar_fig)
                    plt.close(radar_fig)

            with r3:
                img_feats = extract_image_features(tmp_path)
                render_feature_bars(img_feats, [
                    ("ELA Mean", "ela_mean", 30),
                    ("ELA Std", "ela_std", 20),
                    ("Noise Level", "noise_level", 25),
                    ("Edge Density", "edge_density", 0.5),
                    ("GLCM Contrast", "glcm_contrast", 50),
                    ("GLCM Energy", "glcm_energy", 0.5),
                    ("Color Uniform.", "color_uniformity", 80),
                    ("JPEG Grid", "jpeg_grid_ratio", 2),
                    ("Compression", "compression_ratio", 1),
                ])

        # ── Row 3: AI Report ──
        if detector is not None and not is_dummy:
            section_header("🤖", "AI Forensic Report", "Automated analysis explaining the model's reasoning")
            img_f, field_f = extract_all_features(tmp_path, fields)
            report = generate_forensic_report(img_f, field_f, proba, get_effective_threshold())
            st.markdown(report)

        # ── Collapsibles ──
        if not is_dummy:
            with st.expander("All extracted features"):
                img_feats = extract_image_features(tmp_path)
                for k, v in sorted(img_feats.items()):
                    st.text(f"{k}: {v}")
        if ocr_text:
            with st.expander("Raw OCR text"):
                st.code(ocr_text, language=None)

        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 : Batch Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    section_header("📊", "Batch Receipt Analysis", "Upload multiple receipts for bulk processing")

    batch_files = st.file_uploader(
        "Upload receipts", type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True, key="batch_upload",
    )

    if not batch_files:
        st.markdown(
            '<div class="empty-state"><div class="icon">📁</div>'
            '<h4>No files uploaded</h4>'
            '<p>Select multiple receipt images to analyze them all at once. '
            'Results can be exported as CSV or JSON.</p></div>',
            unsafe_allow_html=True,
        )
    elif detector is None:
        st.warning("Load a trained model first to analyze receipts.")
    else:
        results = []
        progress = st.progress(0, text="Starting analysis...")

        for idx, bf in enumerate(batch_files):
            progress.progress((idx + 1) / len(batch_files), text=f"Analyzing {bf.name} ({idx+1}/{len(batch_files)})")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(bf.read())
                tmp_path = tmp.name
            try:
                ocr_text = run_ocr(tmp_path)
                fields = extract_fields_with_fallback(ocr_text, None)
                proba, is_forged = analyze_single(tmp_path, fields, detector, os.path.dirname(tmp_path))
                results.append({
                    "File": bf.name,
                    "Vendor": fields.get("vendor", "-"),
                    "Date": fields.get("date", "-"),
                    "Total": fields.get("total", "-"),
                    "Probability": round(proba, 4),
                    "Verdict": "SUSPICIOUS" if is_forged else "GENUINE",
                })
            except Exception as e:
                results.append({
                    "File": bf.name, "Vendor": "-", "Date": "-", "Total": "-",
                    "Probability": 0, "Verdict": f"ERROR: {e}",
                })
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        progress.empty()

        if results:
            import pandas as pd
            n_sus = sum(1 for r in results if r["Verdict"] == "SUSPICIOUS")
            n_gen = sum(1 for r in results if r["Verdict"] == "GENUINE")

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                render_card("Total Scanned", str(len(results)), "blue")
            with mc2:
                render_card("Genuine", str(n_gen), "green")
            with mc3:
                render_card("Suspicious", str(n_sus), "red")
            with mc4:
                rate = n_sus / len(results) * 100 if results else 0
                render_card("Flag Rate", f"{rate:.0f}%", "amber")

            df = pd.DataFrame(results)
            st.dataframe(
                df.style.map(
                    lambda v: "color: #f87171" if v == "SUSPICIOUS" else ("color: #34d399" if v == "GENUINE" else ""),
                    subset=["Verdict"],
                ),
                use_container_width=True,
                height=min(400, 40 * len(results) + 60),
            )

            ec1, ec2, _ = st.columns([1, 1, 2])
            with ec1:
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "docfusion_results.csv", "text/csv", use_container_width=True)
            with ec2:
                st.download_button("Download JSON", json.dumps(results, indent=2),
                                   "docfusion_results.json", "application/json", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 : Model Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    if detector is None:
        st.markdown(
            '<div class="empty-state"><div class="icon">🤖</div>'
            '<h4>No model loaded</h4>'
            '<p>Train a model first to see performance metrics, feature importances, and pipeline details.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        section_header("🤖", "Model Overview")

        # Pipeline visualization
        st.markdown(
            '<div class="pipeline">'
            '<div class="pipeline-step"><div class="step-num">1</div>OCR + Preprocess</div>'
            '<div class="pipeline-arrow">→</div>'
            '<div class="pipeline-step"><div class="step-num">2</div>46 Image Features</div>'
            '<div class="pipeline-arrow">→</div>'
            '<div class="pipeline-step"><div class="step-num">3</div>23 Field Features</div>'
            '<div class="pipeline-arrow">→</div>'
            '<div class="pipeline-step"><div class="step-num">4</div>IsolationForest</div>'
            '<div class="pipeline-arrow">→</div>'
            '<div class="pipeline-step"><div class="step-num">5</div>Stacking Ensemble</div>'
            '<div class="pipeline-arrow">→</div>'
            '<div class="pipeline-step"><div class="step-num">6</div>Threshold → Verdict</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        model_type = "Stacking Ensemble" if detector._use_stacking else "GradientBoosting"
        with mc1:
            render_card("Architecture", model_type, "purple", "GB + RF → LR" if detector._use_stacking else "Single model")
        with mc2:
            render_card("Total Features", str(len(ALL_FEATURE_NAMES)), "blue", "46 image + 23 field")
        with mc3:
            render_card("Decision Threshold", f"{detector.threshold:.3f}", "amber", "Optimized via CV F1")
        with mc4:
            render_card("Known Vendors", str(len(detector.vendor_encoding)), "green",
                        ", ".join(list(detector.vendor_encoding.keys())[:3]) + "...")

        # Feature importance
        if detector.feature_importances_ is not None:
            section_header("📊", "Feature Importances", "Top features driving the model's decisions")
            importances = detector.feature_importances_
            feat_names = ALL_FEATURE_NAMES[:len(importances)]
            if len(feat_names) < len(importances):
                feat_names = feat_names + [f"anomaly_score_{i}" for i in range(len(importances) - len(feat_names))]

            fig = make_importance_chart(importances, feat_names, top_n=15)
            st.pyplot(fig)
            plt.close(fig)

        # Feature groups
        section_header("🗂️", "Feature Groups")
        groups = [
            ("Image Basics", "blue", [n for n in ALL_FEATURE_NAMES if n in ("file_size","width","height","aspect_ratio","pixel_count","compression_ratio")]),
            ("Pixel Statistics", "purple", [n for n in ALL_FEATURE_NAMES if n.startswith("pixel_")]),
            ("Error Level Analysis", "red", [n for n in ALL_FEATURE_NAMES if n.startswith("ela_")]),
            ("GLCM Texture", "amber", [n for n in ALL_FEATURE_NAMES if n.startswith("glcm_")]),
            ("Color Histogram", "green", [n for n in ALL_FEATURE_NAMES if n.startswith("hist_") or n == "color_uniformity"]),
            ("JPEG Grid", "blue", [n for n in ALL_FEATURE_NAMES if n.startswith("jpeg_")]),
            ("OCR Confidence", "purple", [n for n in ALL_FEATURE_NAMES if n.startswith("ocr_")]),
            ("Statistical", "red", [n for n in ALL_FEATURE_NAMES if "zscore" in n or n.startswith("total_") or n.startswith("benford_") or n.startswith("cents_")]),
            ("Vendor & Date", "green", [n for n in ALL_FEATURE_NAMES if n.startswith("vendor_") or n.startswith("date_") or n == "field_null_count"]),
        ]
        cols = st.columns(3)
        for i, (gname, accent, feats) in enumerate(groups):
            with cols[i % 3]:
                render_card(gname, f"{len(feats)} features", accent, ", ".join(feats[:4]) + ("..." if len(feats) > 4 else ""))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 : Forensic Deep Dive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    section_header("🔎", "Forensic Deep Dive", "Pixel-level analysis for document authenticity verification")

    forensic_file = st.file_uploader(
        "Upload image for forensic analysis",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="forensic_upload",
    )

    if forensic_file is None:
        st.markdown(
            '<div class="empty-state"><div class="icon">🔬</div>'
            '<h4>Upload an image for deep analysis</h4>'
            '<p>Get detailed forensic visualizations including multi-quality ELA, '
            'noise residuals, suspicious region detection, and an AI-generated report.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(forensic_file.read())
            forensic_path = tmp.name

        if is_dummy_image(forensic_path):
            st.info("Placeholder image detected. Upload a real receipt for forensic analysis.")
        else:
            fimg = Image.open(forensic_path).convert("RGB")

            # ── Row 1: Original / ELA / Noise ──
            st.markdown("##### Comparison Panel")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                st.image(fimg, caption="Original", use_container_width=True)
            with fc2:
                ela_fig = make_ela_heatmap(fimg)
                st.pyplot(ela_fig)
                plt.close(ela_fig)
            with fc3:
                noise_fig = make_noise_residual(fimg)
                st.pyplot(noise_fig)
                plt.close(noise_fig)

            # ── Row 2: Multi-quality ELA ──
            st.markdown("##### Multi-Quality ELA")
            eq_cols = st.columns(4)
            for col, q in zip(eq_cols, [75, 85, 90, 95]):
                with col:
                    efig = make_ela_heatmap(fimg, quality=q)
                    st.pyplot(efig)
                    plt.close(efig)

            # ── Row 3: Suspicious regions + metrics ──
            st.markdown("##### Anomaly Detection")
            sc1, sc2 = st.columns([1.5, 1])
            with sc1:
                overlay_fig = make_suspicious_overlay(fimg)
                st.pyplot(overlay_fig)
                plt.close(overlay_fig)
            with sc2:
                ela = compute_ela(fimg)
                ela_gray = ela.mean(axis=2)
                ela_t = ela_gray.mean() + 2 * ela_gray.std()
                sus_pct = (ela_gray > ela_t).mean() * 100

                render_card("Suspicious Area", f"{sus_pct:.1f}%", "red", "Pixels exceeding mean + 2\u03c3")
                render_card("ELA Mean", f"{ela_gray.mean():.2f}", "blue")
                render_card("ELA Std Dev", f"{ela_gray.std():.2f}", "purple")
                render_card("Detection Threshold", f"{ela_t:.2f}", "amber")

            # ── Row 4: Radar + AI Report ──
            if detector is not None:
                ocr_text_f = run_ocr(forensic_path)
                fields_f = extract_fields_with_fallback(ocr_text_f, None)
                proba_f, is_forged_f = analyze_single(forensic_path, fields_f, detector, os.path.dirname(forensic_path))
                img_feats_f, field_feats_f = extract_all_features(forensic_path, fields_f)

                st.markdown("---")
                rc1, rc2 = st.columns([1, 1.3])
                with rc1:
                    section_header("📡", "Risk Profile")
                    render_verdict(is_forged_f, proba_f)
                    cat_scores = compute_category_scores(img_feats_f, field_feats_f)
                    radar_fig = make_radar_chart(cat_scores)
                    st.pyplot(radar_fig)
                    plt.close(radar_fig)
                with rc2:
                    section_header("🤖", "AI Forensic Report")
                    report = generate_forensic_report(img_feats_f, field_feats_f, proba_f, get_effective_threshold())
                    st.markdown(report)

        try:
            os.unlink(forensic_path)
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 : Test Samples
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    section_header("🧪", "Test Samples Explorer", "Browse sample data with ground truth labels")

    test_dir = "./dummy_data/test"
    test_jsonl = os.path.join(test_dir, "test.jsonl")

    if not os.path.exists(test_jsonl):
        st.warning("No test data found at ./dummy_data/test/")
    else:
        records = load_jsonl(test_jsonl)
        sample_ids = [r["id"] for r in records]

        compare_mode = st.toggle("Comparison mode", help="Select two samples to compare side by side")

        if compare_mode:
            s1, s2 = st.columns(2)
            with s1:
                sel1 = st.selectbox("Sample A", sample_ids, key="cmp1")
            with s2:
                sel2 = st.selectbox("Sample B", sample_ids, index=min(1, len(sample_ids) - 1), key="cmp2")
            selections = [sel1, sel2]
        else:
            selections = [st.selectbox("Select a test sample", sample_ids)]

        for sel_id in selections:
            if not sel_id:
                continue

            st.markdown("---")
            rec = next(r for r in records if r["id"] == sel_id)
            image_path = os.path.join(test_dir, rec.get("image_path", ""))
            jsonl_fields = rec.get("fields", {})

            # Header with ground truth
            labels_path = os.path.join(test_dir, "labels.jsonl")
            gt_html = ""
            if os.path.exists(labels_path):
                labels = load_jsonl(labels_path)
                label = next((lb for lb in labels if lb["id"] == sel_id), None)
                if label:
                    actual = label.get("label", {})
                    if actual.get("is_forged", 0):
                        gt_html = (f' &nbsp;<span class="status-pill status-offline">'
                                   f'Ground Truth: FORGED ({actual.get("fraud_type","unknown")})</span>')
                    else:
                        gt_html = ' &nbsp;<span class="status-pill status-online">Ground Truth: GENUINE</span>'

            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.8rem">'
                f'<span style="font-size:1.1rem;font-weight:700;color:#e2e8f0">Sample {sel_id}</span>'
                f'{gt_html}</div>',
                unsafe_allow_html=True,
            )

            cl, cm, cr = st.columns([1, 1, 1.2])

            with cl:
                if os.path.exists(image_path) and not is_dummy_image(image_path):
                    simg = Image.open(image_path).convert("RGB")
                    st.image(simg, use_container_width=True)
                    ela_fig = make_ela_heatmap(simg)
                    st.pyplot(ela_fig)
                    plt.close(ela_fig)
                else:
                    st.info("Dummy/placeholder image")

            with cm:
                for key in ("vendor", "date", "total"):
                    val = jsonl_fields.get(key)
                    render_card(key.upper(), val if val else "N/A", "blue")

                if detector is not None:
                    proba_s = detector.predict_proba([rec], test_dir)[0]
                    is_forged_s = int(proba_s >= get_effective_threshold())
                    render_verdict(is_forged_s, proba_s)

            with cr:
                if detector is not None and os.path.exists(image_path) and not is_dummy_image(image_path):
                    img_fs, field_fs = extract_all_features(image_path, jsonl_fields)
                    cat_s = compute_category_scores(img_fs, field_fs)
                    radar_s = make_radar_chart(cat_s)
                    st.pyplot(radar_s)
                    plt.close(radar_s)

                    with st.expander("AI Forensic Report"):
                        proba_r = detector.predict_proba([rec], test_dir)[0]
                        report_r = generate_forensic_report(img_fs, field_fs, proba_r, get_effective_threshold())
                        st.markdown(report_r)
