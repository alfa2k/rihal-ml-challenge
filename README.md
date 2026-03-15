# DocFusion - Intelligent Document Processing Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rihal-ml-challenge-ahmedf.streamlit.app)

**Live Demo:** [rihal-ml-challenge-ahmedf.streamlit.app](https://rihal-ml-challenge-ahmedf.streamlit.app)

An end-to-end document processing system that extracts structured fields from scanned receipts, detects forged documents using ensemble machine learning, and exposes results via an interactive forensic analysis dashboard.

## Architecture

```
ML/
├── solution.py              # DocFusionSolution class (autograder entry point)
├── app.py                   # Streamlit web dashboard (5 tabs)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
└── docfusion/               # Core library
    ├── ocr.py               # Tesseract OCR with adaptive preprocessing
    ├── fields.py            # Regex field extraction (vendor/date/total)
    ├── image_features.py    # ELA, GLCM texture, color stats, JPEG grid, OCR confidence
    ├── field_features.py    # Z-scores, Benford's Law, cents anomaly, cross features
    ├── detector.py          # Ensemble forgery classifier (Stacking + IsolationForest)
    ├── explainer.py         # AI forensic report generator + radar chart scoring
    └── utils.py             # JSONL I/O, image loading
```

## Setup

### Prerequisites
- Python 3.9+
- Tesseract OCR (`brew install tesseract` on macOS, `apt install tesseract-ocr` on Ubuntu)

### Install

```bash
cd ML
pip install -r requirements.txt
```

### Verify

```bash
python3 check_submission.py --submission . --verbose
```

## Usage

### Train & Predict (CLI)

```python
from solution import DocFusionSolution

sol = DocFusionSolution()
model_dir = sol.train("./dummy_data/train", "./tmp_work")
sol.predict(model_dir, "./dummy_data/test", "./tmp_work/predictions.jsonl")
```

### Web UI

```bash
# Train first (if no model exists)
python3 -c "from solution import DocFusionSolution; s = DocFusionSolution(); s.train('./dummy_data/train', './tmp_work')"

# Launch dashboard
streamlit run app.py
```

### Docker

```bash
docker build -t docfusion .
docker run -p 8501:8501 docfusion
```

Then open http://localhost:8501.

## Technical Approach

| Component | Choice | Rationale |
|-----------|--------|-----------|
| ML Model | Stacking Ensemble (GB + RF + LR) | Combines diverse learners for better generalization |
| Anomaly Detection | IsolationForest meta-feature | Catches out-of-distribution samples |
| Threshold | CV F1-optimized | Data-driven cutoff instead of fixed 0.5 |
| Image Processing | Pillow + NumPy | Smaller deps than OpenCV, sufficient for forensic analysis |
| OCR | pytesseract with preprocessing | Adaptive thresholding + denoising for better accuracy |
| Web UI | Streamlit (5-tab dashboard) | Interactive forensic analysis with custom CSS |
| Explainability | Rule-based forensic reports | Natural language explanations of model decisions |

### Feature Engineering (69 features)

**Image features (46):**
- Error Level Analysis at multiple quality levels (Q75/Q85/Q90/Q95)
- GLCM texture descriptors (contrast, energy, homogeneity, correlation)
- Color histogram statistics (skewness, kurtosis, IQR per channel)
- JPEG 8x8 block boundary artifact detection
- Per-channel pixel mean/std, edge density, noise level
- OCR confidence statistics (mean confidence, low-confidence ratio, text density)
- File metadata (size, dimensions, aspect ratio, compression ratio)

**Field features (23):**
- Per-vendor and global z-scores for statistical outlier detection
- Benford's Law analysis (leading digit surprise score)
- Cents anomaly detection (zero cents, round quarters, repeating digits)
- Cross features (vendor-total interaction, log-total, abs z-score)
- Date decomposition, null field count, digit patterns

### Forgery Detection Pipeline

1. **Preprocessing**: Adaptive threshold + median denoise for OCR
2. **Feature Extraction**: 46 image + 23 field features
3. **Anomaly Scoring**: IsolationForest appended as meta-feature
4. **Ensemble Prediction**: StackingClassifier (GradientBoosting + RandomForest → LogisticRegression)
5. **Threshold Optimization**: CV F1 sweep (0.2–0.8) for optimal decision boundary
6. **Explainability**: Rule-based AI forensic report with radar chart visualization

### Web Dashboard (5 Tabs)

1. **Single Analysis** - Upload receipt, get verdict with donut gauge, radar chart, ELA heatmap, AI report
2. **Batch Analysis** - Multi-file upload with progress bar, results table, CSV/JSON export
3. **Model Dashboard** - Feature importances, model info, feature group breakdown
4. **Forensic Deep Dive** - Side-by-side forensics (original/ELA/noise), multi-quality ELA, suspicious region overlay
5. **Test Samples** - Browse test data with comparison mode, ground truth display

## Levels Addressed

- **Level 1:** EDA notebook with data exploration, visualizations, and correlation analysis
- **Level 2:** OCR + regex field extraction with cascading fallback to JSONL fields
- **Level 3:** Ensemble forgery detector + production-grade Streamlit dashboard with AI explainability
- **Level 4:** Autograder-compatible `solution.py` with train/predict interface
- **Bonus:** Dockerfile for containerization, AI forensic reports, radar chart fingerprinting
