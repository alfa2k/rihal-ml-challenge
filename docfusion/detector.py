"""Forgery detector with ensemble model + threshold optimization."""
from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .image_features import IMAGE_FEATURE_NAMES, extract_image_features
from .field_features import (
    FIELD_FEATURE_NAMES,
    extract_field_features,
    compute_vendor_stats,
    compute_global_stats,
    build_vendor_encoding,
)
from .utils import resolve_image_path

ALL_FEATURE_NAMES = IMAGE_FEATURE_NAMES + FIELD_FEATURE_NAMES


class ForgeryDetector:
    """Ensemble-based document forgery detector with threshold optimization."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.iso_forest = None
        self.vendor_stats = {}
        self.global_stats = {}
        self.vendor_encoding = {}
        self.threshold = 0.5
        self.feature_importances_ = None
        self._use_stacking = False

    def _build_feature_vector(self, record: dict, data_dir: str) -> np.ndarray:
        """Build a single feature vector from a record."""
        fields = record.get("fields", {})
        image_path = resolve_image_path(data_dir, record)

        img_feats = extract_image_features(image_path)
        field_feats = extract_field_features(
            fields, self.vendor_stats, self.global_stats, self.vendor_encoding
        )

        vec = []
        for name in IMAGE_FEATURE_NAMES:
            vec.append(img_feats.get(name, 0.0))
        for name in FIELD_FEATURE_NAMES:
            vec.append(field_feats.get(name, 0.0))
        return np.array(vec, dtype=np.float64)

    def _create_model(self, use_stacking: bool):
        """Create the appropriate model based on data size."""
        if use_stacking:
            self._use_stacking = True
            gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
            )
            self.model = StackingClassifier(
                estimators=[("gb", gb), ("rf", rf)],
                final_estimator=LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=42
                ),
                cv=5,
                passthrough=False,
            )
        else:
            self._use_stacking = False
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )

    def train(self, records: list[dict], data_dir: str) -> None:
        """Train the forgery detector on labeled records."""
        self.vendor_stats = compute_vendor_stats(records)
        self.global_stats = compute_global_stats(records)
        self.vendor_encoding = build_vendor_encoding(records)

        X = []
        y = []
        for rec in records:
            vec = self._build_feature_vector(rec, data_dir)
            X.append(vec)
            label = rec.get("label", {})
            y.append(label.get("is_forged", 0))

        X = np.array(X)
        y = np.array(y)

        # Check if we have enough samples per class for stacking
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = min(counts) if len(counts) > 1 else 0
        use_stacking = min_class_count >= 5

        self._create_model(use_stacking)

        # Fit IsolationForest for anomaly score meta-feature
        try:
            self.iso_forest = IsolationForest(
                n_estimators=100, contamination="auto", random_state=42
            )
            iso_scores = self.iso_forest.fit(X).decision_function(X)
            X_aug = np.column_stack([X, iso_scores])
        except Exception:
            self.iso_forest = None
            X_aug = X

        # Scale and fit
        X_scaled = self.scaler.fit_transform(X_aug)
        self.model.fit(X_scaled, y)

        # Extract feature importances from GB base estimator
        try:
            if self._use_stacking:
                gb_estimator = self.model.estimators_[0]
                self.feature_importances_ = gb_estimator.feature_importances_
            else:
                self.feature_importances_ = self.model.feature_importances_
        except Exception:
            self.feature_importances_ = None

        # Optimize threshold via CV if enough data
        if use_stacking and len(y) >= 10:
            try:
                self.threshold = self._optimize_threshold(X_scaled, y)
            except Exception:
                self.threshold = 0.5
        else:
            self.threshold = 0.5

    def _optimize_threshold(self, X: np.ndarray, y: np.ndarray) -> float:
        """Find optimal threshold by sweeping for max F1 on CV predictions."""
        try:
            cv = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)
            proba = cross_val_predict(self.model, X, y, cv=cv, method="predict_proba")
            if proba.shape[1] == 2:
                scores = proba[:, 1]
            else:
                scores = proba[:, 0]

            best_f1 = 0.0
            best_thresh = 0.5
            for thresh in np.arange(0.2, 0.81, 0.05):
                preds = (scores >= thresh).astype(int)
                tp = ((preds == 1) & (y == 1)).sum()
                fp = ((preds == 1) & (y == 0)).sum()
                fn = ((preds == 0) & (y == 1)).sum()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            return float(best_thresh)
        except Exception:
            return 0.5

    def _augment_features(self, X: np.ndarray) -> np.ndarray:
        """Add IsolationForest anomaly score if available."""
        if self.iso_forest is not None:
            try:
                iso_scores = self.iso_forest.decision_function(X)
                return np.column_stack([X, iso_scores])
            except Exception:
                pass
        return X

    def predict(self, records: list[dict], data_dir: str) -> list[int]:
        """Predict forgery labels using optimized threshold."""
        X = []
        for rec in records:
            vec = self._build_feature_vector(rec, data_dir)
            X.append(vec)

        X = np.array(X)
        X = self._augment_features(X)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)
        if proba.shape[1] == 2:
            scores = proba[:, 1]
        else:
            scores = proba[:, 0]

        return (scores >= self.threshold).astype(int).tolist()

    def predict_proba(self, records: list[dict], data_dir: str) -> list[float]:
        """Predict forgery probability for records."""
        X = []
        for rec in records:
            vec = self._build_feature_vector(rec, data_dir)
            X.append(vec)

        X = np.array(X)
        X = self._augment_features(X)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        if proba.shape[1] == 2:
            return proba[:, 1].tolist()
        return proba[:, 0].tolist()

    def save(self, model_dir: str) -> None:
        """Save model artifacts to a directory."""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        joblib.dump({
            "vendor_stats": self.vendor_stats,
            "global_stats": self.global_stats,
            "vendor_encoding": self.vendor_encoding,
            "threshold": self.threshold,
            "feature_importances": self.feature_importances_,
            "use_stacking": self._use_stacking,
        }, os.path.join(model_dir, "meta.joblib"))
        if self.iso_forest is not None:
            joblib.dump(self.iso_forest, os.path.join(model_dir, "iso_forest.joblib"))

    def load(self, model_dir: str) -> None:
        """Load model artifacts from a directory."""
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        meta = joblib.load(os.path.join(model_dir, "meta.joblib"))
        self.vendor_stats = meta["vendor_stats"]
        self.global_stats = meta["global_stats"]
        self.vendor_encoding = meta["vendor_encoding"]
        self.threshold = meta.get("threshold", 0.5)
        self.feature_importances_ = meta.get("feature_importances", None)
        self._use_stacking = meta.get("use_stacking", False)

        iso_path = os.path.join(model_dir, "iso_forest.joblib")
        if os.path.exists(iso_path):
            self.iso_forest = joblib.load(iso_path)
        else:
            self.iso_forest = None
