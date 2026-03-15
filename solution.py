"""DocFusion Solution - Entry point for the autograder harness."""

import os
import sys

# Ensure the ML directory is on the path so docfusion can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from docfusion.utils import load_jsonl, write_jsonl, resolve_image_path
from docfusion.ocr import run_ocr
from docfusion.fields import extract_fields_with_fallback
from docfusion.detector import ForgeryDetector


class DocFusionSolution:
    def train(self, train_dir: str, work_dir: str) -> str:
        """Train the forgery detection model.

        Args:
            train_dir: Path to directory containing train.jsonl and images/
            work_dir:  Scratch directory for writing model artifacts

        Returns:
            Path to the saved model directory
        """
        model_dir = os.path.join(work_dir, "model")
        os.makedirs(model_dir, exist_ok=True)

        train_jsonl = os.path.join(train_dir, "train.jsonl")
        records = load_jsonl(train_jsonl)

        detector = ForgeryDetector()
        detector.train(records, train_dir)
        detector.save(model_dir)

        return model_dir

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        """Run inference and write predictions to out_path.

        Args:
            model_dir: Path returned by train()
            data_dir:  Path to directory containing test.jsonl and images/
            out_path:  Path where predictions JSONL should be written
        """
        test_jsonl = os.path.join(data_dir, "test.jsonl")
        records = load_jsonl(test_jsonl)

        # Load trained model
        detector = ForgeryDetector()
        detector.load(model_dir)

        # Predict forgery
        predictions_forged = detector.predict(records, data_dir)

        # Build output
        results = []
        for rec, is_forged in zip(records, predictions_forged):
            # Extract fields: try OCR first, fall back to JSONL fields
            image_path = resolve_image_path(data_dir, rec)
            ocr_text = run_ocr(image_path)
            jsonl_fields = rec.get("fields", {})
            extracted = extract_fields_with_fallback(ocr_text, jsonl_fields)

            results.append({
                "id": rec["id"],
                "vendor": extracted.get("vendor"),
                "date": extracted.get("date"),
                "total": extracted.get("total"),
                "is_forged": int(is_forged),
            })

        write_jsonl(results, out_path)
