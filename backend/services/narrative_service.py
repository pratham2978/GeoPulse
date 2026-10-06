"""
Narrative Classification Service for GeoPulse AI Backend
Feature 2: News & Narrative Classification
Interacts with saved ML models, metrics, and live predictor.
"""

import os
import sys
import json
from typing import Dict, Any, Optional

# Ensure ml/feature2/predict is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.feature2.predict.predict_narrative import get_narrative_predictor

class NarrativeService:
    def __init__(self):
        self.models_dir = os.path.join(BASE_DIR, 'models')
        self._summary = None
        self._comparison = None
        self._confusion_matrices = None
        self._roc_data = None
        self._samples = None

    def _load_json(self, filename: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[NarrativeService] Error reading {filename}: {e}")
        return None

    def get_summary(self) -> Dict[str, Any]:
        if not self._summary:
            self._summary = self._load_json('dataset_summary.json') or {}
        return self._summary

    def get_models(self) -> Dict[str, Any]:
        if not self._comparison:
            self._comparison = self._load_json('model_comparison.json') or {}
        return self._comparison

    def get_confusion_matrices(self) -> Dict[str, Any]:
        if not self._confusion_matrices:
            self._confusion_matrices = self._load_json('confusion_matrices.json') or {}
        return self._confusion_matrices

    def get_roc_data(self) -> Dict[str, Any]:
        if not self._roc_data:
            self._roc_data = self._load_json('roc_curve_data.json') or {}
        return self._roc_data

    def get_samples(self) -> list:
        if not self._samples:
            self._samples = self._load_json('sample_articles.json') or []
        return self._samples

    def predict(self, headline: str, text: str = "") -> Dict[str, Any]:
        predictor = get_narrative_predictor()
        return predictor.predict(headline, text)

# Global singleton
narrative_service = NarrativeService()
