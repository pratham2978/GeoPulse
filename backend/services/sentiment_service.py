"""
Sentiment Analysis Service for GeoPulse AI Backend
Feature 3: News Sentiment Analysis
Loads saved ML models, performance metrics, confusion matrices, ROC data, and performs live prediction.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.feature3.predict.predict_sentiment import get_sentiment_predictor

class SentimentService:
    def __init__(self):
        self.models_dir = os.path.join(BASE_DIR, 'models')
        self._summary = None
        self._comparison = None
        self._confusion_matrices = None
        self._roc_data = None
        self._samples = None

    def _load_json(self, filename: str) -> Optional[Any]:
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[SentimentService] Error reading {filename}: {e}")
        return None

    def get_summary(self) -> Dict[str, Any]:
        if not self._summary:
            self._summary = self._load_json('sentiment_summary.json') or {}
        return self._summary

    def get_models(self) -> Dict[str, Any]:
        if not self._comparison:
            self._comparison = self._load_json('sentiment_model_comparison.json') or {}
        return self._comparison

    def get_confusion_matrices(self) -> Dict[str, Any]:
        if not self._confusion_matrices:
            self._confusion_matrices = self._load_json('sentiment_confusion_matrices.json') or {}
        return self._confusion_matrices

    def get_roc_data(self) -> Dict[str, Any]:
        if not self._roc_data:
            self._roc_data = self._load_json('sentiment_roc_data.json') or {}
        return self._roc_data

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self._samples = self._load_json('sentiment_samples.json') or []
        return self._samples

    def predict(self, headline: str = "", text: str = "") -> Dict[str, Any]:
        predictor = get_sentiment_predictor()
        return predictor.predict(headline=headline, text=text)

# Global singleton
sentiment_service = SentimentService()
