"""
Sentiment Prediction Module for Feature 3: News Sentiment Analysis
Loads the pre-trained best supervised ML model (Random Forest / SVM / Logistic Regression)
to classify unseen geopolitical news articles into Positive, Negative, or Neutral.

STRICT REQUIREMENT: Uses saved model, NO retraining on prediction.
Syllabus compliance: Supervised classical ML pipeline.
"""

import os
import re
import time
import joblib
import numpy as np
from typing import Dict, Any, Optional

ORDERED_SENTIMENTS = ["Negative", "Neutral", "Positive"]

class SentimentPredictor:
    def __init__(self, models_dir: Optional[str] = None):
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend', 'models', 'feature3_sentiment_model.joblib'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'feature3_sentiment_model.joblib'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'feature3_sentiment_model.joblib'),
            os.path.join('backend', 'models', 'feature3_sentiment_model.joblib'),
            os.path.join('models', 'feature3_sentiment_model.joblib'),
        ]
        if models_dir:
            possible_paths.insert(0, os.path.join(models_dir, 'feature3_sentiment_model.joblib'))

        self.model_path = None
        for p in possible_paths:
            norm_p = os.path.abspath(p)
            if os.path.exists(norm_p):
                self.model_path = norm_p
                break

        self.model = None
        self.model_name = "Random Forest"
        self._load_artifact()

    def _load_artifact(self):
        if not self.model_path or not os.path.exists(self.model_path):
            print(f"[SentimentPredictor] Warning: Model artifact not found at {self.model_path}")
            return

        try:
            self.model = joblib.load(self.model_path)
            # Inspect pipeline or classifier type
            core_model = self.model
            if hasattr(self.model, "named_steps") and "model" in self.model.named_steps:
                core_model = self.model.named_steps["model"]

            m_type = type(core_model).__name__
            if "LogisticRegression" in m_type:
                self.model_name = "Logistic Regression"
            elif "SVC" in m_type or "Calibrated" in m_type:
                self.model_name = "SVM"
            elif "RandomForest" in m_type:
                self.model_name = "Random Forest"
            elif "DecisionTree" in m_type:
                self.model_name = "Decision Tree"
            elif "KNeighbors" in m_type:
                self.model_name = "k-NN"
            else:
                self.model_name = m_type
        except Exception as e:
            print(f"[SentimentPredictor] Error loading model: {e}")
            self.model = None

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.strip()
        t = re.sub(r'\s+', ' ', t)
        return t

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self, headline: str = "", text: str = "") -> Dict[str, Any]:
        """
        Classifies sentiment of a news headline and article text using the saved model.
        Returns:
            - sentiment: "Positive" | "Negative" | "Neutral"
            - model: str
            - status: "Successfully Classified" | "classified"
            - probabilities: dict of class -> float (e.g. {"Positive": 0.724, "Neutral": 0.181, "Negative": 0.095})
            - confidence_percent: float (e.g. 72.4)
            - latency_ms: float
        """
        start_time = time.perf_counter()

        if not self.model:
            return {
                "success": False,
                "error": "Sentiment model is currently unavailable. Please train the model first.",
                "status": "unavailable",
                "sentiment": None,
                "model": self.model_name,
                "probabilities": None
            }

        h_clean = self.clean_text(headline or "")
        t_clean = self.clean_text(text or "")
        combined = f"{h_clean} {t_clean}".strip()

        if not combined:
            return {
                "success": False,
                "error": "Please enter a news article before analysis.",
                "status": "validation_error",
                "sentiment": None,
                "model": self.model_name,
                "probabilities": None
            }

        # Predict using saved trained Pipeline
        try:
            predicted_label = self.model.predict([combined])[0]
        except Exception as e:
            return {
                "success": False,
                "error": f"Inference error: {str(e)}",
                "status": "error",
                "sentiment": None,
                "model": self.model_name,
                "probabilities": None
            }

        probabilities = None
        probabilities_percent = None
        confidence = 0.0

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([combined])[0]
            classes = getattr(self.model, "classes_", ORDERED_SENTIMENTS)
            probabilities = {}
            probabilities_percent = {}
            for c_name, p in zip(classes, probs):
                p_float = float(p)
                probabilities[c_name] = round(p_float, 4)
                probabilities_percent[c_name] = round(p_float * 100, 1)

            confidence = probabilities_percent.get(predicted_label, round(float(np.max(probs)) * 100, 1))

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "success": True,
            "sentiment": str(predicted_label),
            "model": self.model_name,
            "status": "Successfully Classified",
            "probabilities": probabilities,
            "probabilities_percent": probabilities_percent,
            "confidence": confidence,
            "latency_ms": latency_ms
        }

# Global singleton
_sentiment_predictor = None

def get_sentiment_predictor() -> SentimentPredictor:
    global _sentiment_predictor
    if _sentiment_predictor is None:
        _sentiment_predictor = SentimentPredictor()
    return _sentiment_predictor

if __name__ == "__main__":
    p = get_sentiment_predictor()
    print("Model available:", p.is_available())
    test_article = "Diplomatic talks between the two countries have reduced tensions and markets reacted positively to the announcement."
    print("Test prediction:", p.predict(text=test_article))
