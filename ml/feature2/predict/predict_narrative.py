"""
Narrative Prediction Module for Feature 2: News & Narrative Classification
Loads the pre-trained best model and TF-IDF vectorizer to classify unseen geopolitical articles.
STRICT REQUIREMENT: Uses saved model, NO retraining on prediction.
"""

import os
import re
import time
import joblib
import numpy as np

ORDERED_CLASSES = [
    "Military Conflict",
    "Diplomatic / Political",
    "Energy Risk",
    "Trade & Shipping",
    "Economic Impact",
    "Humanitarian Impact"
]

class NarrativePredictor:
    def __init__(self, models_dir: str = None):
        if not models_dir:
            models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'models')
            if not os.path.exists(models_dir):
                models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

        self.model_path = os.path.join(models_dir, 'narrative_classifier.joblib')
        self.vec_path = os.path.join(models_dir, 'vectorizer.joblib')
        self.model = None
        self.vectorizer = None
        self.model_name = "Logistic Regression (Best Syllabus Classifier)"
        self._load_artifacts()

    def _load_artifacts(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.vec_path):
            raise FileNotFoundError(
                f"Model or Vectorizer artifacts not found at {self.model_path}. "
                "Please run train_models.py first."
            )
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vec_path)
        
        # Derive human-friendly model title
        m_type = type(self.model).__name__
        if "LogisticRegression" in m_type:
            self.model_name = "Logistic Regression (Optimal L2 Regularization)"
        elif "SVC" in m_type or "Calibrated" in m_type:
            self.model_name = "Support Vector Machine (Linear Kernel)"
        elif "RandomForest" in m_type:
            self.model_name = "Random Forest (Ensemble Classifier)"
        elif "DecisionTree" in m_type:
            self.model_name = "Decision Tree (Pruned)"
        elif "KNeighbors" in m_type:
            self.model_name = "k-Nearest Neighbors (k-NN)"

    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.lower()
        t = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def predict(self, headline: str, article_text: str = "") -> dict:
        """
        Predicts narrative category for a given news headline and optional article text.
        Returns:
            - predicted_narrative: str
            - confidence: float (0.0 to 100.0)
            - probabilities: dict of class -> float
            - model_used: str
            - status: str
            - latency_ms: float
        """
        start_time = time.perf_counter()

        headline_clean = self.clean_text(headline or "")
        article_clean = self.clean_text(article_text or "")
        combined = f"{headline_clean} {article_clean}".strip()

        if not combined:
            return {
                "success": False,
                "error": "Please enter a news headline or article text for classification.",
                "predicted_narrative": None,
                "confidence": 0.0,
                "probabilities": {},
                "model_used": self.model_name,
                "status": "Validation Failed"
            }

        # Transform using saved vectorizer
        features = self.vectorizer.transform([combined])

        # Prediction using saved model
        predicted_label = self.model.predict(features)[0]

        # Valid probability extraction
        probabilities = {}
        confidence = 0.0

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            classes = getattr(self.model, "classes_", ORDERED_CLASSES)
            for c_name, p in zip(classes, probs):
                probabilities[c_name] = round(float(p) * 100, 1)
            confidence = probabilities.get(predicted_label, round(float(np.max(probs)) * 100, 1))
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(features)[0]
            # Convert decision scores to normalized probabilities via softmax
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            classes = getattr(self.model, "classes_", ORDERED_CLASSES)
            for c_name, p in zip(classes, probs):
                probabilities[c_name] = round(float(p) * 100, 1)
            confidence = probabilities.get(predicted_label, 85.0)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "success": True,
            "predicted_narrative": predicted_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": self.model_name,
            "status": "Successfully Classified",
            "latency_ms": latency_ms,
            "features_extracted": int(features.nnz)
        }

# Global singleton instance
predictor = None

def get_narrative_predictor() -> NarrativePredictor:
    global predictor
    if predictor is None:
        predictor = NarrativePredictor()
    return predictor

if __name__ == '__main__':
    p = get_narrative_predictor()
    test_headline = "Container shipping operators divert vessels around African coast following missile alerts"
    test_text = "Global logistics forwarders reported extended transit voyages and rising container spot rates."
    res = p.predict(test_headline, test_text)
    print("Test Prediction Result:\n", res)
