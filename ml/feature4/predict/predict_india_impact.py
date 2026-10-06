"""
India Economic Impact Predictor — Feature 4
Loads the pre-trained Multiple Linear Regression model once.
Performs instant inference on new user inputs without retraining.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

FEATURE_COLS = [
    "oil_price_change",
    "commodity_price_change",
    "trade_disruption",
    "shipping_disruption",
    "india_trade_exposure",
    "india_energy_exposure",
    "market_volatility",
]

def impact_level(score: float) -> str:
    """Interpret the predicted 0-100 India economic impact score."""
    if score <= 30:
        return "LOW"
    if score <= 60:
        return "MODERATE"
    if score <= 80:
        return "HIGH"
    return "CRITICAL"

def impact_color(level: str) -> str:
    return {
        "LOW": "#10b981",
        "MODERATE": "#f59e0b",
        "HIGH": "#f97316",
        "CRITICAL": "#f43f5e",
    }.get(level, "#06b6d4")

class IndiaImpactPredictor:
    def __init__(self, models_dir: Optional[str] = None):
        possible = [
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "backend", "models",
                         "feature4_india_economic_impact_model.joblib"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "models",
                         "feature4_india_economic_impact_model.joblib"),
            os.path.join("backend", "models", "feature4_india_economic_impact_model.joblib"),
            os.path.join("models", "feature4_india_economic_impact_model.joblib"),
        ]
        if models_dir:
            possible.insert(0, os.path.join(models_dir, "feature4_india_economic_impact_model.joblib"))

        self.model_path = None
        for p in possible:
            norm = os.path.abspath(p)
            if os.path.exists(norm):
                self.model_path = norm
                break

        self.model = None
        self._load()

    def _load(self):
        if not self.model_path:
            return
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            print(f"[IndiaImpactPredictor] Load error: {e}")

    def is_available(self) -> bool:
        return self.model is not None

    def predict(self,
                oil_price_change: float,
                commodity_price_change: float,
                trade_disruption: float,
                shipping_disruption: float,
                india_trade_exposure: float,
                india_energy_exposure: float,
                market_volatility: float) -> Dict[str, Any]:
        start = time.perf_counter()

        if not self.model:
            return {
                "success": False,
                "error": "Model unavailable. Please train the model first.",
                "predicted_impact": None,
                "impact_level": None,
            }

        try:
            X = pd.DataFrame([{
                "oil_price_change": float(oil_price_change),
                "commodity_price_change": float(commodity_price_change),
                "trade_disruption": float(trade_disruption),
                "shipping_disruption": float(shipping_disruption),
                "india_trade_exposure": float(india_trade_exposure),
                "india_energy_exposure": float(india_energy_exposure),
                "market_volatility": float(market_volatility),
            }])
            score = float(self.model.predict(X)[0])
            score = round(max(0.0, min(100.0, score)), 2)
        except Exception as e:
            return {"success": False, "error": f"Inference error: {e}", "predicted_impact": None}

        lvl = impact_level(score)
        latency = round((time.perf_counter() - start) * 1000, 2)

        return {
            "success": True,
            "predicted_impact": score,
            "impact_level": lvl,
            "impact_color": impact_color(lvl),
            "model": "Multiple Linear Regression",
            "latency_ms": latency,
            "inputs": {
                "oil_price_change": oil_price_change,
                "commodity_price_change": commodity_price_change,
                "trade_disruption": trade_disruption,
                "shipping_disruption": shipping_disruption,
                "india_trade_exposure": india_trade_exposure,
                "india_energy_exposure": india_energy_exposure,
                "market_volatility": market_volatility,
            }
        }


_predictor: Optional[IndiaImpactPredictor] = None

def get_india_impact_predictor() -> IndiaImpactPredictor:
    global _predictor
    if _predictor is None:
        _predictor = IndiaImpactPredictor()
    return _predictor


if __name__ == "__main__":
    p = get_india_impact_predictor()
    print("Available:", p.is_available())
    result = p.predict(15, 8, 65, 70, 18, 22, 7.5)
    print("Prediction:", result)
