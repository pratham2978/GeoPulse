"""
India Energy Supply Risk Intelligence Service — Feature 5
Serves saved classification artifacts, multi-class metrics, and live predictions to the FastAPI backend.
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TIER_COLORS = {
    "Low": "#10b981",
    "Moderate": "#f59e0b",
    "High": "#f97316",
    "Critical": "#f43f5e"
}

class IndiaEnergyService:
    def __init__(self):
        self.model_paths = [
            os.path.join(PROJECT_ROOT, "models", "feature5_india_energy_risk_model.joblib"),
            os.path.join(BASE_DIR, "models", "feature5_india_energy_risk_model.joblib"),
        ]
        self.data_dirs = [
            os.path.join(PROJECT_ROOT, "src", "data"),
            os.path.join(PROJECT_ROOT, "data"),
            os.path.join(BASE_DIR, "models"),
        ]
        self._model = None
        self._summary = None
        self._comparison = None
        self._confusion = None
        self._roc = None
        self._country = None
        self._samples = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        for path in self.model_paths:
            if os.path.exists(path):
                try:
                    self._model = joblib.load(path)
                    print(f"[IndiaEnergyService] Loaded model from: {path}")
                    return self._model
                except Exception as e:
                    print(f"[IndiaEnergyService] Failed to load model from {path}: {e}")
        return None

    def _load_json(self, filename: str) -> Optional[Any]:
        for d in self.data_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[IndiaEnergyService] Error reading {path}: {e}")
        return None

    def get_summary(self) -> Dict[str, Any]:
        if not self._summary:
            self._summary = self._load_json("india_energy_summary.json") or {}
        return self._summary

    def get_models(self) -> List[Dict[str, Any]]:
        if not self._comparison:
            self._comparison = self._load_json("india_energy_comparison.json") or []
        return self._comparison

    def get_confusion_matrix(self) -> Dict[str, Any]:
        if not self._confusion:
            self._confusion = self._load_json("india_energy_confusion_matrices.json") or {}
        return self._confusion

    def get_roc_data(self) -> Dict[str, Any]:
        if not self._roc:
            self._roc = self._load_json("india_energy_roc_data.json") or {}
        return self._roc

    def get_country_analysis(self) -> List[Dict[str, Any]]:
        if not self._country:
            self._country = self._load_json("india_energy_country_analysis.json") or []
        return self._country

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self._samples = self._load_json("india_energy_samples.json") or []
        return self._samples

    def predict(self,
                oil_import_dependency: float,
                oil_price_change: float,
                energy_supply_disruption: float,
                shipping_disruption: float,
                india_energy_exposure: float,
                strategic_route_exposure: float,
                commodity_price_change: float) -> Dict[str, Any]:
        """
        Runs real-time inference using the serialized syllabus champion model.
        Does NOT retrain the model.
        """
        t0 = time.perf_counter()
        model = self._load_model()
        if model is None:
            return {
                "success": False,
                "error": "Energy supply risk model is currently unavailable. Please ensure model is serialized at models/feature5_india_energy_risk_model.joblib."
            }

        input_df = pd.DataFrame([{
            "oil_import_dependency": float(oil_import_dependency),
            "oil_price_change": float(oil_price_change),
            "energy_supply_disruption": float(energy_supply_disruption),
            "shipping_disruption": float(shipping_disruption),
            "india_energy_exposure": float(india_energy_exposure),
            "strategic_route_exposure": float(strategic_route_exposure),
            "commodity_price_change": float(commodity_price_change),
        }])

        try:
            pred_class = model.predict(input_df)[0]
            probabilities = {}
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                classes = model.classes_
                probabilities = {c: round(float(p) * 100, 2) for c, p in zip(classes, probs)}

            latency = round((time.perf_counter() - t0) * 1000, 2)

            # Key drivers analysis
            drivers = []
            if shipping_disruption >= 70:
                drivers.append("Severe Maritime Chokepoint & AIS Route Disruption")
            if india_energy_exposure >= 15:
                drivers.append("High Direct Bilateral Energy Import Vulnerability")
            if oil_price_change >= 10:
                drivers.append("Substantial Global Benchmark Crude Price Spike")
            if energy_supply_disruption >= 60:
                drivers.append("Physical Extraction or Pipeline Infrastructure Outage")
            if not drivers:
                drivers.append("Standard Corridor Operations & Distributed Routing")

            return {
                "success": True,
                "prediction": str(pred_class),
                "risk_color": TIER_COLORS.get(str(pred_class), "#f59e0b"),
                "probabilities": probabilities,
                "model_name": type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else "Syllabus Champion",
                "latency_ms": latency,
                "key_drivers": drivers,
                "inputs": {
                    "oil_import_dependency": oil_import_dependency,
                    "oil_price_change": oil_price_change,
                    "energy_supply_disruption": energy_supply_disruption,
                    "shipping_disruption": shipping_disruption,
                    "india_energy_exposure": india_energy_exposure,
                    "strategic_route_exposure": strategic_route_exposure,
                    "commodity_price_change": commodity_price_change,
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction computation failed: {str(e)}"
            }

india_energy_service = IndiaEnergyService()
