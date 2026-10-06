"""
India Supply-Route Disruption Intelligence Service — Feature 8
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
    "Moderate": "#eab308",
    "High": "#f97316",
    "Critical": "#ef4444"
}

class SupplyRouteService:
    def __init__(self):
        self.model_paths = [
            os.path.join(PROJECT_ROOT, "models", "feature8_india_supply_route_risk_model.joblib"),
            os.path.join(BASE_DIR, "models", "feature8_india_supply_route_risk_model.joblib"),
        ]
        self.data_dirs = [
            os.path.join(PROJECT_ROOT, "src", "data"),
            os.path.join(PROJECT_ROOT, "data"),
            os.path.join(BASE_DIR, "models"),
        ]
        self._model = None
        self._summary = None
        self._models_data = None
        self._confusion = None
        self._roc = None
        self._analysis = None
        self._samples = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        for path in self.model_paths:
            if os.path.exists(path):
                try:
                    self._model = joblib.load(path)
                    print(f"[SupplyRouteService] Loaded model from: {path}")
                    return self._model
                except Exception as e:
                    print(f"[SupplyRouteService] Failed to load model from {path}: {e}")
        return None

    def _load_json(self, filename: str) -> Optional[Any]:
        for d in self.data_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[SupplyRouteService] Error reading {path}: {e}")
        return None

    def get_summary(self) -> Dict[str, Any]:
        if not self._summary:
            self._summary = self._load_json("supply_route_summary.json") or {}
        return self._summary

    def get_models(self) -> Dict[str, Any]:
        if not self._models_data:
            self._models_data = self._load_json("supply_route_models.json") or {}
        return self._models_data

    def get_confusion_matrix(self) -> Dict[str, Any]:
        if not self._confusion:
            self._confusion = self._load_json("supply_route_confusion.json") or {}
        return self._confusion

    def get_roc_data(self) -> Dict[str, Any]:
        if not self._roc:
            self._roc = self._load_json("supply_route_roc.json") or {}
        return self._roc

    def get_route_analysis(self) -> Dict[str, Any]:
        if not self._analysis:
            self._analysis = self._load_json("supply_route_analysis.json") or {}
        return self._analysis

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self._samples = self._load_json("supply_route_samples.json") or []
        return self._samples

    def predict(self,
                route_disruption: float,
                shipping_delay: float,
                freight_cost_change: float,
                trade_volume_exposure: float,
                india_import_exposure: float,
                india_export_exposure: float,
                energy_route_exposure: float,
                commodity_exposure: float,
                conflict_intensity: float) -> Dict[str, Any]:
        """
        Runs real-time inference using the serialized syllabus champion model.
        Does NOT retrain the model.
        """
        t0 = time.perf_counter()
        model = self._load_model()
        if model is None:
            return {
                "success": False,
                "error": "Supply-route risk model is currently unavailable. Please ensure model is serialized at models/feature8_india_supply_route_risk_model.joblib."
            }

        input_df = pd.DataFrame([{
            "route_disruption": float(route_disruption),
            "shipping_delay": float(shipping_delay),
            "freight_cost_change": float(freight_cost_change),
            "trade_volume_exposure": float(trade_volume_exposure),
            "india_import_exposure": float(india_import_exposure),
            "india_export_exposure": float(india_export_exposure),
            "energy_route_exposure": float(energy_route_exposure),
            "commodity_exposure": float(commodity_exposure),
            "conflict_intensity": float(conflict_intensity),
        }])

        try:
            pred_class = model.predict(input_df)[0]
            probabilities = {}
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                classes = model.classes_
                probabilities = {c: round(float(p) * 100, 2) for c, p in zip(classes, probs)}

            latency = round((time.perf_counter() - t0) * 1000, 2)

            # Key drivers diagnosis
            drivers = []
            if float(route_disruption) >= 70:
                drivers.append("Critical Chokepoint Transit Severance / Lane Closure")
            if float(shipping_delay) >= 14:
                drivers.append("Severe Voyage Deviation (>2 Weeks Port Arrival Lag)")
            if float(freight_cost_change) >= 80:
                drivers.append("Triple-Digit Maritime Freight & War Risk Insurance Surcharges")
            if float(india_import_exposure) >= 70 or float(energy_route_exposure) >= 70:
                drivers.append("High Inelastic Import / Strategic Energy Feedstock Exposure")
            if float(conflict_intensity) >= 75:
                drivers.append("Kinetic Military Action / Anti-Ship Missile Threats")
            if not drivers:
                drivers.append("Standard Maritime Corridor Operations & Absorbed Delays")

            # Route recommendation
            if pred_class in ["High", "Critical"]:
                rec = "Trigger strategic diversion protocols (Cape of Good Hope / alternate transshipment hubs); activate petroleum strategic reserves and hedge freight forward agreements."
            elif pred_class == "Moderate":
                rec = "Monitor AIS tracking and war-risk premiums closely. Buffer domestic inventories of critical raw materials."
            else:
                rec = "Maintain routine maritime monitoring. Normal insurance and shipping schedules apply."

            return {
                "success": True,
                "prediction": str(pred_class),
                "risk_color": TIER_COLORS.get(str(pred_class), "#eab308"),
                "probabilities": probabilities,
                "model_name": type(model.named_steps["model"]).__name__ if hasattr(model, "named_steps") else "Syllabus Champion",
                "latency_ms": latency,
                "key_drivers": drivers,
                "strategic_recommendation": rec,
                "inputs": {
                    "route_disruption": route_disruption,
                    "shipping_delay": shipping_delay,
                    "freight_cost_change": freight_cost_change,
                    "trade_volume_exposure": trade_volume_exposure,
                    "india_import_exposure": india_import_exposure,
                    "india_export_exposure": india_export_exposure,
                    "energy_route_exposure": energy_route_exposure,
                    "commodity_exposure": commodity_exposure,
                    "conflict_intensity": conflict_intensity,
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction computation failed: {str(e)}"
            }

supply_route_service = SupplyRouteService()
