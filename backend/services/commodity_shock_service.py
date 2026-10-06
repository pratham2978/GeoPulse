"""
India Oil & Commodity Shock Intelligence Service — Feature 6
Serves saved Multiple Linear Regression artifacts and live predictions to the FastAPI backend.
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

TIER_CONFIG = {
    "CRITICAL": {
        "label": "CRITICAL",
        "color": "#f43f5e",
        "description": "Severe macroeconomic shock. High current account deficit expansion, sharp currency depreciation risk, and domestic food/fuel CPI surge."
    },
    "HIGH": {
        "label": "HIGH",
        "color": "#f97316",
        "description": "Substantial commodity bill inflation. Noticeable pressure on refinery margins, fertilizer subsidies, and industrial input costs."
    },
    "MODERATE": {
        "label": "MODERATE",
        "color": "#f59e0b",
        "description": "Moderate trade friction absorbable via domestic strategic petroleum buffers and bilateral currency settlement arrangements."
    },
    "LOW": {
        "label": "LOW",
        "color": "#10b981",
        "description": "Benign global price movement and stable maritime logistics corridors with negligible domestic inflationary transmission."
    }
}

class CommodityShockService:
    def __init__(self):
        self.model_paths = [
            os.path.join(PROJECT_ROOT, "models", "feature6_india_commodity_shock_model.joblib"),
            os.path.join(BASE_DIR, "models", "feature6_india_commodity_shock_model.joblib"),
        ]
        self.data_dirs = [
            os.path.join(PROJECT_ROOT, "src", "data"),
            os.path.join(PROJECT_ROOT, "data"),
            os.path.join(BASE_DIR, "models"),
        ]
        self._model = None
        self._summary = None
        self._model_metrics = None
        self._coefficients = None
        self._avp = None
        self._country = None
        self._samples = None
        self._trends = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        for path in self.model_paths:
            if os.path.exists(path):
                try:
                    self._model = joblib.load(path)
                    print(f"[CommodityShockService] Loaded model from: {path}")
                    return self._model
                except Exception as e:
                    print(f"[CommodityShockService] Failed to load model from {path}: {e}")
        return None

    def _load_json(self, filename: str) -> Optional[Any]:
        for d in self.data_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[CommodityShockService] Error reading {path}: {e}")
        return None

    def get_summary(self) -> Dict[str, Any]:
        if not self._summary:
            self._summary = self._load_json("commodity_shock_summary.json") or {}
        return self._summary

    def get_model_metrics(self) -> Dict[str, Any]:
        if not self._model_metrics:
            self._model_metrics = self._load_json("commodity_shock_model.json") or {}
        return self._model_metrics

    def get_coefficients(self) -> List[Dict[str, Any]]:
        if not self._coefficients:
            self._coefficients = self._load_json("commodity_shock_coefficients.json") or []
        return self._coefficients

    def get_actual_vs_predicted(self) -> List[Dict[str, Any]]:
        if not self._avp:
            self._avp = self._load_json("commodity_shock_actual_vs_pred.json") or []
        return self._avp

    def get_country_analysis(self) -> List[Dict[str, Any]]:
        if not self._country:
            self._country = self._load_json("commodity_shock_country_analysis.json") or []
        return self._country

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self._samples = self._load_json("commodity_shock_samples.json") or []
        return self._samples

    def get_trends(self) -> List[Dict[str, Any]]:
        if not self._trends:
            self._trends = self._load_json("commodity_shock_trends.json") or []
        return self._trends

    def predict(self,
                crude_oil_change: float,
                natural_gas_change: float,
                gold_price_change: float,
                essential_commodity_change: float,
                trade_disruption: float,
                shipping_disruption: float,
                conflict_intensity: float,
                india_import_dependency: float) -> Dict[str, Any]:
        """
        Runs real-time inference using the serialized Multiple Linear Regression model.
        Does NOT retrain the model.
        """
        t0 = time.perf_counter()
        model = self._load_model()
        if model is None:
            return {
                "success": False,
                "error": "Commodity shock regression model is currently unavailable. Ensure models/feature6_india_commodity_shock_model.joblib exists."
            }

        input_df = pd.DataFrame([{
            "crude_oil_change": float(crude_oil_change),
            "natural_gas_change": float(natural_gas_change),
            "gold_price_change": float(gold_price_change),
            "essential_commodity_change": float(essential_commodity_change),
            "trade_disruption": float(trade_disruption),
            "shipping_disruption": float(shipping_disruption),
            "conflict_intensity": float(conflict_intensity),
            "india_import_dependency": float(india_import_dependency)
        }])

        try:
            raw_prediction = float(model.predict(input_df)[0])
            pred_score = round(max(0.0, min(100.0, raw_prediction)), 2)

            # Classify impact level based on empirical distribution thresholds
            if pred_score >= 75.0:
                impact_level = "CRITICAL"
            elif pred_score >= 55.0:
                impact_level = "HIGH"
            elif pred_score >= 35.0:
                impact_level = "MODERATE"
            else:
                impact_level = "LOW"

            cfg = TIER_CONFIG[impact_level]
            latency = round((time.perf_counter() - t0) * 1000, 2)

            # Key drivers identification
            drivers = []
            if float(crude_oil_change) >= 15.0:
                drivers.append("Severe Crude Import Bill Inflation")
            if float(essential_commodity_change) >= 15.0:
                drivers.append("High Domestic Food & Agri CPI Pressure")
            if float(shipping_disruption) >= 7.0 or float(trade_disruption) >= 7.0:
                drivers.append("Critical Chokepoint & Maritime Freight Surcharges")
            if float(gold_price_change) >= 10.0:
                drivers.append("Safe-Haven Gold Outflows & Current Account Pressure")
            if float(natural_gas_change) >= 20.0:
                drivers.append("Fertilizer & Industrial Energy Squeeze")
            if not drivers:
                drivers.append("Controlled International Price Pass-Through")

            return {
                "success": True,
                "predicted_impact": pred_score,
                "impact_level": impact_level,
                "impact_color": cfg["color"],
                "impact_description": cfg["description"],
                "model_used": "Multiple Linear Regression",
                "latency_ms": latency,
                "key_drivers": drivers,
                "inputs": {
                    "crude_oil_change": crude_oil_change,
                    "natural_gas_change": natural_gas_change,
                    "gold_price_change": gold_price_change,
                    "essential_commodity_change": essential_commodity_change,
                    "trade_disruption": trade_disruption,
                    "shipping_disruption": shipping_disruption,
                    "conflict_intensity": conflict_intensity,
                    "india_import_dependency": india_import_dependency
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction computation failed: {str(e)}"
            }

commodity_shock_service = CommodityShockService()
