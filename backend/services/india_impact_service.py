"""
India Economic Impact Service — Feature 4
Serves saved regression artifacts and live predictions to the FastAPI backend.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, List

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.feature4.predict.predict_india_impact import get_india_impact_predictor


class IndiaImpactService:
    def __init__(self):
        self.models_dir = os.path.join(BASE_DIR, "models")
        self._summary    = None
        self._comparison = None
        self._samples    = None
        self._country    = None
        self._avp        = None   # actual vs predicted

    def _load_json(self, filename: str) -> Optional[Any]:
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[IndiaImpactService] Error reading {filename}: {e}")
        return None

    def get_summary(self) -> Dict[str, Any]:
        if not self._summary:
            self._summary = self._load_json("india_impact_summary.json") or {}
        return self._summary

    def get_models(self) -> Dict[str, Any]:
        if not self._comparison:
            self._comparison = self._load_json("india_impact_comparison.json") or {}
        return self._comparison

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self._samples = self._load_json("india_impact_samples.json") or []
        return self._samples

    def get_country_analysis(self) -> List[Dict[str, Any]]:
        if not self._country:
            self._country = self._load_json("india_impact_country_analysis.json") or []
        return self._country

    def get_actual_vs_predicted(self) -> List[Dict[str, Any]]:
        if not self._avp:
            self._avp = self._load_json("india_impact_actual_vs_pred.json") or []
        return self._avp

    def predict(self,
                oil_price_change: float,
                commodity_price_change: float,
                trade_disruption: float,
                shipping_disruption: float,
                india_trade_exposure: float,
                india_energy_exposure: float,
                market_volatility: float) -> Dict[str, Any]:
        predictor = get_india_impact_predictor()
        return predictor.predict(
            oil_price_change=oil_price_change,
            commodity_price_change=commodity_price_change,
            trade_disruption=trade_disruption,
            shipping_disruption=shipping_disruption,
            india_trade_exposure=india_trade_exposure,
            india_energy_exposure=india_energy_exposure,
            market_volatility=market_volatility,
        )


india_impact_service = IndiaImpactService()
