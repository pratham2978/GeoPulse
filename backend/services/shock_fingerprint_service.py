"""
Feature 9: Geopolitical Shock Fingerprinting & Historical Conflict Comparison Service
Loads PCA and K-Means models and computes real-time conflict fingerprinting and similarity matching.
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

SHOCK_FEATURES = [
    "energy_shock",
    "trade_disruption",
    "shipping_disruption",
    "commodity_shock",
    "financial_stress"
]

class ShockFingerprintService:
    def __init__(self):
        self.model_paths = [
            os.path.join(PROJECT_ROOT, "models", "feature9_geopolitical_shock_fingerprint.joblib"),
            os.path.join(BASE_DIR, "models", "feature9_geopolitical_shock_fingerprint.joblib"),
        ]
        self.data_dirs = [
            os.path.join(PROJECT_ROOT, "src", "data"),
            os.path.join(PROJECT_ROOT, "data"),
            os.path.join(BASE_DIR, "models"),
        ]
        self._artifact = None
        self._summary = None
        self._conflicts = None
        self._clusters = None
        self._pca = None
        self._silhouette = None
        self._presets = None

    def _load_artifact(self):
        if self._artifact is not None:
            return self._artifact
        for path in self.model_paths:
            if os.path.exists(path):
                try:
                    self._artifact = joblib.load(path)
                    print(f"[ShockFingerprintService] Loaded artifact from: {path}")
                    return self._artifact
                except Exception as e:
                    print(f"[ShockFingerprintService] Error loading artifact from {path}: {e}")
        return None

    def _load_json(self, filename: str) -> Optional[Any]:
        for d in self.data_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[ShockFingerprintService] Error reading {path}: {e}")
        return None

    def get_summary(self) -> Dict[str, Any]:
        if not self._summary:
            self._summary = self._load_json("shock_fingerprint_summary.json") or {}
        return self._summary

    def get_conflicts(self) -> List[Dict[str, Any]]:
        if not self._conflicts:
            self._conflicts = self._load_json("shock_fingerprint_conflicts.json") or []
        return self._conflicts

    def get_clusters(self) -> List[Dict[str, Any]]:
        if not self._clusters:
            self._clusters = self._load_json("shock_fingerprint_clusters.json") or []
        return self._clusters

    def get_pca(self) -> List[Dict[str, Any]]:
        if not self._pca:
            self._pca = self._load_json("shock_fingerprint_pca.json") or []
        return self._pca

    def get_silhouette(self) -> List[Dict[str, Any]]:
        if not self._silhouette:
            self._silhouette = self._load_json("shock_fingerprint_silhouette.json") or []
        return self._silhouette

    def get_presets(self) -> List[Dict[str, Any]]:
        if not self._presets:
            self._presets = self._load_json("shock_fingerprint_presets.json") or []
        return self._presets

    def predict(self,
                energy_shock: float,
                trade_disruption: float,
                shipping_disruption: float,
                commodity_shock: float,
                financial_stress: float) -> Dict[str, Any]:
        """
        Projects a shock profile into PCA space and assigns it to a learned cluster.
        Finds the closest historical conflicts using Euclidean distance in standardized shock space.
        """
        t0 = time.perf_counter()
        artifact = self._load_artifact()
        if artifact is None:
            return {
                "success": False,
                "error": "Shock fingerprinting model unavailable. Ensure models/feature9_geopolitical_shock_fingerprint.joblib exists."
            }

        scaler = artifact["scaler"]
        pca = artifact["pca"]
        kmeans = artifact["kmeans"]
        cluster_profiles = artifact["cluster_profiles"]

        raw_df = pd.DataFrame([{
            "energy_shock": float(energy_shock),
            "trade_disruption": float(trade_disruption),
            "shipping_disruption": float(shipping_disruption),
            "commodity_shock": float(commodity_shock),
            "financial_stress": float(financial_stress)
        }])

        try:
            # 1. Scale
            scaled_vector = scaler.transform(raw_df[SHOCK_FEATURES])

            # 2. Project to PCA 2D
            pca_coords = pca.transform(scaled_vector)[0]
            pc1 = round(float(pca_coords[0]), 3)
            pc2 = round(float(pca_coords[1]), 3)

            # 3. K-Means Cluster Assignment
            cluster_id = int(kmeans.predict(scaled_vector)[0])
            profile = next((p for p in cluster_profiles if p.get("cluster_id") == cluster_id), {})

            # Center distance
            center_scaled = kmeans.cluster_centers_[cluster_id]
            dist_to_center = round(float(np.linalg.norm(scaled_vector[0] - center_scaled)), 3)

            # 4. Compare with historical conflicts in dataset
            conflicts = self.get_conflicts()
            matched_conflicts = []
            if conflicts:
                for conf in conflicts:
                    shocks_dict = conf.get("shocks", {})
                    impacts_dict = conf.get("impacts", {})
                    c_df = pd.DataFrame([{
                        "energy_shock": float(shocks_dict.get("energy_shock", 50)),
                        "trade_disruption": float(shocks_dict.get("trade_disruption", 50)),
                        "shipping_disruption": float(shocks_dict.get("shipping_disruption", 50)),
                        "commodity_shock": float(shocks_dict.get("commodity_shock", 50)),
                        "financial_stress": float(shocks_dict.get("financial_stress", 50))
                    }])
                    c_scaled = scaler.transform(c_df[SHOCK_FEATURES])[0]
                    dist = float(np.linalg.norm(scaled_vector[0] - c_scaled))
                    similarity = round((1.0 / (1.0 + (dist * 0.35))) * 100, 1)
                    matched_conflicts.append({
                        "conflict": conf.get("conflict", "Historical Event"),
                        "period": conf.get("period", ""),
                        "cluster_name": conf.get("cluster_name", ""),
                        "cluster_color": conf.get("cluster_color", "#06b6d4"),
                        "pc1": conf.get("pc1", 0),
                        "pc2": conf.get("pc2", 0),
                        "distance": round(dist, 2),
                        "similarity_score": similarity,
                        "oil_price_change": impacts_dict.get("oil_price_change", 0),
                        "inflation_change": impacts_dict.get("inflation_change", 0),
                        "gdp_growth_change": impacts_dict.get("gdp_growth_change", 0),
                    })

                matched_conflicts.sort(key=lambda x: x["distance"])

            # 5. Projected Macro Impact Indicators
            shock_mean = float(raw_df[SHOCK_FEATURES].values.mean())
            projected_oil = round(float(15.0 + (energy_shock * 0.85) + (shipping_disruption * 0.35)), 1)
            projected_inflation = round(float(0.8 + (commodity_shock * 0.045) + (energy_shock * 0.035)), 2)
            projected_gdp_growth = round(float(-0.2 - (shock_mean * 0.03)), 2)
            projected_trade_growth = round(float(-0.5 - (trade_disruption * 0.065)), 2)

            latency = round((time.perf_counter() - t0) * 1000, 2)

            return {
                "success": True,
                "cluster_id": cluster_id,
                "cluster_name": profile.get("name", f"Cluster {cluster_id}"),
                "cluster_tag": profile.get("tag", "Geopolitical Stress"),
                "cluster_desc": profile.get("description", ""),
                "color": profile.get("color", "#38bdf8"),
                "badge_bg": profile.get("badgeBg", "bg-cyan-500/20 text-cyan-400 border-cyan-500/40"),
                "distance_to_center": dist_to_center,
                "pca": {
                    "pc1": pc1,
                    "pc2": pc2,
                    "pc1_variance": 73.27,
                    "pc2_variance": 14.30
                },
                "matched_historical_conflicts": matched_conflicts[:5],
                "projected_macro_impact": {
                    "oil_price_change_pct": projected_oil,
                    "inflation_change_pct": projected_inflation,
                    "gdp_growth_change_pct": projected_gdp_growth,
                    "trade_growth_change_pct": projected_trade_growth
                },
                "latency_ms": latency,
                "inputs": {
                    "energy_shock": energy_shock,
                    "trade_disruption": trade_disruption,
                    "shipping_disruption": shipping_disruption,
                    "commodity_shock": commodity_shock,
                    "financial_stress": financial_stress
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Fingerprint prediction failed: {str(e)}"
            }

shock_fingerprint_service = ShockFingerprintService()
