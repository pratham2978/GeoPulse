"""
India Trade Dependency & Country Risk Service — Feature 7
Serves saved K-Means Clustering artifacts, elbow analysis, and live cluster assignment to the FastAPI backend.
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

RISK_GROUP_COLORS = {
    "Critical Exposure": "#f43f5e",
    "Higher Exposure": "#f97316",
    "Moderate Exposure": "#f59e0b",
    "Lower Exposure": "#10b981"
}

class TradeDependencyService:
    def __init__(self):
        self.model_paths = [
            os.path.join(PROJECT_ROOT, "models", "feature7_india_trade_dependency_kmeans.joblib"),
            os.path.join(BASE_DIR, "models", "feature7_india_trade_dependency_kmeans.joblib"),
        ]
        self.data_dirs = [
            os.path.join(PROJECT_ROOT, "src", "data"),
            os.path.join(PROJECT_ROOT, "data"),
            os.path.join(BASE_DIR, "models"),
        ]
        self._artifact = None
        self._summary = None
        self._clusters = None
        self._elbow = None
        self._scatter = None
        self._countries = None
        self._samples = None

    def _load_artifact(self):
        if self._artifact is not None:
            return self._artifact
        for path in self.model_paths:
            if os.path.exists(path):
                try:
                    self._artifact = joblib.load(path)
                    print(f"[TradeDependencyService] Loaded clustering artifact from: {path}")
                    return self._artifact
                except Exception as e:
                    print(f"[TradeDependencyService] Error loading artifact from {path}: {e}")
        return None

    def _load_json(self, filename: str) -> Optional[Any]:
        for d in self.data_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"[TradeDependencyService] Error reading {path}: {e}")
        return None

    def get_summary(self) -> Dict[str, Any]:
        if not self._summary:
            self._summary = self._load_json("trade_dependency_summary.json") or {}
        return self._summary

    def get_clusters(self) -> List[Dict[str, Any]]:
        if not self._clusters:
            self._clusters = self._load_json("trade_dependency_clusters.json") or []
        return self._clusters

    def get_elbow(self) -> List[Dict[str, Any]]:
        if not self._elbow:
            self._elbow = self._load_json("trade_dependency_elbow.json") or []
        return self._elbow

    def get_scatter(self) -> List[Dict[str, Any]]:
        if not self._scatter:
            self._scatter = self._load_json("trade_dependency_scatter.json") or []
        return self._scatter

    def get_country_analysis(self) -> List[Dict[str, Any]]:
        if not self._countries:
            self._countries = self._load_json("trade_dependency_country_analysis.json") or []
        return self._countries

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self._samples = self._load_json("trade_dependency_samples.json") or []
        return self._samples

    def predict(self,
                india_import_dependency: float,
                india_export_dependency: float,
                energy_dependency: float,
                commodity_dependency: float,
                trade_value: float,
                trade_disruption: float,
                shipping_disruption: float,
                strategic_route_exposure: float) -> Dict[str, Any]:
        """
        Assigns a new country/observation to a learned K-Means cluster.
        Does NOT retrain K-Means.
        """
        t0 = time.perf_counter()
        artifact = self._load_artifact()
        if artifact is None:
            return {
                "success": False,
                "error": "Clustering model is currently unavailable. Ensure models/feature7_india_trade_dependency_kmeans.joblib exists."
            }

        kmeans = artifact["model"]
        scaler = artifact["scaler"]
        features = artifact["features"]
        cluster_to_risk = artifact["cluster_to_risk"]

        input_df = pd.DataFrame([{
            "india_import_dependency": float(india_import_dependency),
            "india_export_dependency": float(india_export_dependency),
            "energy_dependency": float(energy_dependency),
            "commodity_dependency": float(commodity_dependency),
            "trade_value": float(trade_value),
            "trade_disruption": float(trade_disruption),
            "shipping_disruption": float(shipping_disruption),
            "strategic_route_exposure": float(strategic_route_exposure)
        }])

        try:
            # Transform features to standardized space
            input_scaled = scaler.transform(input_df[features])

            # Predict cluster
            cluster = int(kmeans.predict(input_scaled)[0])
            risk_group = cluster_to_risk.get(cluster, cluster_to_risk.get(str(cluster), f"Cluster {cluster}"))
            risk_color = RISK_GROUP_COLORS.get(risk_group, "#38bdf8")

            # Euclidean distances to all 4 cluster centers in standardized space
            centers_scaled = kmeans.cluster_centers_
            dists = np.linalg.norm(centers_scaled - input_scaled, axis=1)
            dist_to_chosen = round(float(dists[cluster]), 3)

            cluster_distances = {
                cluster_to_risk.get(i, f"Cluster {i}"): round(float(dists[i]), 3)
                for i in range(len(dists))
            }

            latency = round((time.perf_counter() - t0) * 1000, 2)

            # Qualitative drivers
            drivers = []
            if float(india_import_dependency) >= 30.0:
                drivers.append("High Indian Import Dependency & Supplier Concentration")
            if float(energy_dependency) >= 25.0:
                drivers.append("Critical Bilateral Crude/Gas Flow Dependence")
            if float(shipping_disruption) >= 7.0 or float(strategic_route_exposure) >= 8.0:
                drivers.append("Vulnerable Maritime Route & Chokepoint Transit")
            if float(india_export_dependency) >= 20.0:
                drivers.append("Significant Destination for Indian Outbound Merchandise")
            if not drivers:
                drivers.append("Balanced Trade Pattern with Resilient Logistics Corridors")

            return {
                "success": True,
                "cluster": cluster,
                "india_trade_risk_group": risk_group,
                "risk_color": risk_color,
                "distance_to_center": dist_to_chosen,
                "cluster_distances": cluster_distances,
                "model_used": f"K-Means Clustering (k={kmeans.n_clusters})",
                "latency_ms": latency,
                "key_drivers": drivers,
                "inputs": {
                    "india_import_dependency": india_import_dependency,
                    "india_export_dependency": india_export_dependency,
                    "energy_dependency": energy_dependency,
                    "commodity_dependency": commodity_dependency,
                    "trade_value": trade_value,
                    "trade_disruption": trade_disruption,
                    "shipping_disruption": shipping_disruption,
                    "strategic_route_exposure": strategic_route_exposure
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Clustering assignment failed: {str(e)}"
            }

trade_dependency_service = TradeDependencyService()
