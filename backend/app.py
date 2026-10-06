"""
FastAPI Backend Application for GeoPulse AI
Core Platform Engines:
- Feature 4: India Economic Impact Intelligence (Multiple Linear Regression)
- Feature 5: India Energy Supply Risk Intelligence (Supervised Multi-Class ML)
- Feature 7: India Trade Dependency & Country Risk (K-Means Clustering)
- Feature 9: Geopolitical Shock Fingerprinting & Historical Conflict Comparison (PCA + K-Means)
"""

import os
import sys

# Ensure backend directory is in python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.india_impact_service import india_impact_service
from services.india_energy_service import india_energy_service
from services.trade_dependency_service import trade_dependency_service
from services.shock_fingerprint_service import shock_fingerprint_service

app = FastAPI(
    title="GeoPulse AI — Global Conflict Impact Intelligence Platform API",
    description="High-performance geopolitical & macroeconomic intelligence API serving Features 4, 5, 7, and 9.",
    version="3.0.0"
)

# Enable CORS for frontend applications (e.g. Vite on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "GeoPulse AI Global Intelligence Platform",
        "version": "3.0.0",
        "active_features": [
            "Feature 4: India Economic Impact Intelligence",
            "Feature 5: India Energy Supply Risk Intelligence",
            "Feature 7: India Trade Dependency & Country Risk",
            "Feature 9: Geopolitical Shock Fingerprinting"
        ]
    }

# =============================================================================
# FEATURE 4: INDIA ECONOMIC IMPACT INTELLIGENCE API ROUTES
# =============================================================================

class IndiaImpactPredictRequest(BaseModel):
    oil_price_change: float = 10.0
    commodity_price_change: float = 5.0
    trade_disruption: float = 50.0
    shipping_disruption: float = 45.0
    india_trade_exposure: float = 10.0
    india_energy_exposure: float = 15.0
    market_volatility: float = 5.0

@app.get("/api/india-impact/summary")
@app.get("/india-impact/summary")
def get_india_impact_summary():
    """Returns dataset overview, train/test split, and target statistics for Feature 4."""
    return india_impact_service.get_summary()

@app.get("/api/india-impact/models")
@app.get("/india-impact/models")
def get_india_impact_models():
    """Returns SLR vs MLR comparison: MAE, RMSE, R², CV R², coefficients, overfitting analysis."""
    return india_impact_service.get_models()

@app.get("/api/india-impact/samples")
@app.get("/india-impact/samples")
def get_india_impact_samples():
    """Returns 3 sample events (high/medium/low impact) from the real dataset."""
    return india_impact_service.get_samples()

@app.get("/api/india-impact/country-analysis")
@app.get("/india-impact/country-analysis")
def get_india_impact_country_analysis():
    """Returns country-wise average India economic impact ranking from the real dataset."""
    return india_impact_service.get_country_analysis()

@app.get("/api/india-impact/actual-vs-predicted")
@app.get("/india-impact/actual-vs-predicted")
def get_india_impact_avp():
    """Returns actual vs predicted scatter data for Multiple Linear Regression."""
    return india_impact_service.get_actual_vs_predicted()

@app.post("/api/india-impact/predict")
@app.post("/india-impact/predict")
def predict_india_impact(req: IndiaImpactPredictRequest):
    """
    Live India Economic Impact Prediction:
    Runs the saved Multiple Linear Regression model.
    """
    result = india_impact_service.predict(
        oil_price_change=req.oil_price_change,
        commodity_price_change=req.commodity_price_change,
        trade_disruption=req.trade_disruption,
        shipping_disruption=req.shipping_disruption,
        india_trade_exposure=req.india_trade_exposure,
        india_energy_exposure=req.india_energy_exposure,
        market_volatility=req.market_volatility,
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=503 if "unavailable" in result.get("error", "") else 400,
            detail=result.get("error", "Prediction failed.")
        )
    return result

# =============================================================================
# FEATURE 5: INDIA ENERGY SUPPLY RISK INTELLIGENCE API ROUTES
# =============================================================================

class IndiaEnergyRiskPredictRequest(BaseModel):
    oil_import_dependency: float = 85.0
    oil_price_change: float = 12.0
    energy_supply_disruption: float = 70.0
    shipping_disruption: float = 65.0
    india_energy_exposure: float = 18.0
    strategic_route_exposure: float = 80.0
    commodity_price_change: float = 8.0

@app.get("/api/india-energy-risk/summary")
@app.get("/india-energy-risk/summary")
def get_india_energy_summary():
    """Returns dataset overview, multi-class distribution, and target statistics for Feature 5."""
    return india_energy_service.get_summary()

@app.get("/api/india-energy-risk/models")
@app.get("/india-energy-risk/models")
def get_india_energy_models():
    """Returns comparison across 5 syllabus classifiers: Accuracy, Precision, Recall, F1, CV F1, Overfitting Gap."""
    return india_energy_service.get_models()

@app.get("/api/india-energy-risk/confusion-matrix")
@app.get("/india-energy-risk/confusion-matrix")
def get_india_energy_confusion_matrix():
    """Returns multi-class confusion matrices for the 5 syllabus models."""
    return india_energy_service.get_confusion_matrix()

@app.get("/api/india-energy-risk/roc-data")
@app.get("/india-energy-risk/roc-data")
def get_india_energy_roc_data():
    """Returns multi-class One-vs-Rest ROC curve data (FPR, TPR, AUC) for models with probability estimates."""
    return india_energy_service.get_roc_data()

@app.get("/api/india-energy-risk/country-analysis")
@app.get("/india-energy-risk/country-analysis")
def get_india_energy_country_analysis():
    """Returns country-wise energy risk share and bilateral vulnerability exposure for India."""
    return india_energy_service.get_country_analysis()

@app.get("/api/india-energy-risk/samples")
@app.get("/india-energy-risk/samples")
def get_india_energy_samples():
    """Returns curated conflict and chokepoint preset scenarios (Hormuz, Red Sea, Nigeria, US Gulf)."""
    return india_energy_service.get_samples()

@app.post("/api/india-energy-risk/predict")
@app.post("/india-energy-risk/predict")
def predict_india_energy_risk(req: IndiaEnergyRiskPredictRequest):
    """
    Live India Energy Supply Risk Prediction:
    Runs the serialized syllabus champion model.
    """
    result = india_energy_service.predict(
        oil_import_dependency=req.oil_import_dependency,
        oil_price_change=req.oil_price_change,
        energy_supply_disruption=req.energy_supply_disruption,
        shipping_disruption=req.shipping_disruption,
        india_energy_exposure=req.india_energy_exposure,
        strategic_route_exposure=req.strategic_route_exposure,
        commodity_price_change=req.commodity_price_change,
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=503 if "unavailable" in result.get("error", "") else 400,
            detail=result.get("error", "Energy supply risk prediction failed.")
        )
    return result

# =============================================================================
# FEATURE 7: INDIA TRADE DEPENDENCY & COUNTRY RISK INTELLIGENCE API ROUTES
# =============================================================================

class TradeDependencyPredictRequest(BaseModel):
    india_import_dependency: float = 70.0
    india_export_dependency: float = 45.0
    energy_dependency: float = 65.0
    commodity_dependency: float = 60.0
    trade_value: float = 5000.0
    trade_disruption: float = 6.0
    shipping_disruption: float = 5.0
    strategic_route_exposure: float = 7.0

@app.get("/api/trade-dependency/summary")
@app.get("/trade-dependency/summary")
def get_trade_dependency_summary():
    """Returns dataset overview, K-Means clustering metrics, and cluster distribution."""
    return trade_dependency_service.get_summary()

@app.get("/api/trade-dependency/clusters")
@app.get("/trade-dependency/clusters")
def get_trade_dependency_clusters():
    """Returns the 4 discovered clusters, centers, record counts, and risk interpretations."""
    return trade_dependency_service.get_clusters()

@app.get("/api/trade-dependency/elbow")
@app.get("/trade-dependency/elbow")
def get_trade_dependency_elbow():
    """Returns within-cluster inertia curve across k=2 to k=8."""
    return trade_dependency_service.get_elbow()

@app.get("/api/trade-dependency/scatter")
@app.get("/trade-dependency/scatter")
def get_trade_dependency_scatter():
    """Returns 2D trade dependency scatter points (Import vs Export) for cluster visualization."""
    return trade_dependency_service.get_scatter()

@app.get("/api/trade-dependency/country-analysis")
@app.get("/trade-dependency/country-analysis")
def get_trade_dependency_country_analysis():
    """Returns country-wise primary risk group and trade exposure profiles."""
    return trade_dependency_service.get_country_analysis()

@app.get("/api/trade-dependency/samples")
@app.get("/trade-dependency/samples")
def get_trade_dependency_samples():
    """Returns curated preset country trade dependency profiles for 1-click exploration."""
    return trade_dependency_service.get_samples()

@app.post("/api/trade-dependency/predict")
@app.post("/trade-dependency/predict")
def predict_trade_dependency(req: TradeDependencyPredictRequest):
    """
    Live India Trade Risk Cluster Assignment:
    Assigns observation to learned K-Means cluster using standardized Euclidean distance.
    """
    result = trade_dependency_service.predict(
        india_import_dependency=req.india_import_dependency,
        india_export_dependency=req.india_export_dependency,
        energy_dependency=req.energy_dependency,
        commodity_dependency=req.commodity_dependency,
        trade_value=req.trade_value,
        trade_disruption=req.trade_disruption,
        shipping_disruption=req.shipping_disruption,
        strategic_route_exposure=req.strategic_route_exposure
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=503 if "unavailable" in result.get("error", "") else 400,
            detail=result.get("error", "Trade risk cluster assignment failed.")
        )
    return result

# =============================================================================
# FEATURE 9: GEOPOLITICAL SHOCK FINGERPRINTING & HISTORICAL COMPARISON API ROUTES
# =============================================================================

class ShockFingerprintPredictRequest(BaseModel):
    energy_shock: float = 75.0
    trade_disruption: float = 80.0
    shipping_disruption: float = 85.0
    commodity_shock: float = 70.0
    financial_stress: float = 65.0

@app.get("/api/shock-fingerprint/summary")
@app.get("/shock-fingerprint/summary")
def get_shock_fingerprint_summary():
    """Returns dataset metrics, PCA variance explained, selected K, and silhouette score."""
    return shock_fingerprint_service.get_summary()

@app.get("/api/shock-fingerprint/conflicts")
@app.get("/shock-fingerprint/conflicts")
def get_shock_fingerprint_conflicts():
    """Returns historical conflict benchmarks with 5 shock dimensions, macroeconomic impacts, and PCA coords."""
    return shock_fingerprint_service.get_conflicts()

@app.get("/api/shock-fingerprint/clusters")
@app.get("/shock-fingerprint/clusters")
def get_shock_fingerprint_clusters():
    """Returns discovered shock fingerprint clusters with PCA centroids, descriptions, and feature means."""
    return shock_fingerprint_service.get_clusters()

@app.get("/api/shock-fingerprint/pca")
@app.get("/shock-fingerprint/pca")
def get_shock_fingerprint_pca():
    """Returns PCA 2D coordinates for all historical conflict events."""
    return shock_fingerprint_service.get_pca()

@app.get("/api/shock-fingerprint/silhouette")
@app.get("/shock-fingerprint/silhouette")
def get_shock_fingerprint_silhouette():
    """Returns silhouette score analysis across candidate K (2 to 8)."""
    return shock_fingerprint_service.get_silhouette()

@app.get("/api/shock-fingerprint/presets")
@app.get("/shock-fingerprint/presets")
def get_shock_fingerprint_presets():
    """Returns curated representative conflict shock scenarios for 1-click exploration."""
    return shock_fingerprint_service.get_presets()

@app.post("/api/shock-fingerprint/predict")
@app.post("/shock-fingerprint/predict")
def predict_shock_fingerprint(req: ShockFingerprintPredictRequest):
    """
    Live Geopolitical Shock Fingerprinting:
    1. Standardizes 5 shock features.
    2. Projects onto 2D PCA plane (PC1, PC2).
    3. Assigns to learned K-Means cluster.
    4. Computes Euclidean distance to historical conflict benchmarks to find closest matches.
    5. Forecasts projected macro inflation, oil, GDP, and trade growth impacts.
    """
    result = shock_fingerprint_service.predict(
        energy_shock=req.energy_shock,
        trade_disruption=req.trade_disruption,
        shipping_disruption=req.shipping_disruption,
        commodity_shock=req.commodity_shock,
        financial_stress=req.financial_stress
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=503 if "unavailable" in result.get("error", "") else 400,
            detail=result.get("error", "Shock fingerprinting failed.")
        )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
