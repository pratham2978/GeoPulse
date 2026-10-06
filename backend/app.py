"""
FastAPI Backend Application for GeoPulse AI
Feature 1: Global Event & Conflict Intelligence REST API
"""

import os
import sys

# Ensure backend directory is in python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.event_service import event_service
from services.narrative_service import narrative_service
from services.sentiment_service import sentiment_service
from pydantic import BaseModel

class NarrativePredictRequest(BaseModel):
    headline: str
    text: Optional[str] = ""

class SentimentPredictRequest(BaseModel):
    headline: Optional[str] = ""
    text: Optional[str] = ""
    article: Optional[str] = ""

app = FastAPI(
    title="GeoPulse AI — Global Event & Conflict Intelligence API",
    description="High-performance geopolitical intelligence API processing GDELT CAMEO events, GeoPulse severity analytics, ML Narrative Classification, and Sentiment Intelligence.",
    version="2.1.0"
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
        "service": "GeoPulse AI Global Event Intelligence",
        "version": "1.0.0",
        "engine": "CAMEO + GeoPulse Severity Analytics"
    }

@app.get("/api/events/stats")
def get_event_statistics():
    """
    Returns global KPI counts:
    - Active Events
    - Conflict Events
    - High / Critical Risk Events
    - Countries Affected
    - Average Severity Score
    - Top Impacted Sectors
    """
    return event_service.get_stats()

@app.get("/api/events")
def get_events(
    page: int = Query(1, ge=1),
    limit: int = Query(15, ge=1, le=100),
    country: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    search: Optional[str] = None,
    date_range: Optional[str] = None,
):
    """
    Paginated and filtered GDELT global events with GeoPulse severity scores.
    """
    return event_service.get_events(
        page=page,
        limit=limit,
        country=country,
        category=category,
        severity=severity,
        search=search,
        date_range=date_range
    )

@app.get("/api/events/search")
def search_events(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Search events by country, region, actor, or conflict description.
    """
    return event_service.get_events(page=1, limit=limit, search=q)

@app.get("/api/events/clusters")
def get_clusters():
    """
    Returns aggregated major geopolitical conflict clusters:
    - Red Sea Maritime Corridor
    - Strait of Hormuz Chokepoint
    - Eastern Europe & Black Sea Grain Corridor
    - Taiwan Strait Alert
    - South China Sea Standoff
    - India Northern Border
    """
    return event_service.get_clusters()

@app.get("/api/events/clusters/{cluster_id}")
def get_cluster_detail(cluster_id: str):
    cluster = event_service.get_cluster_by_id(cluster_id)
    if not cluster:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")
    return cluster

@app.get("/api/events/timeline")
def get_event_timeline():
    """
    Daily category event frequency timeline for trending analysis.
    """
    return event_service.get_timeline()

@app.get("/api/events/map")
def get_map_markers():
    """
    Geocoded intelligence nodes with severity indicators and casualty/event counts for global map.
    """
    return event_service.get_map_markers()

@app.get("/api/events/country/{country_code}")
def get_country_intelligence(country_code: str):
    """
    Returns country-level conflict, political, and economic intelligence profile.
    """
    return event_service.get_country_intelligence(country_code)

@app.get("/api/events/{event_id}")
def get_event_by_id(event_id: int):
    event = event_service.get_event_by_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found.")
    return event

# =============================================================================
# FEATURE 2: NEWS & NARRATIVE CLASSIFICATION API ROUTES
# =============================================================================

@app.get("/api/news-intelligence/summary")
def get_narrative_summary():
    """Returns dataset inspection, cleaning statistics, train/test split, and class distribution."""
    return narrative_service.get_summary()

@app.get("/api/news-intelligence/models")
def get_narrative_models():
    """Returns model comparison table for all 5 syllabus models, best model selection, and tuning results."""
    return narrative_service.get_models()

@app.get("/api/news-intelligence/confusion-matrix")
def get_narrative_confusion_matrix():
    """Returns confusion matrices for the 6 narrative categories across syllabus classifiers."""
    return narrative_service.get_confusion_matrices()

@app.get("/api/news-intelligence/roc-data")
def get_narrative_roc_data():
    """Returns multiclass ROC curve data (FPR, TPR, AUC) for each narrative category."""
    return narrative_service.get_roc_data()

@app.get("/api/news-intelligence/samples")
def get_narrative_samples():
    """Returns 3 authentic sample news articles from the dataset for instant UI demonstration."""
    return narrative_service.get_samples()

@app.post("/api/news-intelligence/predict")
def predict_narrative(req: NarrativePredictRequest):
    """
    Live prediction endpoint:
    Classifies a user-provided news headline and article text using the saved best model.
    Does NOT retrain the model.
    """
    if not req.headline and not req.text:
        raise HTTPException(
            status_code=400,
            detail="Please enter a news headline or article text."
        )
    result = narrative_service.predict(headline=req.headline, text=req.text or "")
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed."))
    return result

# =============================================================================
# FEATURE 3: NEWS SENTIMENT ANALYSIS API ROUTES
# =============================================================================

@app.get("/api/sentiment/summary")
@app.get("/sentiment/summary")
def get_sentiment_summary():
    """Returns sentiment dataset inspection, cleaning statistics, train/test split, and class distribution."""
    return sentiment_service.get_summary()

@app.get("/api/sentiment/models")
@app.get("/sentiment/models")
def get_sentiment_models():
    """Returns model comparison table for all 5 syllabus models, best model selection, and overfitting check."""
    return sentiment_service.get_models()

@app.get("/api/sentiment/confusion-matrix")
@app.get("/sentiment/confusion-matrix")
def get_sentiment_confusion_matrix():
    """Returns confusion matrices for Positive, Neutral, Negative across syllabus classifiers."""
    return sentiment_service.get_confusion_matrices()

@app.get("/api/sentiment/roc-data")
@app.get("/sentiment/roc-data")
def get_sentiment_roc_data():
    """Returns multiclass ROC curve data (FPR, TPR, AUC) for sentiment classes."""
    return sentiment_service.get_roc_data()

@app.get("/api/sentiment/samples")
@app.get("/sentiment/samples")
def get_sentiment_samples():
    """Returns 3 authentic sample news articles from the dataset for instant UI demonstration."""
    return sentiment_service.get_samples()

@app.post("/api/sentiment/predict")
@app.post("/sentiment/predict")
def predict_sentiment(req: SentimentPredictRequest):
    """
    Live Sentiment Prediction Endpoint:
    Classifies a user-provided news headline and article text using the saved best supervised ML model.
    Does NOT retrain the model on request.
    """
    input_text = (req.text or req.article or "").strip()
    headline = (req.headline or "").strip()

    if not headline and not input_text:
        raise HTTPException(
            status_code=400,
            detail="Please enter a news article before analysis."
        )

    result = sentiment_service.predict(headline=headline, text=input_text)
    if not result.get("success"):
        if result.get("status") == "unavailable":
            raise HTTPException(
                status_code=503,
                detail="Sentiment model is currently unavailable. Please train the model first."
            )
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Please enter a news article before analysis.")
        )
    return result

# =============================================================================
# FEATURE 4: INDIA ECONOMIC IMPACT INTELLIGENCE API ROUTES
# =============================================================================
from services.india_impact_service import india_impact_service

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
    Does NOT retrain the model on each request.
    Returns: predicted_impact (0-100), impact_level (LOW/MODERATE/HIGH/CRITICAL), model name.
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
# =============================================================================
# FEATURE 5: INDIA ENERGY SUPPLY RISK INTELLIGENCE API ROUTES
# =============================================================================
from services.india_energy_service import india_energy_service

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
    Does NOT retrain the model on each request.
    Returns: predicted risk tier (Low, Moderate, High, Critical), probability breakdown, latency, drivers.
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
# =============================================================================
# FEATURE 6: INDIA OIL & COMMODITY SHOCK INTELLIGENCE API ROUTES
# =============================================================================
from services.commodity_shock_service import commodity_shock_service

class CommodityShockPredictRequest(BaseModel):
    crude_oil_change: float = 15.0
    natural_gas_change: float = 10.0
    gold_price_change: float = 5.0
    essential_commodity_change: float = 8.0
    trade_disruption: float = 6.0
    shipping_disruption: float = 7.0
    conflict_intensity: float = 8.0
    india_import_dependency: float = 70.0

@app.get("/api/commodity-shock/summary")
@app.get("/commodity-shock/summary")
def get_commodity_shock_summary():
    """Returns dataset summary, target metrics, and model overview for Feature 6."""
    return commodity_shock_service.get_summary()

@app.get("/api/commodity-shock/model")
@app.get("/commodity-shock/model")
def get_commodity_shock_model():
    """Returns Multiple Linear Regression evaluation: MAE, RMSE, R², CV R², and Overfitting Gap."""
    return commodity_shock_service.get_model_metrics()

@app.get("/api/commodity-shock/coefficients")
@app.get("/commodity-shock/coefficients")
def get_commodity_shock_coefficients():
    """Returns sorted regression coefficients representing feature associations with India Commodity Impact."""
    return commodity_shock_service.get_coefficients()

@app.get("/api/commodity-shock/actual-vs-predicted")
@app.get("/commodity-shock/actual-vs-predicted")
def get_commodity_shock_avp():
    """Returns actual vs predicted scatter data points on test set."""
    return commodity_shock_service.get_actual_vs_predicted()

@app.get("/api/commodity-shock/country-analysis")
@app.get("/commodity-shock/country-analysis")
def get_commodity_shock_country_analysis():
    """Returns historical India commodity impact averages and ranges by country/origin."""
    return commodity_shock_service.get_country_analysis()

@app.get("/api/commodity-shock/samples")
@app.get("/commodity-shock/samples")
def get_commodity_shock_samples():
    """Returns curated preset commodity shock scenarios for 1-click exploration."""
    return commodity_shock_service.get_samples()

@app.get("/api/commodity-shock/trends")
@app.get("/commodity-shock/trends")
def get_commodity_shock_trends():
    """Returns historical commodity price shock trends and average India impact by year."""
    return commodity_shock_service.get_trends()

@app.post("/api/commodity-shock/predict")
@app.post("/commodity-shock/predict")
def predict_commodity_shock(req: CommodityShockPredictRequest):
    """
    Live India Commodity Shock Prediction:
    Runs the serialized Multiple Linear Regression model.
    Does NOT retrain the model on each request.
    Returns: predicted_impact (0-100), impact_level (LOW/MODERATE/HIGH/CRITICAL), model used.
    """
    result = commodity_shock_service.predict(
        crude_oil_change=req.crude_oil_change,
        natural_gas_change=req.natural_gas_change,
        gold_price_change=req.gold_price_change,
        essential_commodity_change=req.essential_commodity_change,
        trade_disruption=req.trade_disruption,
        shipping_disruption=req.shipping_disruption,
        conflict_intensity=req.conflict_intensity,
        india_import_dependency=req.india_import_dependency
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=503 if "unavailable" in result.get("error", "") else 400,
            detail=result.get("error", "Commodity shock prediction failed.")
        )
# =============================================================================
# FEATURE 7: INDIA TRADE DEPENDENCY & COUNTRY RISK INTELLIGENCE API ROUTES
# =============================================================================
from services.trade_dependency_service import trade_dependency_service

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
    Does NOT retrain K-Means on each request.
    Returns: cluster, india_trade_risk_group, distance_to_center, cluster_distances.
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
# =============================================================================
# FEATURE 8: INDIA SUPPLY-ROUTE DISRUPTION INTELLIGENCE API ROUTES
# =============================================================================
from services.supply_route_service import supply_route_service

class SupplyRoutePredictRequest(BaseModel):
    route_disruption: float = 70.0
    shipping_delay: float = 65.0
    freight_cost_change: float = 20.0
    trade_volume_exposure: float = 60.0
    india_import_exposure: float = 75.0
    india_export_exposure: float = 45.0
    energy_route_exposure: float = 70.0
    commodity_exposure: float = 60.0
    conflict_intensity: float = 75.0

@app.get("/api/supply-route/summary")
@app.get("/supply-route/summary")
def get_supply_route_summary():
    """Returns dataset summary, class distribution, and champion model specifications."""
    return supply_route_service.get_summary()

@app.get("/api/supply-route/models")
@app.get("/supply-route/models")
def get_supply_route_models():
    """Returns comparative metrics across all 5 syllabus classifiers and overfitting/underfitting checks."""
    return supply_route_service.get_models()

@app.get("/api/supply-route/confusion-matrix")
@app.get("/supply-route/confusion-matrix")
def get_supply_route_confusion():
    """Returns multi-class confusion matrices for each of the 5 syllabus models."""
    return supply_route_service.get_confusion_matrix()

@app.get("/api/supply-route/roc-data")
@app.get("/supply-route/roc-data")
def get_supply_route_roc():
    """Returns One-vs-Rest ROC curve data (FPR, TPR, AUC) for each risk class."""
    return supply_route_service.get_roc_data()

@app.get("/api/supply-route/route-analysis")
@app.get("/supply-route/route-analysis")
def get_supply_route_analysis():
    """Returns route-wise risk distribution and mean disruption indicators."""
    return supply_route_service.get_route_analysis()

@app.get("/api/supply-route/samples")
@app.get("/supply-route/samples")
def get_supply_route_samples():
    """Returns preset maritime disruption scenarios for 1-click exploration."""
    return supply_route_service.get_samples()

@app.post("/api/supply-route/predict")
@app.post("/supply-route/predict")
def predict_supply_route_risk(req: SupplyRoutePredictRequest):
    """
    Live India Supply-Route Risk Classification:
    Uses the trained syllabus champion model to predict Low, Moderate, High, or Critical risk.
    """
    result = supply_route_service.predict(
        route_disruption=req.route_disruption,
        shipping_delay=req.shipping_delay,
        freight_cost_change=req.freight_cost_change,
        trade_volume_exposure=req.trade_volume_exposure,
        india_import_exposure=req.india_import_exposure,
        india_export_exposure=req.india_export_exposure,
        energy_route_exposure=req.energy_route_exposure,
        commodity_exposure=req.commodity_exposure,
        conflict_intensity=req.conflict_intensity
    )
    if not result.get("success"):
        raise HTTPException(
            status_code=503 if "unavailable" in result.get("error", "") else 400,
            detail=result.get("error", "Supply route risk classification failed.")
        )
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

