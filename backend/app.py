"""
FastAPI Backend Application for GeoPulse AI
Feature 1: Global Event & Conflict Intelligence REST API
"""

import os
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.event_service import event_service

app = FastAPI(
    title="GeoPulse AI — Global Event & Conflict Intelligence API",
    description="High-performance geopolitical intelligence API processing GDELT CAMEO events and GeoPulse severity analytics.",
    version="1.0.0"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
