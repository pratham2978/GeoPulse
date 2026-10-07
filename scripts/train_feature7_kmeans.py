"""
Feature 7 — India Trade Dependency & Country Risk Training & K-Means Pipeline
Executes K-Means Clustering (k=4) with feature standardization, elbow analysis, and cluster interpretation.

Saves:
- models/feature7_india_trade_dependency_kmeans.joblib
- src/data/trade_dependency_summary.json
- src/data/trade_dependency_elbow.json
- src/data/trade_dependency_clusters.json
- src/data/trade_dependency_scatter.json
- src/data/trade_dependency_country_analysis.json
- src/data/trade_dependency_samples.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_CLUSTERS = 4
DATA_PATH = "data/feature7_india_trade_dependency.csv"

def run_pipeline():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    required = {
        "country", "event_date", "india_import_dependency",
        "india_export_dependency", "energy_dependency", "commodity_dependency",
        "trade_value", "trade_disruption", "shipping_disruption",
        "strategic_route_exposure"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_columns = [
        "india_import_dependency", "india_export_dependency", "energy_dependency",
        "commodity_dependency", "trade_value", "trade_disruption",
        "shipping_disruption", "strategic_route_exposure"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["country"] = df["country"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates().dropna(subset=numeric_columns + ["country", "event_date"])
    print(f"Cleaned rows: before={before}, after={len(df)}")

    features = numeric_columns
    X = df[features].copy()

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Analysis (k=2..8)
    inertias = []
    k_range = list(range(2, 9))
    elbow_data = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        elbow_data.append({"k": k, "inertia": round(float(km.inertia_), 2)})

    print("Elbow inertias calculated:", elbow_data)

    # Fit Champion K-Means with k=4
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Cluster Centers
    centers_scaled = kmeans.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)

    cluster_centers_df = pd.DataFrame(centers_orig, columns=features)
    cluster_centers_df.index.name = "cluster"

    # Risk group ranking based on average standardized feature exposure
    risk_score = centers_scaled.mean(axis=1)
    ranked_cluster_ids = np.argsort(risk_score) # from lowest to highest composite exposure

    risk_names = [
        "Lower Exposure",
        "Moderate Exposure",
        "Higher Exposure",
        "Critical Exposure"
    ]

    cluster_to_risk = {
        int(cluster_id): risk_names[position]
        for position, cluster_id in enumerate(ranked_cluster_ids)
    }

    df["india_trade_risk_group"] = df["cluster"].map(cluster_to_risk)

    print("\nCluster Mapping:")
    for cid in sorted(cluster_to_risk.keys()):
        print(f"Cluster {cid} -> {cluster_to_risk[cid]}")

    # Cluster Profiles Object
    cluster_profiles = []
    risk_colors = {
        "Critical Exposure": "#f43f5e",
        "Higher Exposure": "#f97316",
        "Moderate Exposure": "#f59e0b",
        "Lower Exposure": "#10b981"
    }

    for cid in range(N_CLUSTERS):
        r_group = cluster_to_risk[cid]
        c_counts = int((df["cluster"] == cid).sum())
        c_pct = round((c_counts / len(df)) * 100, 1)
        mean_vals = cluster_centers_df.loc[cid].to_dict()

        cluster_profiles.append({
            "cluster_id": cid,
            "risk_group": r_group,
            "color": risk_colors[r_group],
            "record_count": c_counts,
            "record_percentage": c_pct,
            "centers": {k: round(float(v), 2) for k, v in mean_vals.items()},
            "scaled_centers": {k: round(float(v), 3) for k, v in zip(features, centers_scaled[cid])}
        })

    cluster_profiles.sort(key=lambda x: risk_names.index(x["risk_group"]))

    # Country-Wise Risk Breakdown
    country_groups = []
    country_grouped = df.groupby("country")

    for country_name, grp in country_grouped:
        mode_risk = grp["india_trade_risk_group"].mode()[0]
        avg_imp = float(grp["india_import_dependency"].mean())
        avg_exp = float(grp["india_export_dependency"].mean())
        avg_energy = float(grp["energy_dependency"].mean())
        avg_comm = float(grp["commodity_dependency"].mean())
        avg_trade_val = float(grp["trade_value"].mean())
        avg_route = float(grp["strategic_route_exposure"].mean())
        avg_ship = float(grp["shipping_disruption"].mean())

        country_groups.append({
            "country": country_name,
            "primary_risk_group": mode_risk,
            "color": risk_colors[mode_risk],
            "total_records": int(len(grp)),
            "avg_import_dep": round(avg_imp, 1),
            "avg_export_dep": round(avg_exp, 1),
            "avg_energy_dep": round(avg_energy, 1),
            "avg_comm_dep": round(avg_comm, 1),
            "avg_trade_val": round(avg_trade_val, 1),
            "avg_route_exp": round(avg_route, 1),
            "avg_ship_disp": round(avg_ship, 1)
        })

    # Sort countries by primary risk severity then import dependency
    country_groups.sort(
        key=lambda x: (risk_names.index(x["primary_risk_group"]), x["avg_import_dep"]),
        reverse=True
    )

    # 2D Scatter Data: Representative Sample for fast browser rendering
    scatter_sample = df.sample(min(800, len(df)), random_state=RANDOM_STATE)
    scatter_points = []
    for _, row in scatter_sample.iterrows():
        scatter_points.append({
            "country": row["country"],
            "import_dep": round(float(row["india_import_dependency"]), 1),
            "export_dep": round(float(row["india_export_dependency"]), 1),
            "energy_dep": round(float(row["energy_dependency"]), 1),
            "trade_val": round(float(row["trade_value"]), 1),
            "cluster": int(row["cluster"]),
            "risk_group": row["india_trade_risk_group"],
            "color": risk_colors[row["india_trade_risk_group"]]
        })

    # Curated Preset Scenarios for 1-Click Exploration
    samples = [
        {
            "id": "critical-energy-chokepoint",
            "name": "Strait of Hormuz Petro-Trade Anchor",
            "country": "Persian Gulf Exporter",
            "tag": "Critical Energy & Chokepoint Dependency",
            "description": "High crude and gas dependency with critical transit vulnerability across the Strait of Hormuz.",
            "inputs": {
                "india_import_dependency": 36.5,
                "india_export_dependency": 4.5,
                "energy_dependency": 46.0,
                "commodity_dependency": 22.0,
                "trade_value": 4800.0,
                "trade_disruption": 8.0,
                "shipping_disruption": 8.5,
                "strategic_route_exposure": 9.2
            },
            "expected_group": "Critical Exposure"
        },
        {
            "id": "manufacturing-critical-hub",
            "name": "High-Tech Electronics & Industrial Hub",
            "country": "East Asian Manufacturer",
            "tag": "High Non-Energy Import Concentration",
            "description": "Massive import concentration in active pharmaceutical ingredients, electronics, and heavy machinery.",
            "inputs": {
                "india_import_dependency": 48.0,
                "india_export_dependency": 11.5,
                "energy_dependency": 4.0,
                "commodity_dependency": 58.0,
                "trade_value": 9800.0,
                "trade_disruption": 7.2,
                "shipping_disruption": 6.5,
                "strategic_route_exposure": 7.5
            },
            "expected_group": "Higher Exposure"
        },
        {
            "id": "balanced-western-partner",
            "name": "Diversified Western Services & Trade Partner",
            "country": "Transatlantic Partner",
            "tag": "High Export Value & Open Maritime Lanes",
            "description": "Large bilateral merchandise and services trade with balanced dependencies and open Atlantic shipping.",
            "inputs": {
                "india_import_dependency": 21.0,
                "india_export_dependency": 42.0,
                "energy_dependency": 11.0,
                "commodity_dependency": 16.0,
                "trade_value": 11500.0,
                "trade_disruption": 4.0,
                "shipping_disruption": 4.5,
                "strategic_route_exposure": 5.0
            },
            "expected_group": "Moderate Exposure"
        },
        {
            "id": "peripheral-specialized-exporter",
            "name": "Niche Agricultural & Mineral Partner",
            "country": "South American / African Exporter",
            "tag": "Low Strategic Disruption Exposure",
            "description": "Targeted bilateral commodity exchange with minimal chokepoint friction and low aggregate dependency.",
            "inputs": {
                "india_import_dependency": 7.5,
                "india_export_dependency": 5.0,
                "energy_dependency": 3.0,
                "commodity_dependency": 18.0,
                "trade_value": 950.0,
                "trade_disruption": 3.0,
                "shipping_disruption": 3.5,
                "strategic_route_exposure": 4.0
            },
            "expected_group": "Lower Exposure"
        }
    ]

    # Overall Summary
    summary = {
        "total_records": int(len(df)),
        "countries_count": int(df["country"].nunique()),
        "date_range_start": df["event_date"].min().strftime("%Y-%m-%d"),
        "date_range_end": df["event_date"].max().strftime("%Y-%m-%d"),
        "n_clusters": N_CLUSTERS,
        "inertia": round(float(kmeans.inertia_), 2),
        "features": features,
        "risk_names": risk_names,
        "cluster_distribution": {
            r_name: int((df["india_trade_risk_group"] == r_name).sum())
            for r_name in risk_names
        }
    }

    # Save Model Artifact
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/feature7_india_trade_dependency_kmeans.joblib"
    artifact = {
        "model": kmeans,
        "scaler": scaler,
        "features": features,
        "cluster_to_risk": cluster_to_risk,
        "cluster_centers": cluster_centers_df
    }
    joblib.dump(artifact, model_save_path)
    print(f"Saved complete clustering artifact to: {model_save_path}")

    # Save UI JSONs
    os.makedirs("src/data", exist_ok=True)
    with open("src/data/trade_dependency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("src/data/trade_dependency_elbow.json", "w") as f:
        json.dump(elbow_data, f, indent=2)
    with open("src/data/trade_dependency_clusters.json", "w") as f:
        json.dump(cluster_profiles, f, indent=2)
    with open("src/data/trade_dependency_scatter.json", "w") as f:
        json.dump(scatter_points, f, indent=2)
    with open("src/data/trade_dependency_country_analysis.json", "w") as f:
        json.dump(country_groups, f, indent=2)
    with open("src/data/trade_dependency_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("All Feature 7 model artifacts and UI JSON files exported successfully!")

if __name__ == "__main__":
    run_pipeline()
