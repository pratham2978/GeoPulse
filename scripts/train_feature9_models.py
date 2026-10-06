"""
Feature 9: Geopolitical Shock Fingerprinting & Historical Conflict Comparison
Adhering strictly to syllabus:
- Module 4: PCA (2D projection, explained variance ratio)
- Module 4: K-Means Clustering (Silhouette analysis for K selection, fitting K-Means, cluster characteristics)
- Module 4: Feature Engineering Basics (StandardScaler standardization)
- Module 5: Visualization and interpretation of model output
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = "data/feature9_geopolitical_shock_fingerprints.csv"

SHOCK_FEATURES = [
    "energy_shock",
    "trade_disruption",
    "shipping_disruption",
    "commodity_shock",
    "financial_stress"
]

IMPACT_FEATURES = [
    "inflation_change",
    "gdp_growth_change",
    "trade_growth_change",
    "oil_price_change"
]

# Semantic cluster naming based on feature means inspection
CLUSTER_PROFILES = {
    "Systemic Energy & Inflation Shocks": {
        "tag": "Severe Global Stagflation",
        "description": "Massive crude/energy supply cutoffs, triple-digit oil price spikes, severe global inflation, and major GDP contraction.",
        "color": "#ef4444",
        "badgeBg": "bg-rose-500/20 text-rose-400 border-rose-500/40"
    },
    "Maritime Chokepoint & Supply-Chain Crises": {
        "tag": "Maritime & Freight Stress",
        "description": "Severe transit lane blockage, surging container freight costs, and trade flow friction with moderate macroeconomic contraction.",
        "color": "#06b6d4",
        "badgeBg": "bg-cyan-500/20 text-cyan-400 border-cyan-500/40"
    },
    "Compound Geopolitical & Financial Liquidity Shocks": {
        "tag": "Financial & Trade Contagion",
        "description": "Cross-border war combined with acute credit/liquidity contraction, market crashes, and broad industrial commodity surges.",
        "color": "#a855f7",
        "badgeBg": "bg-purple-500/20 text-purple-400 border-purple-500/40"
    },
    "Localized Conventional Warfare": {
        "tag": "Contained Regional Friction",
        "description": "Geographically focused military operations with contained contagion to global maritime routes and global energy balances.",
        "color": "#10b981",
        "badgeBg": "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"
    }
}

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    required_columns = ["conflict", "period", *SHOCK_FEATURES, *IMPACT_FEATURES]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["conflict"] = df["conflict"].astype(str).str.strip()
    df["period"] = df["period"].astype(str).str.strip()

    for col in SHOCK_FEATURES + IMPACT_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=SHOCK_FEATURES).reset_index(drop=True)
    print(f"Cleaned dataset: {len(df)} rows")

    # 1. Standardize shock features
    X = df[SHOCK_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA to 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df["PC1"] = [round(float(v), 4) for v in X_pca[:, 0]]
    df["PC2"] = [round(float(v), 4) for v in X_pca[:, 1]]

    explained = pca.explained_variance_ratio_
    print(f"PC1 variance: {explained[0]*100:.2f}% | PC2 variance: {explained[1]*100:.2f}% | Total: {explained.sum()*100:.2f}%")

    # 3. Silhouette Analysis for K Selection (candidate K: 2 to min(8, len(df)-1))
    max_k = min(8, len(df) - 1)
    scores = {}
    silhouette_list = []

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = float(silhouette_score(X_scaled, labels))
        scores[k] = score
        silhouette_list.append({"k": k, "silhouette_score": round(score, 4)})

    best_k = max(scores, key=scores.get)
    print(f"Selected K = {best_k} (Silhouette = {scores[best_k]:.4f})")

    # 4. Fit Champion K-Means
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    print("Cluster distribution:\n", df["cluster"].value_counts().sort_index())

    # 5. Cluster Characterization & Semantic Labeling
    cluster_means = df.groupby("cluster")[SHOCK_FEATURES].mean()
    cluster_impact_means = df.groupby("cluster")[IMPACT_FEATURES].mean()

    # Assign meaningful labels based on dominant feature characteristics
    clusters_info = []
    semantic_names = list(CLUSTER_PROFILES.keys())

    for c_id in range(best_k):
        c_df = df[df["cluster"] == c_id]
        mean_shocks = cluster_means.loc[c_id].to_dict()
        mean_impacts = cluster_impact_means.loc[c_id].to_dict()

        # Determine best semantic name by characteristic
        if mean_shocks["energy_shock"] >= 80 and mean_shocks["commodity_shock"] >= 75:
            c_name = "Systemic Energy & Inflation Shocks"
        elif mean_shocks["shipping_disruption"] >= 75:
            c_name = "Maritime Chokepoint & Supply-Chain Crises"
        elif mean_shocks["financial_stress"] >= 70:
            c_name = "Compound Geopolitical & Financial Liquidity Shocks"
        else:
            c_name = "Localized Conventional Warfare"

        profile = CLUSTER_PROFILES.get(c_name, {
            "tag": f"Cluster {c_id}",
            "description": "Distinct historical shock configuration.",
            "color": "#38bdf8",
            "badgeBg": "bg-sky-500/20 text-sky-400 border-sky-500/40"
        })

        # Calculate cluster center in PCA space
        center_scaled = kmeans.cluster_centers_[c_id].reshape(1, -1)
        center_pca = pca.transform(center_scaled)[0]

        clusters_info.append({
            "cluster_id": int(c_id),
            "name": c_name,
            "tag": profile["tag"],
            "description": profile["description"],
            "color": profile["color"],
            "badgeBg": profile["badgeBg"],
            "record_count": len(c_df),
            "center_pca": [round(float(center_pca[0]), 4), round(float(center_pca[1]), 4)],
            "shock_means": {k: round(float(v), 2) for k, v in mean_shocks.items()},
            "impact_means": {k: round(float(v), 2) for k, v in mean_impacts.items()},
            "conflicts": c_df["conflict"].tolist()
        })

    # Save model artifacts
    os.makedirs("models", exist_ok=True)
    model_path = "models/feature9_geopolitical_shock_fingerprint.joblib"
    artifact = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "shock_features": SHOCK_FEATURES,
        "impact_features": IMPACT_FEATURES,
        "selected_k": best_k,
        "cluster_profiles": clusters_info
    }
    joblib.dump(artifact, model_path)
    print(f"Saved model artifact to {model_path}")

    # Prepare frontend exports
    os.makedirs("src/data", exist_ok=True)

    # 1. Summary
    summary_data = {
        "dataset_name": "feature9_geopolitical_shock_fingerprints.csv",
        "total_conflicts": len(df),
        "shock_features": SHOCK_FEATURES,
        "impact_features": IMPACT_FEATURES,
        "selected_k": best_k,
        "silhouette_score": round(float(scores[best_k]), 4),
        "pc1_variance": round(float(explained[0] * 100), 2),
        "pc2_variance": round(float(explained[1] * 100), 2),
        "total_pca_variance": round(float(explained.sum() * 100), 2)
    }
    with open("src/data/shock_fingerprint_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # 2. Conflicts list with PCA coordinates and cluster assignments
    conflicts_list = []
    for _, row in df.iterrows():
        c_id = int(row["cluster"])
        c_info = next((c for c in clusters_info if c["cluster_id"] == c_id), None)
        conflicts_list.append({
            "conflict": row["conflict"],
            "period": row["period"],
            "cluster": c_id,
            "cluster_name": c_info["name"] if c_info else f"Cluster {c_id}",
            "cluster_color": c_info["color"] if c_info else "#38bdf8",
            "pc1": float(row["PC1"]),
            "pc2": float(row["PC2"]),
            "shocks": {f: float(row[f]) for f in SHOCK_FEATURES},
            "impacts": {f: float(row[f]) for f in IMPACT_FEATURES}
        })
    with open("src/data/shock_fingerprint_conflicts.json", "w") as f:
        json.dump(conflicts_list, f, indent=2)

    # 3. Clusters detail
    with open("src/data/shock_fingerprint_clusters.json", "w") as f:
        json.dump(clusters_info, f, indent=2)

    # 4. PCA details
    pca_export = {
        "components": [
            {
                "name": "PC1",
                "variance_ratio": round(float(explained[0] * 100), 2),
                "loadings": {feat: round(float(pca.components_[0][i]), 4) for i, feat in enumerate(SHOCK_FEATURES)}
            },
            {
                "name": "PC2",
                "variance_ratio": round(float(explained[1] * 100), 2),
                "loadings": {feat: round(float(pca.components_[1][i]), 4) for i, feat in enumerate(SHOCK_FEATURES)}
            }
        ]
    }
    with open("src/data/shock_fingerprint_pca.json", "w") as f:
        json.dump(pca_export, f, indent=2)

    # 5. Silhouette curve
    with open("src/data/shock_fingerprint_silhouette.json", "w") as f:
        json.dump(silhouette_list, f, indent=2)

    # 6. Presets for live testing
    presets = [
        {
            "id": "red_sea_crisis",
            "name": "2024 Red Sea Houthi Drone & Missile Standoff",
            "category": "Maritime & Chokepoint Disruption",
            "description": "Commercial vessel rerouting around Africa, container freight surcharges, war-risk insurance spikes.",
            "inputs": {
                "energy_shock": 74.0,
                "trade_disruption": 85.0,
                "shipping_disruption": 92.0,
                "commodity_shock": 68.0,
                "financial_stress": 66.0
            }
        },
        {
            "id": "russia_ukraine_escalation",
            "name": "Russia-Ukraine Energy & Commodity Embargo",
            "category": "Compound Global Commodity Shock",
            "description": "Severed Nord Stream / Black Sea pipelines, European gas crisis, fertilizer spikes, multi-decade high inflation.",
            "inputs": {
                "energy_shock": 95.0,
                "trade_disruption": 90.0,
                "shipping_disruption": 82.0,
                "commodity_shock": 94.0,
                "financial_stress": 85.0
            }
        },
        {
            "id": "strait_of_hormuz_standoff",
            "name": "Strait of Hormuz Tanker Interdiction & Mine Threat",
            "category": "Severe Hydrocarbon Supply Choke",
            "description": "Immediate naval interdiction in the Persian Gulf, 20M bpd crude traffic halted, benchmark Brent surge.",
            "inputs": {
                "energy_shock": 92.0,
                "trade_disruption": 72.0,
                "shipping_disruption": 86.0,
                "commodity_shock": 70.0,
                "financial_stress": 78.0
            }
        },
        {
            "id": "taiwan_strait_naval_drills",
            "name": "Taiwan Strait Air-Sea Blockade Simulation",
            "category": "Advanced Semiconductor & Electronics Shock",
            "description": "East Asian maritime shipping corridors rerouted, semiconductor supply-chain halt, acute global market stress.",
            "inputs": {
                "energy_shock": 50.0,
                "trade_disruption": 94.0,
                "shipping_disruption": 88.0,
                "commodity_shock": 72.0,
                "financial_stress": 92.0
            }
        },
        {
            "id": "regional_border_skirmish",
            "name": "Isolated High-Altitude Border Clash",
            "category": "Contained Conventional Friction",
            "description": "Tactical skirmish between regional powers without closure of primary maritime lanes or global oil flows.",
            "inputs": {
                "energy_shock": 26.0,
                "trade_disruption": 32.0,
                "shipping_disruption": 20.0,
                "commodity_shock": 24.0,
                "financial_stress": 42.0
            }
        }
    ]
    with open("src/data/shock_fingerprint_presets.json", "w") as f:
        json.dump(presets, f, indent=2)

    print("All exports successfully generated!")

if __name__ == "__main__":
    main()
