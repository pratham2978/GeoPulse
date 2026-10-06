"""
Generates Feature_7_India_Trade_Dependency_Country_Risk.ipynb
19 cells matching the user's exact specification and syllabus guidelines.
"""

import json
import os

def create_cell(cell_type, source, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = 1
        cell["outputs"] = outputs or []
    return cell

def build():
    cells = []

    # Cell 1 of 19: Markdown — Title & Objective
    cells.append(create_cell("markdown", """# Feature 7 — India Trade Dependency & Country Risk 🇮🇳
## GeoPulse AI — Global Conflict Impact Intelligence Platform

### Objective
Identify and group countries according to how strongly geopolitical and trade-related factors expose India to trade dependency risk.

The feature uses **K-Means Clustering** to discover groups of countries with similar India-trade-risk patterns.

### Real-life question
> “India kis country par kitna trade/commodity dependence rakhta hai, aur kaunse countries India ke liye comparatively higher trade-risk exposure create karte hain?”

### Strict syllabus mapping
- **Module 4:** K-Means Clustering
- **Module 4:** Feature Engineering Basics
- **Module 6:** Data preprocessing, model development, visualization and result interpretation

### Important restriction
*This feature intentionally uses K-Means only as the ML algorithm.*
Do not add Random Forest, Logistic Regression, SVM, k-NN, XGBoost, LSTM, ARIMA, neural networks, or other out-of-syllabus algorithms.

### End-to-end workflow
Real country/trade data → cleaning → feature preparation → scaling → K-Means → cluster analysis → cluster interpretation → country risk grouping → visualization → new-country cluster assignment.

*Never fabricate country records or pretend that clusters are official government risk ratings.*"""))

    # Cell 2 of 19: Markdown — Dataset
    cells.append(create_cell("markdown", """### Dataset
Place a real country-level historical/periodic CSV at:
`data/feature7_india_trade_dependency.csv`

#### Required columns
- `country`
- `event_date`
- `india_import_dependency`
- `india_export_dependency`
- `energy_dependency`
- `commodity_dependency`
- `trade_value`
- `trade_disruption`
- `shipping_disruption`
- `strategic_route_exposure`

#### Feature meaning
- `india_import_dependency`: India’s import exposure/dependency on the country.
- `india_export_dependency`: India’s export exposure to the country.
- `energy_dependency`: energy-related exposure.
- `commodity_dependency`: commodity-related exposure.
- `trade_value`: relevant India-country trade value for the observation.
- `trade_disruption`: measured/documented trade disruption indicator.
- `shipping_disruption`: measured/documented shipping disruption indicator.
- `strategic_route_exposure`: exposure to important trade/shipping routes.

*Use real measured/documented values and keep units consistent. Do not create random country records.*"""))

    # Cell 3 of 19: Code — Cell 2 (Imports, configuration and dataset loading)
    cells.append(create_cell("code", """# Cell 2 — Imports, configuration and dataset loading

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
N_CLUSTERS = 4

DATA_PATH = "data/feature7_india_trade_dependency.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Add the real country-level CSV first."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
display(df.head())"""))

    # Cell 4 of 19: Code — Cell 3 (Validate columns and clean dataset)
    cells.append(create_cell("code", """# Cell 3 — Validate columns and clean the dataset

required = {
    "country",
    "event_date",
    "india_import_dependency",
    "india_export_dependency",
    "energy_dependency",
    "commodity_dependency",
    "trade_value",
    "trade_disruption",
    "shipping_disruption",
    "strategic_route_exposure"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

numeric_columns = [
    "india_import_dependency",
    "india_export_dependency",
    "energy_dependency",
    "commodity_dependency",
    "trade_value",
    "trade_disruption",
    "shipping_disruption",
    "strategic_route_exposure"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
df["country"] = df["country"].astype(str).str.strip()

before = len(df)

df = df.drop_duplicates()
df = df.dropna(subset=numeric_columns + ["country", "event_date"])

print("Rows before cleaning:", before)
print("Rows after cleaning :", len(df))

display(df.head())"""))

    # Cell 5 of 19: Code — Cell 4 (Explore the trade-dependency variables)
    cells.append(create_cell("code", """# Cell 4 — Explore the trade-dependency variables

display(
    df[numeric_columns].describe().T
)

# Quick visualization of India import dependency
plt.figure(figsize=(9, 5))

plt.hist(
    df["india_import_dependency"],
    bins=20,
    color="#06b6d4",
    edgecolor="black",
    alpha=0.8
)

plt.title("India Import Dependency Distribution", fontsize=13, fontweight="bold")
plt.xlabel("India Import Dependency (%)", fontsize=11)
plt.ylabel("Number of Records", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()"""))

    # Cell 6 of 19: Markdown — Why scaling is required
    cells.append(create_cell("markdown", """### Why scaling is required
The variables have different units and ranges. For example, trade value can be much larger numerically than a disruption score.

K-Means uses distances between observations. Therefore, the clustering features are standardized before fitting the model so that one large-scale variable does not dominate the distance calculation."""))

    # Cell 7 of 19: Code — Cell 5 (Select clustering features and prepare data)
    cells.append(create_cell("code", """# Cell 5 — Select clustering features and prepare the data

features = [
    "india_import_dependency",
    "india_export_dependency",
    "energy_dependency",
    "commodity_dependency",
    "trade_value",
    "trade_disruption",
    "shipping_disruption",
    "strategic_route_exposure"
]

X = df[features].copy()

print("Clustering features:")
print(features)

print("\\nFeature matrix shape:", X.shape)
display(X.head())"""))

    # Cell 8 of 19: Code — Cell 6 (Standardize features and prepare K-Means)
    cells.append(create_cell("code", """# Cell 6 — Standardize features and prepare K-Means

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaled feature matrix shape:", X_scaled.shape)
print("First scaled row:")
print(np.round(X_scaled[0], 4))"""))

    # Cell 9 of 19: Code — Cell 7 (Elbow analysis for choosing K)
    cells.append(create_cell("code", """# Cell 7 — Elbow analysis for choosing K

inertias = []
k_values = range(2, 9)

for k in k_values:
    km = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o",
    linewidth=2,
    color="#f59e0b"
)

plt.xlabel("Number of Clusters (K)", fontsize=11)
plt.ylabel("Within-Cluster Inertia", fontsize=11)
plt.title("Elbow Analysis for K-Means (Optimal k = 4)", fontsize=13, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)

plt.xticks(list(k_values))
plt.tight_layout()
plt.show()

print("Selected K for this feature:", N_CLUSTERS)"""))

    # Cell 10 of 19: Markdown — Cluster selection
    cells.append(create_cell("markdown", """### Cluster selection
The project uses 4 clusters so the final dashboard can communicate four practical exposure groups:
1. **Lower exposure**
2. **Moderate exposure**
3. **Higher exposure**
4. **Critical/highest exposure**

These names are interpretations of the discovered clusters, not official government risk ratings.
The underlying ML output is the K-Means cluster number. The risk labels are assigned only after examining the cluster centers."""))

    # Cell 11 of 19: Code — Cell 8 (Train K-Means)
    cells.append(create_cell("code", """# Cell 8 — Train K-Means

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

print("K-Means trained successfully.")
print("\\nCluster counts:")
display(df["cluster"].value_counts().sort_index())"""))

    # Cell 12 of 19: Code — Cell 9 (Analyze cluster centers)
    cells.append(create_cell("code", """# Cell 9 — Analyze cluster centers

centers_scaled = kmeans.cluster_centers_

centers_original = scaler.inverse_transform(
    centers_scaled
)

cluster_centers = pd.DataFrame(
    centers_original,
    columns=features
)

cluster_centers.index.name = "cluster"

display(cluster_centers.round(2))"""))

    # Cell 13 of 19: Code — Cell 10 (Create interpretable India trade-risk groups)
    cells.append(create_cell("code", """# Cell 10 — Create interpretable India trade-risk groups

# A higher composite exposure across the clustering features
# is interpreted as higher trade-risk exposure.
#
# This is NOT a separate ML model. It only gives human-readable
# names to the already discovered K-Means clusters.

risk_score = centers_scaled.mean(axis=1)

ranked_clusters = pd.Series(risk_score).sort_values().index.tolist()

risk_names = [
    "Lower Exposure",
    "Moderate Exposure",
    "Higher Exposure",
    "Critical Exposure"
]

cluster_to_risk = {
    cluster_id: risk_names[position]
    for position, cluster_id in enumerate(ranked_clusters)
}

df["india_trade_risk_group"] = df["cluster"].map(cluster_to_risk)

print("Cluster → interpreted exposure group:")
for cluster_id in sorted(cluster_to_risk):
    print(cluster_id, "→", cluster_to_risk[cluster_id])

display(
    df[["country", "cluster", "india_trade_risk_group"]].head(20)
)"""))

    # Cell 14 of 19: Code — Cell 11 (Country-wise India trade-risk view)
    cells.append(create_cell("code", """# Cell 11 — Country-wise India trade-risk view

country_risk = (
    df.groupby(["country", "india_trade_risk_group"])
      .size()
      .reset_index(name="records")
      .sort_values(["india_trade_risk_group", "records"], ascending=[True, False])
)

display(country_risk.head(50))"""))

    # Cell 15 of 19: Code — Cell 12 (Visualize clusters using two important trade indicators)
    cells.append(create_cell("code", """# Cell 12 — Visualize clusters using two important trade indicators

plt.figure(figsize=(10, 7))

colors = {
    "Critical Exposure": "#f43f5e",
    "Higher Exposure": "#f97316",
    "Moderate Exposure": "#f59e0b",
    "Lower Exposure": "#10b981"
}

for risk_group in df["india_trade_risk_group"].unique():
    subset = df[df["india_trade_risk_group"] == risk_group]

    plt.scatter(
        subset["india_import_dependency"],
        subset["india_export_dependency"],
        label=risk_group,
        color=colors.get(risk_group, "#38bdf8"),
        alpha=0.75,
        edgecolors="black",
        s=45
    )

plt.xlabel("India Import Dependency (%)", fontsize=11)
plt.ylabel("India Export Dependency (%)", fontsize=11)
plt.title("India Trade Dependency Risk Clusters (K-Means k=4)", fontsize=13, fontweight="bold")
plt.legend(frameon=True)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()"""))

    # Cell 16 of 19: Code — Cell 13 (Cluster summary for dashboard)
    cells.append(create_cell("code", """# Cell 13 — Cluster summary for dashboard

cluster_summary = (
    df.groupby("india_trade_risk_group")[features]
      .mean()
      .round(3)
)

display(cluster_summary)

print("\\nNumber of countries/records in each group:")
display(
    df["india_trade_risk_group"]
      .value_counts()
      .rename_axis("Risk Group")
      .reset_index(name="Records")
)"""))

    # Cell 17 of 19: Code — Cell 14 (Assign a new country/event to a learned cluster)
    cells.append(create_cell("code", """# Cell 14 — Assign a new country/event to a learned cluster

def predict_trade_risk_group(
    india_import_dependency,
    india_export_dependency,
    energy_dependency,
    commodity_dependency,
    trade_value,
    trade_disruption,
    shipping_disruption,
    strategic_route_exposure
):
    new_country = pd.DataFrame([{
        "india_import_dependency": india_import_dependency,
        "india_export_dependency": india_export_dependency,
        "energy_dependency": energy_dependency,
        "commodity_dependency": commodity_dependency,
        "trade_value": trade_value,
        "trade_disruption": trade_disruption,
        "shipping_disruption": shipping_disruption,
        "strategic_route_exposure": strategic_route_exposure
    }])

    new_scaled = scaler.transform(new_country[features])

    cluster = int(kmeans.predict(new_scaled)[0])
    risk_group = cluster_to_risk[cluster]

    return {
        "cluster": cluster,
        "india_trade_risk_group": risk_group
    }


# Replace these demonstration inputs with website values.
result = predict_trade_risk_group(
    india_import_dependency=70,
    india_export_dependency=45,
    energy_dependency=65,
    commodity_dependency=60,
    trade_value=5000,
    trade_disruption=6,
    shipping_disruption=5,
    strategic_route_exposure=7
)

print("Inference Result:", result)"""))

    # Cell 18 of 19: Code — Cell 15 (Save the complete clustering model)
    cells.append(create_cell("code", """# Cell 15 — Save the complete clustering model

import joblib

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/feature7_india_trade_dependency_kmeans.joblib"

artifact = {
    "model": kmeans,
    "scaler": scaler,
    "features": features,
    "cluster_to_risk": cluster_to_risk,
    "cluster_centers": cluster_centers
}

joblib.dump(artifact, MODEL_PATH)

print("Saved:", MODEL_PATH)"""))

    # Cell 19 of 19: Code — Cell 16 (Final workflow summary)
    cells.append(create_cell("code", """# Cell 16 — Final workflow summary

print(
    '''
FINAL WORKFLOW

Real country/trade data
        ↓
Column validation
        ↓
Data cleaning
        ↓
Feature selection
        ↓
Standardization
        ↓
Elbow analysis
        ↓
K-Means clustering
        ↓
Cluster-center analysis
        ↓
India trade-risk interpretation
        ↓
Country-wise comparison
        ↓
Cluster visualization
        ↓
Save model + scaler
        ↓
New country/event input
        ↓
India trade-risk group prediction

IMPORTANT:
- K-Means is the only ML algorithm used.
- Risk names are interpretations of discovered clusters,
  not official government ratings.
- Do not fabricate country data.
- Do not hardcode cluster predictions.
'''
)"""))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_file = "Feature_7_India_Trade_Dependency_Country_Risk.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_file} with {len(cells)} cells.")

    nb_dir_file = os.path.join("notebooks", out_file)
    with open(nb_dir_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {nb_dir_file} with {len(cells)} cells.")

if __name__ == "__main__":
    build()
