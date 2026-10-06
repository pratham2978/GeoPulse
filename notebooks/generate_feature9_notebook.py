"""
Generate Feature 9 Jupyter Notebook with 25 cells exactly matching the user specification:
Feature 9 — Geopolitical Shock Fingerprinting & Historical Conflict Comparison
"""

import os
import json

def create_notebook():
    cells = [
        # Cell 1 of 25 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Feature 9 — Geopolitical Shock Fingerprinting & Historical Conflict Comparison\n",
                "\n",
                "### Objective\n",
                "This feature answers:\n",
                "> *“Past conflicts mein kya measurable economic/geopolitical shock hua tha, aur current conflict ka shock pattern historically kis type ke conflict se similar hai?”*\n",
                "\n",
                "The system creates a shock fingerprint for historical conflicts and a current event using measurable indicators such as energy, trade, shipping, commodity and financial/economic signals.\n",
                "\n",
                "The output is historical pattern comparison, not a guaranteed future-loss prediction.\n",
                "\n",
                "### Main outputs\n",
                "- Historical conflict shock fingerprint\n",
                "- Current event shock fingerprint\n",
                "- PCA 2D visualization\n",
                "- K-Means historical conflict clusters\n",
                "- Closest historical pattern/cluster\n",
                "- Then vs Now comparison\n",
                "- Historical observed impact evidence\n",
                "- Potential impact channels for the current event\n",
                "\n",
                "### Strict AI/ML syllabus mapping\n",
                "- **Module 4:** PCA\n",
                "- **Module 4:** K-Means Clustering\n",
                "- **Module 4:** Feature Engineering Basics\n",
                "- **Module 5:** Visualization and interpretation of model output\n",
                "\n",
                "### Important\n",
                "No Random Forest, XGBoost, LSTM, ARIMA, Transformer, BERT, Naive Bayes or other out-of-syllabus ML algorithm is used.\n",
                "\n",
                "The notebook does not invent historical losses or future losses. Historical impact values must come from the supplied/verified dataset."
            ]
        },
        # Cell 2 of 25 - Markdown
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Dataset\n",
                "Expected path:\n",
                "`data/feature9_geopolitical_shock_fingerprints.csv`\n",
                "\n",
                "Each row represents one historical conflict/event observation.\n",
                "\n",
                "#### Required columns\n",
                "- `conflict`\n",
                "- `period`\n",
                "- `energy_shock`\n",
                "- `trade_disruption`\n",
                "- `shipping_disruption`\n",
                "- `commodity_shock`\n",
                "- `financial_stress`\n",
                "- `inflation_change`\n",
                "- `gdp_growth_change`\n",
                "- `trade_growth_change`\n",
                "- `oil_price_change`\n",
                "\n",
                "#### Interpretation\n",
                "The first five variables form the main shock fingerprint.\n",
                "\n",
                "The last four economic variables are historical observed impact indicators used for the Then vs Now evidence section.\n",
                "\n",
                "Do not fabricate these values. Use historical/verified datasets and document the source in the project report."
            ]
        },
        # Cell 3 of 25 - Code (Cell 1)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 1 — Imports and configuration\n",
                "\n",
                "import os\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.decomposition import PCA\n",
                "from sklearn.cluster import KMeans\n",
                "from sklearn.metrics import silhouette_score\n",
                "\n",
                "DATA_PATH = \"data/feature9_geopolitical_shock_fingerprints.csv\"\n",
                "\n",
                "SHOCK_FEATURES = [\n",
                "    \"energy_shock\",\n",
                "    \"trade_disruption\",\n",
                "    \"shipping_disruption\",\n",
                "    \"commodity_shock\",\n",
                "    \"financial_stress\"\n",
                "]\n",
                "\n",
                "IMPACT_FEATURES = [\n",
                "    \"inflation_change\",\n",
                "    \"gdp_growth_change\",\n",
                "    \"trade_growth_change\",\n",
                "    \"oil_price_change\"\n",
                "]\n",
                "\n",
                "print(\"Configuration loaded.\")"
            ]
        },
        # Cell 4 of 25 - Code (Cell 2)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 2 — Load dataset\n",
                "\n",
                "if not os.path.exists(DATA_PATH):\n",
                "    raise FileNotFoundError(f\"Dataset not found: {DATA_PATH}\")\n",
                "\n",
                "df = pd.read_csv(DATA_PATH)\n",
                "\n",
                "print(\"Dataset shape:\", df.shape)\n",
                "display(df.head())"
            ]
        },
        # Cell 5 of 25 - Code (Cell 3)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 3 — Validate and clean dataset\n",
                "\n",
                "required_columns = [\n",
                "    \"conflict\", \"period\",\n",
                "    *SHOCK_FEATURES,\n",
                "    *IMPACT_FEATURES\n",
                "]\n",
                "\n",
                "missing_columns = [c for c in required_columns if c not in df.columns]\n",
                "\n",
                "if missing_columns:\n",
                "    raise ValueError(f\"Missing required columns: {missing_columns}\")\n",
                "\n",
                "df = df.copy()\n",
                "\n",
                "df[\"conflict\"] = df[\"conflict\"].astype(str).str.strip()\n",
                "df[\"period\"] = df[\"period\"].astype(str).str.strip()\n",
                "\n",
                "for col in SHOCK_FEATURES + IMPACT_FEATURES:\n",
                "    df[col] = pd.to_numeric(df[col], errors=\"coerce\")\n",
                "\n",
                "before = len(df)\n",
                "df = df.dropna(subset=SHOCK_FEATURES).reset_index(drop=True)\n",
                "after = len(df)\n",
                "\n",
                "print(\"Rows before cleaning:\", before)\n",
                "print(\"Rows after cleaning:\", after)\n",
                "print(\"Missing values in shock features:\")\n",
                "print(df[SHOCK_FEATURES].isna().sum())"
            ]
        },
        # Cell 6 of 25 - Markdown (Cell 4)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 4 — Exploratory analysis\n",
                "Before applying PCA/K-Means, inspect how the historical shock indicators vary across conflicts.\n",
                "\n",
                "This is descriptive analysis and visualization, not a separate ML algorithm."
            ]
        },
        # Cell 7 of 25 - Code (Cell 4)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 4 — Historical shock distribution\n",
                "\n",
                "df[SHOCK_FEATURES].plot(\n",
                "    kind=\"box\",\n",
                "    figsize=(12, 6),\n",
                "    rot=20\n",
                ")\n",
                "plt.title(\"Historical Geopolitical Shock Feature Distribution\")\n",
                "plt.ylabel(\"Measured / normalized shock value\")\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Cell 8 of 25 - Markdown (Cell 5)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 5 — Prepare the shock fingerprint\n",
                "The five shock indicators are standardized because they can have different scales.\n",
                "\n",
                "Standardization is preprocessing, not an additional ML algorithm."
            ]
        },
        # Cell 9 of 25 - Code (Cell 5)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 5 — Standardize shock features\n",
                "\n",
                "X = df[SHOCK_FEATURES].copy()\n",
                "\n",
                "scaler = StandardScaler()\n",
                "X_scaled = scaler.fit_transform(X)\n",
                "\n",
                "print(\"Scaled matrix shape:\", X_scaled.shape)"
            ]
        },
        # Cell 10 of 25 - Markdown (Cell 6)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 6 — PCA\n",
                "PCA reduces the multi-dimensional shock fingerprint into two principal components for visualization.\n",
                "\n",
                "The two components do not mean “loss” by themselves. They represent directions of maximum variance in the supplied shock indicators."
            ]
        },
        # Cell 11 of 25 - Code (Cell 6)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 6 — PCA to 2 dimensions\n",
                "\n",
                "pca = PCA(n_components=2, random_state=42)\n",
                "X_pca = pca.fit_transform(X_scaled)\n",
                "\n",
                "df[\"PC1\"] = X_pca[:, 0]\n",
                "df[\"PC2\"] = X_pca[:, 1]\n",
                "\n",
                "explained = pca.explained_variance_ratio_\n",
                "\n",
                "print(\"PC1 explained variance:\", round(explained[0] * 100, 2), \"%\")\n",
                "print(\"PC2 explained variance:\", round(explained[1] * 100, 2), \"%\")\n",
                "print(\"Total explained variance:\", round(explained.sum() * 100, 2), \"%\")"
            ]
        },
        # Cell 12 of 25 - Code (Cell 7)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 7 — PCA visualization\n",
                "\n",
                "plt.figure(figsize=(11, 7))\n",
                "\n",
                "plt.scatter(df[\"PC1\"], df[\"PC2\"], s=80)\n",
                "\n",
                "for _, row in df.iterrows():\n",
                "    plt.annotate(\n",
                "        row[\"conflict\"],\n",
                "        (row[\"PC1\"], row[\"PC2\"]),\n",
                "        xytext=(5, 5),\n",
                "        textcoords=\"offset points\",\n",
                "        fontsize=8\n",
                "    )\n",
                "\n",
                "plt.xlabel(\"Principal Component 1\")\n",
                "plt.ylabel(\"Principal Component 2\")\n",
                "plt.title(\"Historical Geopolitical Shock Fingerprints — PCA\")\n",
                "plt.grid(alpha=0.25)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Cell 13 of 25 - Markdown (Cell 8)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 8 — Select K for K-Means\n",
                "Use silhouette score to compare candidate cluster counts.\n",
                "\n",
                "This helps identify groups of historical conflicts with similar shock fingerprints."
            ]
        },
        # Cell 14 of 25 - Code (Cell 8)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 8 — Silhouette analysis for K selection\n",
                "\n",
                "max_k = min(8, len(df) - 1)\n",
                "\n",
                "if max_k < 2:\n",
                "    raise ValueError(\"At least 3 historical observations are required for clustering.\")\n",
                "\n",
                "scores = {}\n",
                "\n",
                "for k in range(2, max_k + 1):\n",
                "    model = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
                "    labels = model.fit_predict(X_scaled)\n",
                "    scores[k] = silhouette_score(X_scaled, labels)\n",
                "\n",
                "print(\"Silhouette scores:\")\n",
                "for k, score in scores.items():\n",
                "    print(f\"K={k}: {score:.4f}\")\n",
                "\n",
                "best_k = max(scores, key=scores.get)\n",
                "print(\"Selected K:\", best_k)"
            ]
        },
        # Cell 15 of 25 - Markdown (Cell 9)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 9 — K-Means historical conflict clustering\n",
                "K-Means groups historical conflicts according to similarity in their shock fingerprints.\n",
                "\n",
                "Cluster names such as “Energy Shock” should only be assigned after inspecting cluster characteristics; they are interpretations, not learned labels."
            ]
        },
        # Cell 16 of 25 - Code (Cell 9)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 9 — Fit K-Means\n",
                "\n",
                "kmeans = KMeans(\n",
                "    n_clusters=best_k,\n",
                "    random_state=42,\n",
                "    n_init=10\n",
                ")\n",
                "\n",
                "df[\"cluster\"] = kmeans.fit_predict(X_scaled)\n",
                "\n",
                "print(\"Cluster counts:\")\n",
                "print(df[\"cluster\"].value_counts().sort_index())"
            ]
        },
        # Cell 17 of 25 - Code (Cell 10)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 10 — Inspect cluster characteristics\n",
                "\n",
                "cluster_summary = df.groupby(\"cluster\")[SHOCK_FEATURES].mean().round(3)\n",
                "\n",
                "display(cluster_summary)\n",
                "\n",
                "print(\"Interpret clusters using the feature means above.\")\n",
                "print(\"Do not assign a semantic cluster name before inspecting the actual values.\")"
            ]
        },
        # Cell 18 of 25 - Code (Cell 11)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 11 — Cluster visualization in PCA space\n",
                "\n",
                "plt.figure(figsize=(11, 7))\n",
                "\n",
                "for cluster_id in sorted(df[\"cluster\"].unique()):\n",
                "    subset = df[df[\"cluster\"] == cluster_id]\n",
                "    plt.scatter(\n",
                "        subset[\"PC1\"],\n",
                "        subset[\"PC2\"],\n",
                "        s=90,\n",
                "        label=f\"Cluster {cluster_id}\"\n",
                "    )\n",
                "\n",
                "for _, row in df.iterrows():\n",
                "    plt.annotate(\n",
                "        row[\"conflict\"],\n",
                "        (row[\"PC1\"], row[\"PC2\"]),\n",
                "        xytext=(5, 5),\n",
                "        textcoords=\"offset points\",\n",
                "        fontsize=8\n",
                "    )\n",
                "\n",
                "plt.xlabel(\"PC1\")\n",
                "plt.ylabel(\"PC2\")\n",
                "plt.title(\"Historical Conflict Clusters — PCA + K-Means\")\n",
                "plt.legend()\n",
                "plt.grid(alpha=0.25)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Cell 19 of 25 - Markdown (Cell 12)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 12 — Historical impact evidence\n",
                "This section connects each conflict fingerprint with observed historical economic indicators.\n",
                "\n",
                "It does not train another prediction model.\n",
                "\n",
                "The values must come from reliable historical datasets/sources."
            ]
        },
        # Cell 20 of 25 - Code (Cell 12)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 12 — Historical impact table\n",
                "\n",
                "impact_table = df[\n",
                "    [\n",
                "        \"conflict\",\n",
                "        \"period\",\n",
                "        *SHOCK_FEATURES,\n",
                "        *IMPACT_FEATURES,\n",
                "        \"cluster\"\n",
                "    ]\n",
                "].sort_values([\"cluster\", \"conflict\"])\n",
                "\n",
                "display(impact_table)"
            ]
        },
        # Cell 21 of 25 - Markdown (Cell 13)
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Cell 13 — Then vs Now comparison\n",
                "Add one current-event row using currently observed/measured indicators.\n",
                "\n",
                "The current row is compared with historical clusters.\n",
                "\n",
                "**Important:**\n",
                "- This is a pattern comparison.\n",
                "- It is NOT a claim that the current event will reproduce the historical loss."
            ]
        },
        # Cell 22 of 25 - Code (Cell 13)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 13 — Current event input\n",
                "\n",
                "current_event = {\n",
                "    \"conflict\": \"CURRENT_EVENT\",\n",
                "    \"period\": \"CURRENT\",\n",
                "    \"energy_shock\": 0.0,\n",
                "    \"trade_disruption\": 0.0,\n",
                "    \"shipping_disruption\": 0.0,\n",
                "    \"commodity_shock\": 0.0,\n",
                "    \"financial_stress\": 0.0\n",
                "}\n",
                "\n",
                "current_df = pd.DataFrame([current_event])\n",
                "\n",
                "current_scaled = scaler.transform(current_df[SHOCK_FEATURES])\n",
                "\n",
                "current_pca = pca.transform(current_scaled)\n",
                "\n",
                "current_cluster = kmeans.predict(current_scaled)[0]\n",
                "\n",
                "print(\"Current event cluster:\", current_cluster)\n",
                "print(\"Current event PCA coordinates:\", current_pca[0])"
            ]
        },
        # Cell 23 of 25 - Code (Cell 14)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 14 — Find historical observations in the same cluster\n",
                "\n",
                "historical_cluster = df[df[\"cluster\"] == current_cluster].copy()\n",
                "\n",
                "print(\"Historical observations in the current event's cluster:\")\n",
                "display(\n",
                "    historical_cluster[\n",
                "        [\n",
                "            \"conflict\",\n",
                "            \"period\",\n",
                "            *SHOCK_FEATURES,\n",
                "            *IMPACT_FEATURES\n",
                "        ]\n",
                "    ].sort_values(\"conflict\")\n",
                ")\n",
                "\n",
                "print(\"\\nInterpretation:\")\n",
                "print(\n",
                "    \"The displayed historical observations share a similar shock-pattern cluster \"\n",
                "    \"with the current event based on the supplied features.\"\n",
                ")"
            ]
        },
        # Cell 24 of 25 - Code (Cell 15)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 15 — Then vs Now visualization\n",
                "\n",
                "comparison_features = [\n",
                "    \"energy_shock\",\n",
                "    \"trade_disruption\",\n",
                "    \"shipping_disruption\",\n",
                "    \"commodity_shock\",\n",
                "    \"financial_stress\"\n",
                "]\n",
                "\n",
                "current_values = current_df[comparison_features].iloc[0].values\n",
                "\n",
                "historical_mean = historical_cluster[comparison_features].mean().values\n",
                "\n",
                "comparison = pd.DataFrame({\n",
                "    \"Indicator\": comparison_features,\n",
                "    \"Historical Cluster Mean\": historical_mean,\n",
                "    \"Current Event\": current_values\n",
                "})\n",
                "\n",
                "comparison.set_index(\"Indicator\").plot(\n",
                "    kind=\"bar\",\n",
                "    figsize=(12, 6)\n",
                ")\n",
                "\n",
                "plt.title(\"Then vs Now — Geopolitical Shock Fingerprint\")\n",
                "plt.ylabel(\"Standardized / supplied indicator value\")\n",
                "plt.xticks(rotation=25)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "display(comparison.round(3))"
            ]
        },
        # Cell 25 of 25 - Code (Cell 16)
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cell 16 — Save model artifacts and final output\n",
                "\n",
                "import joblib\n",
                "\n",
                "os.makedirs(\"models\", exist_ok=True)\n",
                "\n",
                "artifact = {\n",
                "    \"scaler\": scaler,\n",
                "    \"pca\": pca,\n",
                "    \"kmeans\": kmeans,\n",
                "    \"shock_features\": SHOCK_FEATURES,\n",
                "    \"impact_features\": IMPACT_FEATURES,\n",
                "    \"selected_k\": best_k\n",
                "}\n",
                "\n",
                "MODEL_PATH = \"models/feature9_geopolitical_shock_fingerprint.joblib\"\n",
                "\n",
                "joblib.dump(artifact, MODEL_PATH)\n",
                "\n",
                "print(\"Saved:\", MODEL_PATH)\n",
                "\n",
                "print(\"\\nFINAL WORKFLOW\")\n",
                "print(\"Historical conflicts\")\n",
                "print(\"      ↓\")\n",
                "print(\"Shock indicators\")\n",
                "print(\"      ↓\")\n",
                "print(\"Standardization\")\n",
                "print(\"      ↓\")\n",
                "print(\"PCA\")\n",
                "print(\"      ↓\")\n",
                "print(\"K-Means\")\n",
                "print(\"      ↓\")\n",
                "print(\"Historical shock clusters\")\n",
                "print(\"      ↓\")\n",
                "print(\"Current event fingerprint\")\n",
                "print(\"      ↓\")\n",
                "print(\"Historical pattern comparison\")\n",
                "print(\"      ↓\")\n",
                "print(\"Then vs Now + potential impact channels\")"
            ]
        }
    ]

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
        "nbformat_minor": 5
    }

    # Save to root and notebooks directory
    root_path = "Feature_9_Geopolitical_Shock_Fingerprinting.ipynb"
    nb_path = "notebooks/Feature_9_Geopolitical_Shock_Fingerprinting.ipynb"

    with open(root_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    os.makedirs("notebooks", exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated Feature 9 notebook with {len(cells)} cells in '{root_path}' and '{nb_path}'!")

if __name__ == "__main__":
    create_notebook()
