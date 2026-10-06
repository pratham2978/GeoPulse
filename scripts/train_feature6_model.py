"""
Feature 6 — India Oil & Commodity Shock Model Training & Evaluation
Trains Multiple Linear Regression model adhering strictly to Modules 3, 5, and 6.

Saves:
- models/feature6_india_commodity_shock_model.joblib
- src/data/commodity_shock_summary.json
- src/data/commodity_shock_model.json
- src/data/commodity_shock_coefficients.json
- src/data/commodity_shock_actual_vs_pred.json
- src/data/commodity_shock_country_analysis.json
- src/data/commodity_shock_samples.json
- src/data/commodity_shock_trends.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA_PATH = "data/feature6_india_commodity_shock.csv"

def run_pipeline():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    required = {
        "country", "event_date", "crude_oil_change", "natural_gas_change",
        "gold_price_change", "essential_commodity_change", "trade_disruption",
        "shipping_disruption", "conflict_intensity", "india_import_dependency",
        "india_commodity_impact"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    numeric = [
        "crude_oil_change", "natural_gas_change", "gold_price_change",
        "essential_commodity_change", "trade_disruption", "shipping_disruption",
        "conflict_intensity", "india_import_dependency", "india_commodity_impact"
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["country"] = df["country"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates().dropna(subset=numeric + ["country", "event_date"])
    print(f"Cleaned: before={before}, after={len(df)}")

    features = [
        "crude_oil_change", "natural_gas_change", "gold_price_change",
        "essential_commodity_change", "trade_disruption", "shipping_disruption",
        "conflict_intensity", "india_import_dependency"
    ]
    target = "india_commodity_impact"

    X = df[features]
    y = df[target]

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Multiple Linear Regression Pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Performance Metrics
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))
    r2_test = float(r2_score(y_test, y_pred_test))
    r2_train = float(r2_score(y_train, y_pred_train))
    gap = float(r2_train - r2_test)

    # 5-Fold Cross Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())

    print("\n========== MODEL PERFORMANCE ==========")
    print(f"MAE       : {mae:.3f}")
    print(f"RMSE      : {rmse:.3f}")
    print(f"Test R²   : {r2_test:.4f}")
    print(f"Train R²  : {r2_train:.4f}")
    print(f"Gap       : {gap:.4f}")
    print(f"Mean CV R²: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # Fit Diagnostic
    if gap > 0.15:
        fit_status = "Possible Overfitting"
    elif r2_train < 0.50 and r2_test < 0.50:
        fit_status = "Possible Underfitting"
    else:
        fit_status = "Well-Regularized & Optimal"

    # Regression Coefficients
    linear_estimator = model.named_steps["model"]
    raw_coefs = linear_estimator.coef_
    intercept = float(linear_estimator.intercept_)

    feature_labels = {
        "crude_oil_change": "Crude Oil Shock (%)",
        "natural_gas_change": "Natural Gas Shock (%)",
        "gold_price_change": "Gold Price Change (%)",
        "essential_commodity_change": "Essential Commodity Shock (%)",
        "trade_disruption": "Trade Disruption Index (1-10)",
        "shipping_disruption": "Shipping Disruption Index (1-10)",
        "conflict_intensity": "Conflict Intensity (1-10)",
        "india_import_dependency": "India Import Dependency (%)"
    }

    coefficients = []
    for f, c in zip(features, raw_coefs):
        coefficients.append({
            "feature": f,
            "label": feature_labels.get(f, f),
            "coefficient": round(float(c), 4),
            "absolute_impact": round(float(abs(c)), 4),
            "direction": "Positive Pressure" if c > 0 else "Mitigating Factor"
        })
    coefficients.sort(key=lambda x: x["absolute_impact"], reverse=True)

    # Actual vs Predicted data points (Test Set)
    avp_points = []
    for act, pred in zip(y_test, y_pred_test):
        avp_points.append({
            "actual": round(float(act), 2),
            "predicted": round(float(pred), 2),
            "residual": round(float(act - pred), 2)
        })

    # Country-Wise Impact Analysis
    country_stats = (
        df.groupby("country")["india_commodity_impact"]
        .agg(["mean", "min", "max", "count", "std"])
        .reset_index()
    )
    country_rows = []
    for _, r in country_stats.iterrows():
        c_mean = float(r["mean"])
        tier = "CRITICAL" if c_mean >= 75 else "HIGH" if c_mean >= 55 else "MODERATE" if c_mean >= 35 else "LOW"
        country_rows.append({
            "country": str(r["country"]),
            "avg_impact": round(c_mean, 2),
            "min_impact": round(float(r["min"]), 2),
            "max_impact": round(float(r["max"]), 2),
            "std_impact": round(float(r["std"]) if pd.notnull(r["std"]) else 0.0, 2),
            "events_count": int(r["count"]),
            "risk_tier": tier
        })
    country_rows.sort(key=lambda x: x["avg_impact"], reverse=True)

    # Annual Commodity Shock Historical Trends
    df["year"] = df["event_date"].dt.year
    trends_df = df.groupby("year").agg({
        "crude_oil_change": "mean",
        "natural_gas_change": "mean",
        "gold_price_change": "mean",
        "essential_commodity_change": "mean",
        "india_commodity_impact": "mean"
    }).reset_index()
    trends = []
    for _, tr in trends_df.iterrows():
        trends.append({
            "year": int(tr["year"]),
            "crude_oil": round(float(tr["crude_oil_change"]), 2),
            "natural_gas": round(float(tr["natural_gas_change"]), 2),
            "gold": round(float(tr["gold_price_change"]), 2),
            "essential": round(float(tr["essential_commodity_change"]), 2),
            "india_impact": round(float(tr["india_commodity_impact"]), 2)
        })

    # Curated Preset Scenarios for 1-Click Exploration
    samples = [
        {
            "id": "persian-gulf-crisis",
            "name": "Persian Gulf Naval Escalation & Hormuz Alert",
            "country": "Iran",
            "tag": "Severe Crude & LNG Squeeze",
            "description": "Retaliatory missile attacks and maritime seizures shut tanker lanes in Strait of Hormuz. Widespread crude price surge and freight insurance spike.",
            "inputs": {
                "crude_oil_change": 24.5,
                "natural_gas_change": 28.0,
                "gold_price_change": 12.0,
                "essential_commodity_change": 14.5,
                "trade_disruption": 8.5,
                "shipping_disruption": 9.2,
                "conflict_intensity": 9.0,
                "india_import_dependency": 82.5
            },
            "expected_tier": "CRITICAL"
        },
        {
            "id": "black-sea-blockade",
            "name": "Black Sea Agri & Fertilizer Corridor Freeze",
            "country": "Ukraine",
            "tag": "Essential Edible Oil & Potash Shock",
            "description": "Blockade of Odesa and Danube ports halts sunflower oil and grain shipments. Ammonia pipeline shutdown triggers domestic fertilizer inflation.",
            "inputs": {
                "crude_oil_change": 14.0,
                "natural_gas_change": 22.0,
                "gold_price_change": 7.5,
                "essential_commodity_change": 32.0,
                "trade_disruption": 7.8,
                "shipping_disruption": 8.0,
                "conflict_intensity": 8.8,
                "india_import_dependency": 76.0
            },
            "expected_tier": "HIGH"
        },
        {
            "id": "red-sea-reroute",
            "name": "Red Sea Houthi Drone Interdictions",
            "country": "Yemen",
            "tag": "Maritime Rerouting & War Risk Surcharge",
            "description": "Attacks at Bab el-Mandeb force commercial container ships and tankers to reroute via Cape of Good Hope, adding 12 days transit and shipping fees.",
            "inputs": {
                "crude_oil_change": 8.5,
                "natural_gas_change": 11.5,
                "gold_price_change": 5.0,
                "essential_commodity_change": 7.2,
                "trade_disruption": 7.2,
                "shipping_disruption": 9.0,
                "conflict_intensity": 8.2,
                "india_import_dependency": 79.5
            },
            "expected_tier": "HIGH"
        },
        {
            "id": "opec-quota-rebalance",
            "name": "OPEC+ Supply Extension with High Inventories",
            "country": "Saudi Arabia",
            "tag": "Modest Benchmark Crude Pass-through",
            "description": "OPEC+ extends voluntary cuts by 500k bpd into strong Atlantic supply. Indian domestic refiners offset with term contracts.",
            "inputs": {
                "crude_oil_change": 5.0,
                "natural_gas_change": 3.0,
                "gold_price_change": 1.5,
                "essential_commodity_change": 2.5,
                "trade_disruption": 3.5,
                "shipping_disruption": 3.0,
                "conflict_intensity": 3.0,
                "india_import_dependency": 75.0
            },
            "expected_tier": "MODERATE"
        },
        {
            "id": "atlantic-basin-buffer",
            "name": "US-Brazil Record Bumper Harvest & Energy Surge",
            "country": "USA",
            "tag": "Global Commodity Supply Cushion",
            "description": "Record US Permian basin crude extraction and South American soybean export expansion soften international benchmark prices.",
            "inputs": {
                "crude_oil_change": -6.5,
                "natural_gas_change": -4.0,
                "gold_price_change": -2.0,
                "essential_commodity_change": -5.5,
                "trade_disruption": 1.5,
                "shipping_disruption": 1.8,
                "conflict_intensity": 1.5,
                "india_import_dependency": 69.0
            },
            "expected_tier": "LOW"
        }
    ]

    # Summary
    summary = {
        "total_records": int(len(df)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "countries_count": int(df["country"].nunique()),
        "date_range_start": df["event_date"].min().strftime("%Y-%m-%d"),
        "date_range_end": df["event_date"].max().strftime("%Y-%m-%d"),
        "target_variable": target,
        "target_mean": round(float(y.mean()), 2),
        "target_min": round(float(y.min()), 2),
        "target_max": round(float(y.max()), 2),
        "target_std": round(float(y.std()), 2),
        "target_p25": round(float(y.quantile(0.25)), 2),
        "target_p50": round(float(y.median()), 2),
        "target_p75": round(float(y.quantile(0.75)), 2),
        "features": features,
        "model_name": "Multiple Linear Regression",
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2_test": round(r2_test, 4),
        "r2_train": round(r2_train, 4),
        "cv_r2_mean": round(cv_mean, 4),
        "cv_r2_std": round(cv_std, 4),
        "train_test_gap": round(gap, 4),
        "fit_diagnostic": fit_status,
        "intercept": round(intercept, 4)
    }

    # Model Performance Object
    model_metrics = {
        "model_name": "Multiple Linear Regression",
        "mae": round(mae, 3),
        "mse": round(mse, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2_test, 4),
        "train_r2": round(r2_train, 4),
        "gap": round(gap, 4),
        "cv_r2_mean": round(cv_mean, 4),
        "cv_r2_std": round(cv_std, 4),
        "fit_status": fit_status,
        "coefficients": coefficients,
        "intercept": round(intercept, 4)
    }

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/feature6_india_commodity_shock_model.joblib"
    joblib.dump(model, model_save_path)
    print(f"Saved model to {model_save_path}")

    # Save frontend JSON artifacts
    os.makedirs("src/data", exist_ok=True)
    with open("src/data/commodity_shock_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("src/data/commodity_shock_model.json", "w") as f:
        json.dump(model_metrics, f, indent=2)
    with open("src/data/commodity_shock_coefficients.json", "w") as f:
        json.dump(coefficients, f, indent=2)
    with open("src/data/commodity_shock_actual_vs_pred.json", "w") as f:
        json.dump(avp_points, f, indent=2)
    with open("src/data/commodity_shock_country_analysis.json", "w") as f:
        json.dump(country_rows, f, indent=2)
    with open("src/data/commodity_shock_samples.json", "w") as f:
        json.dump(samples, f, indent=2)
    with open("src/data/commodity_shock_trends.json", "w") as f:
        json.dump(trends, f, indent=2)

    print("All Feature 6 artifacts and UI JSON files exported successfully!")

if __name__ == "__main__":
    run_pipeline()
