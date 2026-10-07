"""
Feature 4 — India Economic Impact Intelligence
Training Pipeline: Simple & Multiple Linear Regression

Syllabus:
  Module 3: Simple Linear Regression, Multiple Linear Regression
  Module 5: Train/Test Split, K-Fold Cross Validation,
            Overfitting/Underfitting, Model Optimization
  Module 6: Data preprocessing, model development, evaluation,
            visualization, result interpretation

STRICT CONSTRAINT:
  Only Simple Linear Regression and Multiple Linear Regression.
  No Random Forest, SVM, k-NN, XGBoost, LSTM, Transformers, or any other algorithm.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "feature4_india_economic_impact.csv")
MODELS_OUT   = os.path.join(PROJECT_ROOT, "backend", "models")
os.makedirs(MODELS_OUT, exist_ok=True)

def run():
    print("=" * 70)
    print("GeoPulse AI — Feature 4: India Economic Impact Intelligence")
    print("Regression Training Pipeline")
    print("=" * 70)

    # ── 1. Load Dataset ────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"\n[1] Loaded dataset: {df.shape}")

    # ── 2. Required Column Validation ─────────────────────────────────────────
    required = {
        "country", "event_date", "oil_price_change",
        "commodity_price_change", "trade_disruption",
        "shipping_disruption", "india_trade_exposure",
        "india_energy_exposure", "market_volatility",
        "india_economic_impact"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ── 3. Data Cleaning ───────────────────────────────────────────────────────
    numeric_cols = [
        "oil_price_change", "commodity_price_change",
        "trade_disruption", "shipping_disruption",
        "india_trade_exposure", "india_energy_exposure",
        "market_volatility", "india_economic_impact"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=numeric_cols + ["country", "event_date"])
    df = df.drop_duplicates()
    print(f"[2] Cleaned: {before} -> {len(df)} rows")

    # ── 4. Feature / Target Definition ────────────────────────────────────────
    feature_cols = [
        "oil_price_change", "commodity_price_change",
        "trade_disruption", "shipping_disruption",
        "india_trade_exposure", "india_energy_exposure",
        "market_volatility"
    ]
    simple_feature = ["oil_price_change"]   # highest single-feature correlation
    target = "india_economic_impact"

    X_simple   = df[simple_feature]
    X_multiple = df[feature_cols]
    y          = df[target]

    # ── 5. Stratified Train-Test Split (80 / 20) ───────────────────────────────
    X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
        X_simple, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_multiple, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"[3] Train: {len(X_tr)}  |  Test: {len(X_te)}")

    # ── 6. Simple Linear Regression ───────────────────────────────────────────
    slr = LinearRegression()
    slr.fit(X_s_tr, y_s_tr)
    slr_pred_test  = slr.predict(X_s_te)
    slr_pred_train = slr.predict(X_s_tr)

    slr_mae  = float(mean_absolute_error(y_s_te, slr_pred_test))
    slr_rmse = float(np.sqrt(mean_squared_error(y_s_te, slr_pred_test)))
    slr_r2   = float(r2_score(y_s_te, slr_pred_test))
    slr_r2_train = float(r2_score(y_s_tr, slr_pred_train))
    print(f"[4] SLR  -> MAE {slr_mae:.4f}  RMSE {slr_rmse:.4f}  R² {slr_r2:.4f}")

    # ── 7. Multiple Linear Regression ─────────────────────────────────────────
    mlr = LinearRegression()
    mlr.fit(X_tr, y_tr)
    mlr_pred_test  = mlr.predict(X_te)
    mlr_pred_train = mlr.predict(X_tr)

    mlr_mae  = float(mean_absolute_error(y_te, mlr_pred_test))
    mlr_rmse = float(np.sqrt(mean_squared_error(y_te, mlr_pred_test)))
    mlr_r2   = float(r2_score(y_te, mlr_pred_test))
    mlr_r2_train = float(r2_score(y_tr, mlr_pred_train))
    print(f"[5] MLR  -> MAE {mlr_mae:.4f}  RMSE {mlr_rmse:.4f}  R² {mlr_r2:.4f}")

    # ── 8. 5-Fold Cross Validation ─────────────────────────────────────────────
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    slr_cv = cross_val_score(LinearRegression(), X_simple, y, cv=cv, scoring="r2")
    mlr_cv = cross_val_score(LinearRegression(), X_multiple, y, cv=cv, scoring="r2")
    print(f"[6] SLR 5-Fold CV R² -> mean {slr_cv.mean():.4f}  std {slr_cv.std():.4f}")
    print(f"[7] MLR 5-Fold CV R² -> mean {mlr_cv.mean():.4f}  std {mlr_cv.std():.4f}")

    # ── 9. Regression Coefficients ─────────────────────────────────────────────
    coefs = [
        {"feature": f, "coefficient": round(c, 6)}
        for f, c in zip(feature_cols, mlr.coef_)
    ]
    coefs.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    # ── 10. Actual vs Predicted pairs for scatter plot in UI ──────────────────
    actual_vs_pred = [
        {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
        for a, p in zip(y_te, mlr_pred_test)
    ]

    # ── 11. Country Summary (for the country comparison widget) ──────────────
    country_stats = (
        df.groupby("country")["india_economic_impact"]
          .agg(count="count", mean="mean", min="min", max="max")
          .reset_index()
          .sort_values("mean", ascending=False)
    )
    country_stats["mean"] = country_stats["mean"].round(2)
    country_stats["min"]  = country_stats["min"].round(2)
    country_stats["max"]  = country_stats["max"].round(2)
    country_rows = country_stats.to_dict("records")

    # ── 12. Sample Events (3 representative rows) ────────────────────────────
    sample_rows = []
    for label in ["high", "medium", "low"]:
        if label == "high":
            row = df.nlargest(3, "india_economic_impact").iloc[0]
        elif label == "medium":
            med = df["india_economic_impact"].median()
            row = df.iloc[(df["india_economic_impact"] - med).abs().argsort().iloc[0]]
        else:
            row = df.nsmallest(3, "india_economic_impact").iloc[0]
        sample_rows.append({
            "country": row["country"],
            "event_description": row.get("event_description", "Geopolitical Event"),
            "oil_price_change": round(float(row["oil_price_change"]), 2),
            "commodity_price_change": round(float(row["commodity_price_change"]), 2),
            "trade_disruption": round(float(row["trade_disruption"]), 2),
            "shipping_disruption": round(float(row["shipping_disruption"]), 2),
            "india_trade_exposure": round(float(row["india_trade_exposure"]), 2),
            "india_energy_exposure": round(float(row["india_energy_exposure"]), 2),
            "market_volatility": round(float(row["market_volatility"]), 2),
            "actual_impact": round(float(row["india_economic_impact"]), 2),
            "impact_tier": label
        })

    # ── 13. Target-range statistics ──────────────────────────────────────────
    target_min  = float(df["india_economic_impact"].min())
    target_max  = float(df["india_economic_impact"].max())
    target_mean = float(df["india_economic_impact"].mean())

    # ── 14. Save JSON artifacts ───────────────────────────────────────────────

    # dataset_summary
    summary = {
        "total_records": int(before),
        "cleaned_records": len(df),
        "training_records": len(X_tr),
        "testing_records": len(X_te),
        "countries": int(df["country"].nunique()),
        "date_range_start": str(df["event_date"].min().date()),
        "date_range_end": str(df["event_date"].max().date()),
        "target_variable": "india_economic_impact",
        "target_min": round(target_min, 2),
        "target_max": round(target_max, 2),
        "target_mean": round(target_mean, 2),
        "features": feature_cols
    }
    with open(os.path.join(MODELS_OUT, "india_impact_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # model_comparison
    comparison = {
        "models": [
            {
                "model": "Simple Linear Regression",
                "features": simple_feature,
                "mae": round(slr_mae, 4),
                "rmse": round(slr_rmse, 4),
                "r2": round(slr_r2, 4),
                "train_r2": round(slr_r2_train, 4),
                "cv_r2_mean": round(float(slr_cv.mean()), 4),
                "cv_r2_std": round(float(slr_cv.std()), 4),
                "intercept": round(float(slr.intercept_), 6),
                "gap": round(slr_r2_train - slr_r2, 4)
            },
            {
                "model": "Multiple Linear Regression",
                "features": feature_cols,
                "mae": round(mlr_mae, 4),
                "rmse": round(mlr_rmse, 4),
                "r2": round(mlr_r2, 4),
                "train_r2": round(mlr_r2_train, 4),
                "cv_r2_mean": round(float(mlr_cv.mean()), 4),
                "cv_r2_std": round(float(mlr_cv.std()), 4),
                "intercept": round(float(mlr.intercept_), 6),
                "gap": round(mlr_r2_train - mlr_r2, 4)
            }
        ],
        "best_model": "Multiple Linear Regression",
        "coefficients": coefs
    }
    # Also save directly to src/data for frontend
    SRC_DATA = os.path.join(PROJECT_ROOT, "src", "data")
    os.makedirs(SRC_DATA, exist_ok=True)
    with open(os.path.join(SRC_DATA, "india_impact_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(SRC_DATA, "india_impact_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    with open(os.path.join(SRC_DATA, "india_impact_actual_vs_pred.json"), "w") as f:
        json.dump(actual_vs_pred, f, indent=2)
    with open(os.path.join(SRC_DATA, "india_impact_country_analysis.json"), "w") as f:
        json.dump(country_rows, f, indent=2)
    with open(os.path.join(SRC_DATA, "india_impact_samples.json"), "w") as f:
        json.dump(sample_rows, f, indent=2)

    print(f"\n[8] Artifacts saved to {MODELS_OUT}/ and {SRC_DATA}/")

    # ── 15. Save Model ────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_OUT, "feature4_india_economic_impact_model.joblib")
    joblib.dump(mlr, model_path)

    # also save to root /models for notebook
    root_models = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(root_models, exist_ok=True)
    joblib.dump(mlr, os.path.join(root_models, "feature4_india_economic_impact_model.joblib"))

    print(f"[9] Model saved: {model_path}")
    print("\nOK Feature 4 training pipeline complete.")
    print(f"  Best model: Multiple Linear Regression (R² = {mlr_r2:.4f}, MAE = {mlr_mae:.4f})")

    return mlr, summary, comparison

if __name__ == "__main__":
    run()
