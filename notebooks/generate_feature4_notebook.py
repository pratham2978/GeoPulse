"""
Generates Feature_4_India_Economic_Impact_Intelligence.ipynb
36 cells matching the academic specification exactly.
"""

import json, os

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

    # Cell 1: Title
    cells.append(create_cell("markdown", """# Feature 4 — India Economic Impact Intelligence 🇮🇳
## GeoPulse AI — Global Conflict Impact Intelligence Platform

**Objective:** Estimate how a global country / event can affect India's economy using supervised regression.

### Core question
> How much economic impact could this country/event have on India?

### Strict syllabus mapping
- **Module 3:** Simple Linear Regression, Multiple Linear Regression
- **Module 5:** Training/Validation/Testing, Train-Test Split, Cross Validation, Overfitting/Underfitting, Model Optimization
- **Module 6:** Data preprocessing, model development, evaluation, visualization and result interpretation.

*Do not add Random Forest, SVM, k-NN, XGBoost, LSTM, Transformers or other ML algorithms to this feature.*
"""))

    # Cell 2: Environment setup
    cells.append(create_cell("code", """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA_PATH = "data/feature4_india_economic_impact.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. "
        "Place the real historical India-impact CSV there."
    )

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
display(df.head())
"""))

    # Cell 3: Section 1 heading
    cells.append(create_cell("markdown", """## 1. Expected Dataset Structure

**Required fields:**
- `country` — geopolitical partner country
- `event_date` — date of event
- `oil_price_change` — Brent crude monthly % change
- `commodity_price_change` — metal/agri commodity % change
- `trade_disruption` — % of bilateral trade affected (0–100)
- `shipping_disruption` — freight rate spike index (0–100)
- `india_trade_exposure` — India bilateral trade share % (0–100)
- `india_energy_exposure` — India energy import dependency share % (0–100)
- `market_volatility` — India VIX z-score (0–10)
- `india_economic_impact` — **composite measurable target (0–100)** derived from weighted real economic driver formula

`india_economic_impact` must be a real measurable target prepared from historical data, not a manually invented ML score.
Document its exact definition in the project report.
"""))

    # Cell 4: Column validation
    cells.append(create_cell("code", """required = {
    "country", "event_date", "oil_price_change",
    "commodity_price_change", "trade_disruption",
    "shipping_disruption", "india_trade_exposure",
    "india_energy_exposure", "market_volatility",
    "india_economic_impact"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

print("Missing values:")
display(df.isna().sum())
"""))

    # Cell 5: Section 2 heading
    cells.append(create_cell("markdown", "## 2. Data Cleaning and Preprocessing"))

    # Cell 6: Cleaning
    cells.append(create_cell("code", """df = df.copy()

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

print("Rows before cleaning:", before)
print("Rows after cleaning:", len(df))
"""))

    # Cell 7: Section 3
    cells.append(create_cell("markdown", "## 3. Exploratory Data Analysis"))

    # Cell 8: EDA - distribution
    cells.append(create_cell("code", """plt.figure(figsize=(8, 5))
plt.hist(df["india_economic_impact"], bins=20, color="#06b6d4", edgecolor="#0e7490")
plt.title("Distribution of India Economic Impact Score")
plt.xlabel("India Economic Impact (0-100)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print("\\nDescriptive statistics:")
display(df["india_economic_impact"].describe())
"""))

    # Cell 9: Correlation heatmap
    cells.append(create_cell("code", """feature_cols = [
    "oil_price_change", "commodity_price_change",
    "trade_disruption", "shipping_disruption",
    "india_trade_exposure", "india_energy_exposure",
    "market_volatility"
]

corr = df[feature_cols + ["india_economic_impact"]].corr()

plt.figure(figsize=(9, 7))
im = plt.imshow(corr, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=60, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
"""))

    # Cell 10: Section 4
    cells.append(create_cell("markdown", """## 4. Train-Test Split

**Simple Linear Regression** demonstrates one predictor (`oil_price_change`).
**Multiple Linear Regression** uses all selected numeric exposure/market variables.
"""))

    # Cell 11: Split
    cells.append(create_cell("code", """target = "india_economic_impact"
simple_features = ["oil_price_change"]
multiple_features = feature_cols

X_simple = df[simple_features]
X_multiple = df[multiple_features]
y = df[target]

X_simple_train, X_simple_test, y_simple_train, y_simple_test = train_test_split(
    X_simple, y, test_size=0.20, random_state=RANDOM_STATE
)

X_train, X_test, y_train, y_test = train_test_split(
    X_multiple, y, test_size=0.20, random_state=RANDOM_STATE
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
"""))

    # Cell 12: Section 5
    cells.append(create_cell("markdown", "## 5. Simple Linear Regression"))

    # Cell 13: SLR
    cells.append(create_cell("code", """simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_simple_train)

simple_pred = simple_model.predict(X_simple_test)

simple_mae  = mean_absolute_error(y_simple_test, simple_pred)
simple_rmse = np.sqrt(mean_squared_error(y_simple_test, simple_pred))
simple_r2   = r2_score(y_simple_test, simple_pred)

print(f"Simple Linear Regression (oil_price_change -> india_economic_impact)")
print(f"  Coefficient : {simple_model.coef_[0]:.6f}")
print(f"  Intercept   : {simple_model.intercept_:.6f}")
print(f"  MAE         : {simple_mae:.4f}")
print(f"  RMSE        : {simple_rmse:.4f}")
print(f"  R²          : {simple_r2:.4f}")
"""))

    # Cell 14: Section 6
    cells.append(create_cell("markdown", "## 6. Multiple Linear Regression"))

    # Cell 15: MLR
    cells.append(create_cell("code", """multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)

multiple_pred = multiple_model.predict(X_test)

multiple_mae  = mean_absolute_error(y_test, multiple_pred)
multiple_rmse = np.sqrt(mean_squared_error(y_test, multiple_pred))
multiple_r2   = r2_score(y_test, multiple_pred)

print(f"Multiple Linear Regression (7 features)")
print(f"  MAE  : {multiple_mae:.4f}")
print(f"  RMSE : {multiple_rmse:.4f}")
print(f"  R²   : {multiple_r2:.4f}")
"""))

    # Cell 16: Section 7
    cells.append(create_cell("markdown", "## 7. Actual vs Predicted"))

    # Cell 17: scatter plot
    cells.append(create_cell("code", """plt.figure(figsize=(7, 6))
plt.scatter(y_test, multiple_pred, alpha=0.6, color="#06b6d4", edgecolors="#0e7490")
low  = min(float(y_test.min()), float(multiple_pred.min()))
high = max(float(y_test.max()), float(multiple_pred.max()))
plt.plot([low, high], [low, high], "--", color="#f43f5e", linewidth=1.5)
plt.xlabel("Actual India Economic Impact")
plt.ylabel("Predicted India Economic Impact")
plt.title("Actual vs Predicted — Multiple Linear Regression")
plt.tight_layout()
plt.show()
"""))

    # Cell 18: Section 8
    cells.append(create_cell("markdown", "## 8. Regression Coefficients"))

    # Cell 19: Coefficients
    cells.append(create_cell("code", """coef_df = pd.DataFrame({
    "Feature": multiple_features,
    "Coefficient": multiple_model.coef_
}).sort_values("Coefficient", key=lambda s: s.abs(), ascending=False)

display(coef_df)
print("Intercept:", multiple_model.intercept_)
"""))

    # Cell 20: Section 9
    cells.append(create_cell("markdown", "## 9. Five-Fold Cross Validation"))

    # Cell 21: CV
    cells.append(create_cell("code", """cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

simple_cv   = cross_val_score(LinearRegression(), X_simple,   y, cv=cv, scoring="r2")
multiple_cv = cross_val_score(LinearRegression(), X_multiple, y, cv=cv, scoring="r2")

cv_results = pd.DataFrame({
    "Model": ["Simple Linear Regression", "Multiple Linear Regression"],
    "Mean CV R²": [simple_cv.mean(), multiple_cv.mean()],
    "CV Std": [simple_cv.std(), multiple_cv.std()]
})

display(cv_results)
"""))

    # Cell 22: Section 10
    cells.append(create_cell("markdown", """## 10. Model Comparison

For regression, MAE, RMSE and R² are used as practical regression evaluation measures.
The official syllabus text supplied by the student does not explicitly list these three by name,
so they should be described as supporting regression evaluation metrics, not as separate syllabus algorithms.
"""))

    # Cell 23: comparison table
    cells.append(create_cell("code", """comparison = pd.DataFrame({
    "Model": ["Simple Linear Regression", "Multiple Linear Regression"],
    "MAE":   [simple_mae, multiple_mae],
    "RMSE":  [simple_rmse, multiple_rmse],
    "R2":    [simple_r2, multiple_r2],
    "CV_R2": [simple_cv.mean(), multiple_cv.mean()]
})

display(comparison)
"""))

    # Cell 24: Section 11
    cells.append(create_cell("markdown", "## 11. Overfitting / Underfitting Check"))

    # Cell 25: Overfit analysis
    cells.append(create_cell("code", """train_simple   = simple_model.predict(X_simple_train)
train_multiple = multiple_model.predict(X_train)

fit_analysis = pd.DataFrame({
    "Model": ["Simple Linear Regression", "Multiple Linear Regression"],
    "Train R2": [
        r2_score(y_simple_train, train_simple),
        r2_score(y_train, train_multiple)
    ],
    "Test R2": [simple_r2, multiple_r2]
})

fit_analysis["Train-Test Gap"] = fit_analysis["Train R2"] - fit_analysis["Test R2"]
display(fit_analysis)
"""))

    # Cell 26: Section 12
    cells.append(create_cell("markdown", "## 12. Country-wise India Impact Analysis"))

    # Cell 27: Country groupby
    cells.append(create_cell("code", """country_summary = (
    df.groupby("country")["india_economic_impact"]
      .agg(["count", "mean", "min", "max"])
      .sort_values("mean", ascending=False)
)

display(country_summary.head(20))
"""))

    # Cell 28: Section 13
    cells.append(create_cell("markdown", "## 13. Country Comparison Visualization"))

    # Cell 29: bar chart
    cells.append(create_cell("code", """top = country_summary.head(10)

plt.figure(figsize=(10, 6))
plt.bar(top.index.astype(str), top["mean"], color="#06b6d4", edgecolor="#0e7490")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average India Economic Impact Score")
plt.title("Countries with Highest Average Impact on India's Economy")
plt.tight_layout()
plt.show()
"""))

    # Cell 30: Section 14
    cells.append(create_cell("markdown", """## 14. Live New-Event Prediction

The website will send a NEW country's/event's numerical indicators to the trained Multiple Linear Regression model.
**The model must NOT retrain when the user clicks Predict.**
"""))

    # Cell 31: predict function
    cells.append(create_cell("code", """def predict_india_impact(
    oil_price_change,
    commodity_price_change,
    trade_disruption,
    shipping_disruption,
    india_trade_exposure,
    india_energy_exposure,
    market_volatility
):
    new_data = pd.DataFrame([{
        "oil_price_change": oil_price_change,
        "commodity_price_change": commodity_price_change,
        "trade_disruption": trade_disruption,
        "shipping_disruption": shipping_disruption,
        "india_trade_exposure": india_trade_exposure,
        "india_energy_exposure": india_energy_exposure,
        "market_volatility": market_volatility
    }])
    return float(multiple_model.predict(new_data)[0])

# Example input. Replace these values with actual user input.
predicted = predict_india_impact(
    oil_price_change=10,
    commodity_price_change=7,
    trade_disruption=60,
    shipping_disruption=55,
    india_trade_exposure=70,
    india_energy_exposure=65,
    market_volatility=6
)

print("Predicted India Economic Impact:", predicted)
"""))

    # Cell 32: Section 15
    cells.append(create_cell("markdown", """## 15. Impact-Level Interpretation

These are display bands for the predicted numerical target, **not** an additional ML algorithm:
- **0–30** → LOW
- **31–60** → MODERATE
- **61–80** → HIGH
- **81–100** → CRITICAL

The actual target range must be checked against the dataset before using these bands in the final UI.
"""))

    # Cell 33: impact level
    cells.append(create_cell("code", """def impact_level(score):
    if score <= 30:   return "LOW"
    if score <= 60:   return "MODERATE"
    if score <= 80:   return "HIGH"
    return "CRITICAL"

print("Predicted Score:", round(predicted, 2))
print("Impact Level:", impact_level(predicted))
"""))

    # Cell 34: Section 16
    cells.append(create_cell("markdown", """## 16. Final Workflow

```
Historical Country/Event Data
    -> Preprocessing
    -> Simple Linear Regression + Multiple Linear Regression
    -> 5-Fold Cross Validation
    -> Evaluation (MAE, RMSE, R², Overfitting Check)
    -> Best Regression Model (Multiple Linear Regression)
    -> New User Input
    -> Predicted India Economic Impact
```

Do not claim the model predicts India's entire GDP.
It predicts the specific measurable India-impact target defined by the dataset.
"""))

    # Cell 35: Section 17
    cells.append(create_cell("markdown", """## 17. Model Saving

Save the trained model after validation.
The website should load the saved model and use it for new user predictions instead of retraining on every request.
"""))

    # Cell 36: save
    cells.append(create_cell("code", """import joblib

models_dir = "models"
if not os.path.exists(models_dir):
    models_dir = os.path.join("..", "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "feature4_india_economic_impact_model.joblib")
joblib.dump(multiple_model, model_path)
print("Saved:", model_path)
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs("notebooks", exist_ok=True)
    path1 = os.path.join("notebooks", "Feature_4_India_Economic_Impact_Intelligence.ipynb")
    path2 = "Feature_4_India_Economic_Impact_Intelligence.ipynb"
    for p in [path1, path2]:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)

    print(f"Notebook written: {path1}  ({len(cells)} cells)")

if __name__ == "__main__":
    build()
