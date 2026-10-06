"""
Generates Feature_6_India_Oil_Commodity_Shock.ipynb
16 cells matching the user's exact specification and syllabus guidelines.
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

    # Cell 1: Title and Objective
    cells.append(create_cell("code", """# Feature 6 — India Oil & Commodity Shock Analysis 🇮🇳
#
# Objective:
# Analyze how geopolitical and commodity-price shocks
# affect India's economic/commodity impact using
# Multiple Linear Regression.
#
# Workflow:
# Real data → preprocessing → Multiple Linear Regression
# → cross-validation → evaluation → interpretation
# → new event prediction."""))

    # Cell 2: Imports & Dataset
    cells.append(create_cell("code", """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
DATA_PATH = "data/feature6_india_commodity_shock.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. Add the real historical CSV first."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
display(df.head())"""))

    # Cell 3: Dataset Validation & Preprocessing
    cells.append(create_cell("code", """required = {
    "country",
    "event_date",
    "crude_oil_change",
    "natural_gas_change",
    "gold_price_change",
    "essential_commodity_change",
    "trade_disruption",
    "shipping_disruption",
    "conflict_intensity",
    "india_import_dependency",
    "india_commodity_impact"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

numeric = [
    "crude_oil_change",
    "natural_gas_change",
    "gold_price_change",
    "essential_commodity_change",
    "trade_disruption",
    "shipping_disruption",
    "conflict_intensity",
    "india_import_dependency",
    "india_commodity_impact"
]

for c in numeric:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["event_date"] = pd.to_datetime(
    df["event_date"],
    errors="coerce"
)

df["country"] = df["country"].astype(str).str.strip()

before = len(df)

df = df.drop_duplicates()
df = df.dropna(
    subset=numeric + ["country", "event_date"]
)

print("Before:", before)
print("After:", len(df))

display(df.head())"""))

    # Cell 4: Impact Distribution
    cells.append(create_cell("code", """plt.figure(figsize=(8, 5))

plt.hist(
    df["india_commodity_impact"],
    bins=20,
    color="#f59e0b",
    edgecolor="black",
    alpha=0.85
)

plt.title("India Commodity Impact Distribution", fontsize=13, fontweight="bold")
plt.xlabel("India Commodity Impact Index (0-100)", fontsize=11)
plt.ylabel("Number of Historical Records", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()"""))

    # Cell 5: Feature Selection & Train/Test
    cells.append(create_cell("code", """features = [
    "crude_oil_change",
    "natural_gas_change",
    "gold_price_change",
    "essential_commodity_change",
    "trade_disruption",
    "shipping_disruption",
    "conflict_intensity",
    "india_import_dependency"
]

target = "india_commodity_impact"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Training:", len(X_train))
print("Testing:", len(X_test))"""))

    # Cell 6: Multiple Linear Regression
    cells.append(create_cell("code", """model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
        LinearRegression()
    )
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model trained successfully.")"""))

    # Cell 7: Model Evaluation
    cells.append(create_cell("code", """mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("========== MODEL PERFORMANCE ==========")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))"""))

    # Cell 8: Cross Validation
    cells.append(create_cell("code", """cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print("========== CROSS VALIDATION ==========")
print("CV Scores:", np.round(cv_scores, 4))
print("Mean CV R²:", round(cv_scores.mean(), 4))
print("CV Std:", round(cv_scores.std(), 4))"""))

    # Cell 9: Actual vs Predicted
    cells.append(create_cell("code", """plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    color="#06b6d4",
    alpha=0.75,
    edgecolor="black"
)

# Reference diagonal ideal fit line
min_val = min(y_test.min(), y_pred.min()) - 2
max_val = max(y_test.max(), y_pred.max()) + 2
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.8, label="Ideal Fit (y = x)")

plt.xlabel("Actual India Commodity Impact", fontsize=11)
plt.ylabel("Predicted India Commodity Impact", fontsize=11)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

plt.title(
    "Actual vs Predicted India Commodity Impact (Multiple Linear Regression)",
    fontsize=13,
    fontweight="bold"
)

plt.tight_layout()
plt.show()"""))

    # Cell 10: Regression Coefficients
    cells.append(create_cell("code", """linear_model = model.named_steps["model"]

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": linear_model.coef_
})

coefficients["Absolute Impact"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute Impact",
    ascending=False
)

display(coefficients)"""))

    # Cell 11: Coefficient Visualization
    cells.append(create_cell("code", """plt.figure(figsize=(10, 6))

colors = ["#10b981" if c > 0 else "#f43f5e" for c in coefficients["Coefficient"]]

plt.barh(
    coefficients["Feature"],
    coefficients["Coefficient"],
    color=colors,
    edgecolor="black",
    alpha=0.85
)

plt.axvline(0, color="white", linewidth=1, linestyle="-")
plt.xlabel("Regression Coefficient (Influence on Impact Index)", fontsize=11)
plt.ylabel("Macro/Geopolitical Feature", fontsize=11)
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.title(
    "Factors Affecting India Commodity Impact (Linear Association)",
    fontsize=13,
    fontweight="bold"
)

plt.tight_layout()
plt.show()"""))

    # Cell 12: Overfitting / Underfitting Check
    cells.append(create_cell("code", """train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

train_r2 = r2_score(
    y_train,
    train_prediction
)

test_r2 = r2_score(
    y_test,
    test_prediction
)

print("========== FIT CHECK ==========")
print("Training R²:", round(train_r2, 4))
print("Testing R² :", round(test_r2, 4))
print("Gap        :", round(train_r2 - test_r2, 4))

if train_r2 - test_r2 > 0.15:
    print("Possible overfitting detected.")

elif train_r2 < 0.50 and test_r2 < 0.50:
    print("Model may be underfitting.")

else:
    print("No major overfitting/underfitting indication. Model is well-regularized.")"""))

    # Cell 13: Country-wise Analysis
    cells.append(create_cell("code", """country_impact = (
    df.groupby("country")["india_commodity_impact"]
    .agg(["mean", "min", "max", "count"])
    .sort_values("mean", ascending=False)
)

display(
    country_impact.head(30)
)"""))

    # Cell 14: Live Prediction
    cells.append(create_cell("code", """def predict_india_commodity_impact(
    crude_oil_change,
    natural_gas_change,
    gold_price_change,
    essential_commodity_change,
    trade_disruption,
    shipping_disruption,
    conflict_intensity,
    india_import_dependency
):

    row = pd.DataFrame([{
        "crude_oil_change": crude_oil_change,
        "natural_gas_change": natural_gas_change,
        "gold_price_change": gold_price_change,
        "essential_commodity_change": essential_commodity_change,
        "trade_disruption": trade_disruption,
        "shipping_disruption": shipping_disruption,
        "conflict_intensity": conflict_intensity,
        "india_import_dependency": india_import_dependency
    }])

    prediction = model.predict(row)[0]

    return {
        "predicted_india_commodity_impact": round(float(prediction), 2)
    }


# Replace these values with website user input.

result = predict_india_commodity_impact(
    15,
    10,
    5,
    8,
    6,
    7,
    8,
    70
)

print("Prediction Result:", result)"""))

    # Cell 15: Save Model
    cells.append(create_cell("code", """import joblib

os.makedirs(
    "models",
    exist_ok=True
)

MODEL_PATH = (
    "models/"
    "feature6_india_commodity_shock_model.joblib"
)

joblib.dump(
    model,
    MODEL_PATH
)

print("Saved:", MODEL_PATH)"""))

    # Cell 16: Final Workflow
    cells.append(create_cell("code", """# Final workflow:
#
# Real historical data
#        ↓
# Data preprocessing
#        ↓
# Feature selection
#        ↓
# Train/Test Split
#        ↓
# Multiple Linear Regression
#        ↓
# Cross Validation
#        ↓
# MAE / RMSE / R²
#        ↓
# Overfitting / Underfitting Check
#        ↓
# Coefficient Analysis
#        ↓
# Country-wise Analysis
#        ↓
# Save Trained Model
#        ↓
# New User Input
#        ↓
# Actual India Commodity Impact Prediction
#
# Never use fake metrics.
# Never use fabricated training rows.
# Never replace Multiple Linear Regression with
# an out-of-syllabus ML algorithm."""))

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

    out_file = "Feature_6_India_Oil_Commodity_Shock.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {out_file} with {len(cells)} cells.")

    nb_dir_file = os.path.join("notebooks", out_file)
    with open(nb_dir_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {nb_dir_file} with {len(cells)} cells.")

if __name__ == "__main__":
    build()
