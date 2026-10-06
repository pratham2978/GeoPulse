"""
Feature 8: India Supply-Route Disruption Intelligence
Training Script adhering strictly to syllabus:
- Module 3: Logistic Regression, k-NN, Decision Tree, Random Forest, SVM
- Module 5: Train/Test Split (80/20 Stratified), 5-Fold StratifiedKFold CV,
            Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC, Overfitting/Underfitting, GridSearchCV
- Module 6: Data preprocessing, evaluation, visualization data generation
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)

RANDOM_STATE = 42
DATA_PATH = "data/feature8_india_supply_route_risk.csv"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    numeric = [
        "route_disruption",
        "shipping_delay",
        "freight_cost_change",
        "trade_volume_exposure",
        "india_import_exposure",
        "india_export_exposure",
        "energy_route_exposure",
        "commodity_exposure",
        "conflict_intensity"
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["route"] = df["route"].astype(str).str.strip()
    df["country_or_region"] = df["country_or_region"].astype(str).str.strip()
    df["supply_route_risk"] = df["supply_route_risk"].astype(str).str.strip()

    df = df.drop_duplicates()
    df = df.dropna(subset=numeric + ["route", "country_or_region", "event_date", "supply_route_risk"])
    print(f"Cleaned dataset: {len(df)} rows")

    features = numeric
    X = df[features]
    y = df["supply_route_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

    # 5 Syllabus Classifiers
    base_models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "k-NN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))
        ])
    }

    # Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    baseline_results = []
    trained_baseline = {}

    for name, pipe in base_models.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        cv_scores[name] = {"mean": float(scores.mean()), "std": float(scores.std())}
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        trained_baseline[name] = pipe

        baseline_results.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "cv_f1_mean": cv_scores[name]["mean"],
            "cv_f1_std": cv_scores[name]["std"]
        })

    # Hyperparameter tuning
    grids = {
        "Logistic Regression": {"model__C": [0.1, 1, 10]},
        "k-NN": {"model__n_neighbors": [3, 5, 7, 9]},
        "Decision Tree": {"model__max_depth": [5, 10, 15, None]},
        "Random Forest": {"model__n_estimators": [100, 150], "model__max_depth": [8, 12, None]},
        "SVM": {"model__C": [0.1, 1, 10]}
    }

    tuned_models = {}
    tuned_best_params = {}
    tuned_results = []
    fit_analysis = []

    classes = sorted(list(y.unique()))

    confusion_matrices = {}
    roc_data = {}

    for name, grid in grids.items():
        print(f"Tuning {name}...")
        search = GridSearchCV(base_models[name], grid, cv=cv, scoring="f1_weighted", n_jobs=-1)
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        tuned_models[name] = best_pipe
        tuned_best_params[name] = search.best_params_

        # Metrics on train and test
        train_pred = best_pipe.predict(X_train)
        test_pred = best_pipe.predict(X_test)

        train_acc = float(accuracy_score(y_train, train_pred))
        test_acc = float(accuracy_score(y_test, test_pred))
        train_f1 = float(f1_score(y_train, train_pred, average="weighted", zero_division=0))
        test_f1 = float(f1_score(y_test, test_pred, average="weighted", zero_division=0))
        test_prec = float(precision_score(y_test, test_pred, average="weighted", zero_division=0))
        test_rec = float(recall_score(y_test, test_pred, average="weighted", zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_pred, labels=classes)
        confusion_matrices[name] = {
            "matrix": cm.tolist(),
            "labels": classes
        }

        # ROC-AUC
        roc_info = {}
        if hasattr(best_pipe, "predict_proba"):
            proba = best_pipe.predict_proba(X_test)
            y_bin = label_binarize(y_test, classes=classes)
            ovr_auc = float(roc_auc_score(y_bin, proba, multi_class="ovr", average="weighted"))
            roc_info["weighted_auc"] = round(ovr_auc, 4)
            curves = []
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                # downsample curve points for lightweight JSON
                stride = max(1, len(fpr) // 30)
                sub_fpr = [round(float(val), 4) for val in fpr[::stride]]
                sub_tpr = [round(float(val), 4) for val in tpr[::stride]]
                if sub_fpr[-1] != 1.0:
                    sub_fpr.append(1.0)
                    sub_tpr.append(1.0)
                curves.append({
                    "class": cls,
                    "auc": round(float(auc(fpr, tpr)), 4),
                    "fpr": sub_fpr,
                    "tpr": sub_tpr
                })
            roc_info["curves"] = curves
        else:
            roc_info["weighted_auc"] = None
            roc_info["curves"] = []

        roc_data[name] = roc_info

        # Overfitting check
        gap = round(train_f1 - test_f1, 4)
        if gap > 0.15:
            fit_status = "High Overfitting Risk"
        elif gap > 0.06:
            fit_status = "Mild Overfitting"
        elif gap < -0.05:
            fit_status = "Underfitting"
        else:
            fit_status = "Well Generalized"

        fit_analysis.append({
            "model": name,
            "train_f1": round(train_f1, 4),
            "test_f1": round(test_f1, 4),
            "train_test_gap": gap,
            "fit_status": fit_status
        })

        tuned_results.append({
            "model": name,
            "accuracy": round(test_acc, 4),
            "precision": round(test_prec, 4),
            "recall": round(test_rec, 4),
            "f1": round(test_f1, 4),
            "cv_f1": round(float(search.best_score_), 4),
            "best_params": search.best_params_,
            "roc_auc": roc_info.get("weighted_auc")
        })

    tuned_results_sorted = sorted(tuned_results, key=lambda x: x["f1"], reverse=True)
    best_model_name = tuned_results_sorted[0]["model"]
    best_model = tuned_models[best_model_name]
    print(f"\nChampion Model: {best_model_name} with F1 = {tuned_results_sorted[0]['f1']}")

    # Save champion model
    os.makedirs("models", exist_ok=True)
    model_path = "models/feature8_india_supply_route_risk_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    # Route-wise risk analysis
    route_risk = (
        df.groupby(["route", "supply_route_risk"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    route_risk_list = []
    for _, row in route_risk.iterrows():
        route_name = row["route"]
        tot = sum(row[cls] for cls in classes if cls in row)
        crit_pct = round((row.get("Critical", 0) + row.get("High", 0)) / tot * 100, 1) if tot > 0 else 0
        route_risk_list.append({
            "route": route_name,
            "low": int(row.get("Low", 0)),
            "moderate": int(row.get("Moderate", 0)),
            "high": int(row.get("High", 0)),
            "critical": int(row.get("Critical", 0)),
            "total": int(tot),
            "high_critical_pct": crit_pct
        })

    route_summary = (
        df.groupby("route")[
            [
                "route_disruption",
                "shipping_delay",
                "freight_cost_change",
                "india_import_exposure",
                "energy_route_exposure",
                "commodity_exposure",
                "conflict_intensity"
            ]
        ]
        .mean()
        .reset_index()
    )
    route_metrics_list = []
    for _, row in route_summary.iterrows():
        route_metrics_list.append({
            "route": row["route"],
            "avg_disruption": round(float(row["route_disruption"]), 1),
            "avg_delay_days": round(float(row["shipping_delay"]), 1),
            "avg_freight_change_pct": round(float(row["freight_cost_change"]), 1),
            "avg_import_exposure": round(float(row["india_import_exposure"]), 1),
            "avg_energy_exposure": round(float(row["energy_route_exposure"]), 1),
            "avg_commodity_exposure": round(float(row["commodity_exposure"]), 1),
            "avg_conflict_intensity": round(float(row["conflict_intensity"]), 1)
        })

    # Prepare export JSONs for UI
    os.makedirs("src/data", exist_ok=True)

    summary_data = {
        "dataset_name": "feature8_india_supply_route_risk.csv",
        "total_records": len(df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "features": features,
        "classes": classes,
        "class_distribution": df["supply_route_risk"].value_counts().to_dict(),
        "champion_model": best_model_name,
        "champion_f1": tuned_results_sorted[0]["f1"],
        "champion_accuracy": tuned_results_sorted[0]["accuracy"],
        "champion_params": tuned_best_params[best_model_name]
    }

    with open("src/data/supply_route_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    with open("src/data/supply_route_models.json", "w") as f:
        json.dump({
            "models": tuned_results_sorted,
            "fit_analysis": fit_analysis,
            "champion_model": best_model_name
        }, f, indent=2)

    with open("src/data/supply_route_confusion.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)

    with open("src/data/supply_route_roc.json", "w") as f:
        json.dump(roc_data, f, indent=2)

    with open("src/data/supply_route_analysis.json", "w") as f:
        json.dump({
            "route_risk_distribution": route_risk_list,
            "route_metrics": route_metrics_list
        }, f, indent=2)

    # 8 curated realistic sample scenarios
    samples = [
        {
            "id": "red_sea_houthi",
            "name": "Bab el-Mandeb / Red Sea Crisis (Houthi Attacks)",
            "route": "Bab el-Mandeb / Red Sea",
            "region": "Middle East / Red Sea",
            "description": "Commercial missile attacks forcing vessel rerouting via Cape of Good Hope, severe freight spikes.",
            "inputs": {
                "route_disruption": 85.0,
                "shipping_delay": 18.0,
                "freight_cost_change": 165.0,
                "trade_volume_exposure": 82.0,
                "india_import_exposure": 78.0,
                "india_export_exposure": 88.0,
                "energy_route_exposure": 65.0,
                "commodity_exposure": 74.0,
                "conflict_intensity": 90.0
            }
        },
        {
            "id": "hormuz_blockade",
            "name": "Strait of Hormuz Escalation / Tanker Seizure",
            "route": "Strait of Hormuz",
            "region": "Persian Gulf",
            "description": "Military standoff closing maritime lanes for crude oil & LNG tankers to Mangalore, Jamnagar, and Kochi.",
            "inputs": {
                "route_disruption": 92.0,
                "shipping_delay": 22.0,
                "freight_cost_change": 190.0,
                "trade_volume_exposure": 95.0,
                "india_import_exposure": 94.0,
                "india_export_exposure": 52.0,
                "energy_route_exposure": 98.0,
                "commodity_exposure": 86.0,
                "conflict_intensity": 94.0
            }
        },
        {
            "id": "malacca_congestion",
            "name": "Malacca Strait Naval Drills & Congestion",
            "route": "Malacca Strait",
            "region": "Southeast Asia / Indo-Pacific",
            "description": "Geopolitical tension in the South China Sea causing port delays in Singapore and Strait choke slowdown.",
            "inputs": {
                "route_disruption": 55.0,
                "shipping_delay": 8.5,
                "freight_cost_change": 45.0,
                "trade_volume_exposure": 75.0,
                "india_import_exposure": 68.0,
                "india_export_exposure": 62.0,
                "energy_route_exposure": 40.0,
                "commodity_exposure": 58.0,
                "conflict_intensity": 50.0
            }
        },
        {
            "id": "suez_canal_grounding",
            "name": "Suez Canal Blockage / Grounding",
            "route": "Suez Canal",
            "region": "Egypt / Mediterranean",
            "description": "Physical maritime obstruction delaying European container traffic bound for Nhava Sheva and Mundra.",
            "inputs": {
                "route_disruption": 78.0,
                "shipping_delay": 14.0,
                "freight_cost_change": 110.0,
                "trade_volume_exposure": 80.0,
                "india_import_exposure": 72.0,
                "india_export_exposure": 82.0,
                "energy_route_exposure": 52.0,
                "commodity_exposure": 65.0,
                "conflict_intensity": 35.0
            }
        },
        {
            "id": "cape_detour",
            "name": "Cape of Good Hope Reroute (High Weather & Cost)",
            "route": "Cape of Good Hope",
            "region": "South Africa / Southern Ocean",
            "description": "Extended voyage transit +14 days with elevated bunker fuel and insurance premiums for Indian exporters.",
            "inputs": {
                "route_disruption": 45.0,
                "shipping_delay": 15.0,
                "freight_cost_change": 85.0,
                "trade_volume_exposure": 65.0,
                "india_import_exposure": 58.0,
                "india_export_exposure": 68.0,
                "energy_route_exposure": 45.0,
                "commodity_exposure": 50.0,
                "conflict_intensity": 20.0
            }
        },
        {
            "id": "black_sea_grain",
            "name": "Black Sea Drone Attacks on Commercial Shipping",
            "route": "Black Sea & Bosporus",
            "region": "Eastern Europe / Black Sea",
            "description": "Sunflower oil and fertilizer shipments to India subjected to maritime war-risk insurance surcharges.",
            "inputs": {
                "route_disruption": 72.0,
                "shipping_delay": 12.0,
                "freight_cost_change": 95.0,
                "trade_volume_exposure": 48.0,
                "india_import_exposure": 56.0,
                "india_export_exposure": 28.0,
                "energy_route_exposure": 32.0,
                "commodity_exposure": 72.0,
                "conflict_intensity": 85.0
            }
        },
        {
            "id": "mozambique_piracy",
            "name": "Mozambique Channel Security Alert",
            "route": "Mozambique Channel",
            "region": "East Africa / Indian Ocean",
            "description": "Insurgency flare-up along coastal gas infrastructure, mild disruption to East African coal and mineral trade.",
            "inputs": {
                "route_disruption": 38.0,
                "shipping_delay": 4.5,
                "freight_cost_change": 25.0,
                "trade_volume_exposure": 35.0,
                "india_import_exposure": 42.0,
                "india_export_exposure": 30.0,
                "energy_route_exposure": 38.0,
                "commodity_exposure": 45.0,
                "conflict_intensity": 45.0
            }
        },
        {
            "id": "persian_gulf_routine",
            "name": "Persian Gulf Coastal Normal Patrols",
            "route": "Persian Gulf Coastal",
            "region": "Gulf Cooperation Council",
            "description": "Routine maritime operations with slight customs inspections and baseline normal freight conditions.",
            "inputs": {
                "route_disruption": 18.0,
                "shipping_delay": 1.5,
                "freight_cost_change": 6.0,
                "trade_volume_exposure": 55.0,
                "india_import_exposure": 58.0,
                "india_export_exposure": 45.0,
                "energy_route_exposure": 52.0,
                "commodity_exposure": 40.0,
                "conflict_intensity": 15.0
            }
        }
    ]

    with open("src/data/supply_route_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("All exports successfully generated!")

if __name__ == "__main__":
    main()
