"""
Trains, tunes, and evaluates 5 Syllabus Supervised Classifiers for Feature 5:
- Logistic Regression
- k-NN
- Decision Tree
- Random Forest
- SVM

Saves:
- models/feature5_india_energy_risk_model.joblib
- src/data/india_energy_summary.json
- src/data/india_energy_comparison.json
- src/data/india_energy_confusion_matrices.json
- src/data/india_energy_roc_data.json
- src/data/india_energy_country_analysis.json
- src/data/india_energy_samples.json
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
DATA_PATH = "data/feature5_india_energy_risk.csv"

def run_pipeline():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape}")

    required = {
        "country", "event_date", "oil_import_dependency", "oil_price_change",
        "energy_supply_disruption", "shipping_disruption", "india_energy_exposure",
        "strategic_route_exposure", "commodity_price_change", "energy_risk"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    numeric = [
        "oil_import_dependency", "oil_price_change", "energy_supply_disruption",
        "shipping_disruption", "india_energy_exposure", "strategic_route_exposure",
        "commodity_price_change"
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["country"] = df["country"].astype(str).str.strip()
    df["energy_risk"] = df["energy_risk"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates().dropna(subset=numeric + ["country", "event_date", "energy_risk"])
    after = len(df)
    print(f"Cleaned: before={before}, after={after}")

    features = numeric
    X = df[features]
    y = df["energy_risk"]

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 5 Syllabus Classifiers definition
    models = {
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
            ("model", SVC(kernel="linear", probability=True, max_iter=1500, random_state=RANDOM_STATE))
        ])
    }

    # Stratified 3-Fold Cross Validation on 3000 sample for speed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_idx, _ = train_test_split(np.arange(len(X_train)), train_size=min(3000, len(X_train)), stratify=y_train, random_state=RANDOM_STATE)
    X_cv_sample, y_cv_sample = X_train.iloc[cv_idx], y_train.iloc[cv_idx]
    
    tune_idx, _ = train_test_split(np.arange(len(X_train)), train_size=min(2500, len(X_train)), stratify=y_train, random_state=RANDOM_STATE)
    X_tune, y_tune = X_train.iloc[tune_idx], y_train.iloc[tune_idx]

    # Baseline training and evaluation
    baseline_results = []
    trained_baseline = {}
    for name, m in models.items():
        print(f"Training baseline: {name}...")
        if name == "SVM":
            m.fit(X_tune, y_tune)
        else:
            m.fit(X_train, y_train)
        pred = m.predict(X_test)
        trained_baseline[name] = m
        cv_scores = cross_val_score(m, X_cv_sample, y_cv_sample, cv=cv, scoring="f1_weighted", n_jobs=1)
        baseline_results.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
        })

    # Hyperparameter Tuning via GridSearchCV
    grids = {
        "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
        "k-NN": {"model__n_neighbors": [3, 5, 7]},
        "Decision Tree": {"model__max_depth": [5, 10, None]},
        "Random Forest": {"model__n_estimators": [80], "model__max_depth": [10, None]},
        "SVM": {"model__C": [1.0]}
    }

    tuned_models = {}
    best_params = {}
    for name, g in grids.items():
        print(f"Tuning {name}...")
        search = GridSearchCV(models[name], g, cv=cv, scoring="f1_weighted", n_jobs=1)
        search.fit(X_tune, y_tune)
        best_est = search.best_estimator_
        if name == "SVM":
            best_est.fit(X_tune, y_tune)
        else:
            best_est.fit(X_train, y_train)
        tuned_models[name] = best_est
        best_params[name] = search.best_params_
        print(f"Tuned {name}: best_score={search.best_score_:.4f}, params={search.best_params_}")

    # Final evaluation after tuning
    final_comparison = []
    confusion_matrices = {}
    roc_curves_data = {}
    labels = sorted(y.unique()) # ['Critical', 'High', 'Low', 'Moderate']

    ybin_test = label_binarize(y_test, classes=labels)

    for name, m in tuned_models.items():
        pred_test = m.predict(X_test)
        pred_train = m.predict(X_train)

        acc = float(accuracy_score(y_test, pred_test))
        prec = float(precision_score(y_test, pred_test, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, pred_test, average="weighted", zero_division=0))
        f1_test = float(f1_score(y_test, pred_test, average="weighted", zero_division=0))
        f1_train = float(f1_score(y_train, pred_train, average="weighted", zero_division=0))
        gap = float(f1_train - f1_test)

        cv_s = cross_val_score(m, X_cv_sample, y_cv_sample, cv=cv, scoring="f1_weighted", n_jobs=1)

        # ROC AUC
        roc_auc = None
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test)
            try:
                roc_auc = float(roc_auc_score(ybin_test, proba, multi_class="ovr", average="weighted"))
            except Exception:
                roc_auc = None

        final_comparison.append({
            "name": name,
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1_test * 100, 2),
            "train_f1": round(f1_train * 100, 2),
            "cv_f1_mean": round(float(cv_s.mean()) * 100, 2),
            "cv_f1_std": round(float(cv_s.std()) * 100, 2),
            "train_test_gap": round(gap * 100, 2),
            "roc_auc": round(roc_auc * 100, 2) if roc_auc else None,
            "best_params": best_params[name],
            "overfitting_status": "Overfitting Risk" if gap > 0.08 else "Well-Regularized" if gap >= 0 else "Underfitting Risk"
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, pred_test, labels=labels)
        confusion_matrices[name] = {
            "matrix": cm.tolist(),
            "labels": labels
        }

        # ROC Curve Points for interactive visualization
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test)
            curve_classes = []
            for i, l in enumerate(labels):
                if ybin_test[:, i].sum() in (0, len(ybin_test)):
                    continue
                fpr, tpr, _ = roc_curve(ybin_test[:, i], proba[:, i])
                curve_auc = float(auc(fpr, tpr))
                # Sample down points for compact JSON
                idx = np.linspace(0, len(fpr) - 1, min(15, len(fpr))).astype(int)
                points = [{"fpr": round(float(fpr[j]), 3), "tpr": round(float(tpr[j]), 3)} for j in idx]
                curve_classes.append({
                    "class": l,
                    "auc": round(curve_auc, 3),
                    "points": points
                })
            roc_curves_data[name] = curve_classes

    # Sort models by F1 descending
    final_comparison.sort(key=lambda x: x["f1"], reverse=True)
    best_name = final_comparison[0]["name"]
    best_model = tuned_models[best_name]
    print(f"\n>>> Selected Best Model: {best_name} (F1: {final_comparison[0]['f1']}%, CV F1: {final_comparison[0]['cv_f1_mean']}%)")

    # Serialize best model
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/feature5_india_energy_risk_model.joblib"
    joblib.dump(best_model, model_save_path, compress=3)
    print(f"Saved best model to {model_save_path}")

    # Country Risk Analysis
    country_risk_dist = (
        df.groupby("country")["energy_risk"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    country_rows = []
    # Get mean exposure and count
    country_meta = df.groupby("country").agg({
        "india_energy_exposure": "mean",
        "strategic_route_exposure": "mean",
        "energy_risk": "count"
    }).rename(columns={"energy_risk": "total_events"})

    for c in country_risk_dist.index:
        shares = country_risk_dist.loc[c].to_dict()
        crit = float(shares.get("Critical", 0.0) * 100)
        high = float(shares.get("High", 0.0) * 100)
        mod = float(shares.get("Moderate", 0.0) * 100)
        low = float(shares.get("Low", 0.0) * 100)
        meta = country_meta.loc[c]

        # Strategic threat rating
        threat_score = crit * 1.0 + high * 0.65 + mod * 0.35
        rating = "CRITICAL" if threat_score >= 50 else "HIGH" if threat_score >= 30 else "MODERATE" if threat_score >= 15 else "LOW"

        country_rows.append({
            "country": c,
            "total_events": int(meta["total_events"]),
            "energy_exposure_pct": round(float(meta["india_energy_exposure"]), 1),
            "route_exposure": round(float(meta["strategic_route_exposure"]), 1),
            "critical_share": round(crit, 1),
            "high_share": round(high, 1),
            "moderate_share": round(mod, 1),
            "low_share": round(low, 1),
            "composite_threat_score": round(threat_score, 1),
            "strategic_tier": rating
        })

    country_rows.sort(key=lambda x: x["composite_threat_score"], reverse=True)

    # 3 Realistic Curated Sample Scenarios for 1-click exploration
    samples = [
        {
            "id": "hormuz-blockade",
            "name": "Strait of Hormuz Naval Chokepoint Crisis",
            "country": "Iran",
            "tag": "Critical Maritime Chokepoint",
            "description": "Naval confrontation & drone interdictions near Bandar Abbas. ~85% of Persian Gulf crude/LNG transits this narrow 21-mile strait.",
            "inputs": {
                "oil_import_dependency": 87.5,
                "oil_price_change": 22.5,
                "energy_supply_disruption": 85.0,
                "shipping_disruption": 94.0,
                "india_energy_exposure": 18.5,
                "strategic_route_exposure": 95.0,
                "commodity_price_change": 16.0
            },
            "expected_risk": "Critical"
        },
        {
            "id": "red-sea-houthi",
            "name": "Red Sea / Bab el-Mandeb Tanker Assault",
            "country": "Yemen",
            "tag": "Suez & Cape Route Divergence",
            "description": "Anti-ship ballistic missile and drone strikes targeting commercial crude carriers transiting Bab el-Mandeb toward Suez.",
            "inputs": {
                "oil_import_dependency": 86.8,
                "oil_price_change": 14.0,
                "energy_supply_disruption": 68.0,
                "shipping_disruption": 88.0,
                "india_energy_exposure": 8.0,
                "strategic_route_exposure": 85.0,
                "commodity_price_change": 10.5
            },
            "expected_risk": "High"
        },
        {
            "id": "niger-delta-strike",
            "name": "Niger Delta Bonny Light Pipeline Sabotage",
            "country": "Nigeria",
            "tag": "Secondary Sweet Crude Interruption",
            "description": "Militant blast shuts down onshore export pipeline network, cutting sweet crude loadings to Indian coastal refineries.",
            "inputs": {
                "oil_import_dependency": 85.0,
                "oil_price_change": 6.5,
                "energy_supply_disruption": 58.0,
                "shipping_disruption": 38.0,
                "india_energy_exposure": 4.5,
                "strategic_route_exposure": 45.0,
                "commodity_price_change": 4.2
            },
            "expected_risk": "Moderate"
        },
        {
            "id": "us-permian-buffer",
            "name": "US Gulf Coast WTI Export Expansion",
            "country": "USA",
            "tag": "Diversified Atlantic Basin Flow",
            "description": "High inventory buffers and record US export loadings dampen international supply premiums; open Atlantic shipping lane.",
            "inputs": {
                "oil_import_dependency": 85.2,
                "oil_price_change": -3.5,
                "energy_supply_disruption": 15.0,
                "shipping_disruption": 20.0,
                "india_energy_exposure": 7.0,
                "strategic_route_exposure": 35.0,
                "commodity_price_change": -1.8
            },
            "expected_risk": "Low"
        }
    ]

    # Dataset Summary
    summary = {
        "total_records": int(len(df)),
        "cleaned_records": int(after),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "countries_count": int(df["country"].nunique()),
        "date_range_start": df["event_date"].min().strftime("%Y-%m-%d"),
        "date_range_end": df["event_date"].max().strftime("%Y-%m-%d"),
        "target_variable": "energy_risk",
        "classes": labels,
        "class_distribution": {k: int(v) for k, v in df["energy_risk"].value_counts().items()},
        "features": features,
        "best_model_name": best_name,
        "best_model_f1": final_comparison[0]["f1"],
        "best_model_cv_f1": final_comparison[0]["cv_f1_mean"]
    }

    # Save to src/data/
    os.makedirs("src/data", exist_ok=True)
    with open("src/data/india_energy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("src/data/india_energy_comparison.json", "w") as f:
        json.dump(final_comparison, f, indent=2)
    with open("src/data/india_energy_confusion_matrices.json", "w") as f:
        json.dump(confusion_matrices, f, indent=2)
    with open("src/data/india_energy_roc_data.json", "w") as f:
        json.dump(roc_curves_data, f, indent=2)
    with open("src/data/india_energy_country_analysis.json", "w") as f:
        json.dump(country_rows, f, indent=2)
    with open("src/data/india_energy_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("\nSuccessfully generated all feature 5 model artifacts and UI JSON files!")

if __name__ == "__main__":
    run_pipeline()
