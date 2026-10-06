"""
Generates the comprehensive, syllabus-compliant Jupyter Notebook:
Feature_3_Sentiment_Analysis.ipynb

Covers all 17 mandatory sections and 35 cells matching the academic specification.
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

def build_notebook():
    cells = []

    # Cell 1: Frontmatter & Syllabus Mapping
    cells.append(create_cell("markdown", """# Feature 3 — Sentiment Analysis
## GeoPulse AI — Global Conflict Impact Intelligence Platform

Train supervised ML models to classify geopolitical/news text into sentiment categories.

### Strict syllabus mapping
- **Module 3:** Logistic Regression, k-NN, Decision Tree, Random Forest, SVM
- **Module 5:** Train/Test Split, Cross Validation, Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC, Overfitting/Underfitting, Hyperparameter Tuning
- **Module 6:** Data preprocessing, model development, evaluation, visualization and result interpretation.

*No Naive Bayes, BERT, Transformers, LSTM, XGBoost or other out-of-syllabus classifiers.*

### Dataset
Place a real labelled CSV at:
`data/feature3_sentiment.csv`

Required columns:
- `text`
- `label`

Recommended labels: Positive, Negative, Neutral.
No synthetic training data is created.
"""))

    # Cell 2: Environment setup
    cells.append(create_cell("code", """import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, roc_auc_score
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
print("Environment ready.")
"""))

    # Cell 3: Section 1 Markdown
    cells.append(create_cell("markdown", "## 1. Load Real Dataset"))

    # Cell 4: Section 1 Code
    cells.append(create_cell("code", """DATA_PATH = "data/feature3_sentiment.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("..", "data", "feature3_sentiment.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\\n"
        "Place a real labelled CSV there with columns: text, label."
    )

df = pd.read_csv(DATA_PATH)
required = {"text", "label"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

print("Shape:", df.shape)
display(df.head())
"""))

    # Cell 5: Section 2 Markdown
    cells.append(create_cell("markdown", "## 2. Dataset Inspection"))

    # Cell 6: Section 2 Code
    cells.append(create_cell("code", """print("Columns:", list(df.columns))
print("\\nData types:")
display(df.dtypes)

print("\\nMissing values:")
display(df[["text", "label"]].isna().sum())

print("\\nDuplicate rows:", df.duplicated().sum())
print("\\nClass distribution:")
display(df["label"].value_counts())
"""))

    # Cell 7: Section 3 Markdown
    cells.append(create_cell("markdown", "## 3. Data Cleaning"))

    # Cell 8: Section 3 Code
    cells.append(create_cell("code", """df = df[["text", "label"]].copy()
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

df = df[(df["text"] != "") & (df["label"] != "")]
df = df.drop_duplicates()

print("Records after cleaning:", len(df))
display(df["label"].value_counts())
"""))

    # Cell 9: Section 4 Markdown
    cells.append(create_cell("markdown", """## 4. Feature Preparation
CountVectorizer is used only as an implementation-level feature preparation step to convert text into numerical features for classical supervised ML. It is not treated as a separate ML algorithm.
"""))

    # Cell 10: Section 4 Code
    cells.append(create_cell("code", """X = df["text"]
y = df["label"]

print("Classes:", sorted(y.unique()))
print("Number of classes:", y.nunique())
"""))

    # Cell 11: Section 5 Markdown
    cells.append(create_cell("markdown", "## 5. Sentiment Distribution"))

    # Cell 12: Section 5 Code
    cells.append(create_cell("code", """plt.figure(figsize=(8, 5))
y.value_counts().plot(kind="bar", color=["#06b6d4", "#10b981", "#f43f5e"])
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""))

    # Cell 13: Section 6 Markdown
    cells.append(create_cell("markdown", "## 6. Stratified Train-Test Split"))

    # Cell 14: Section 6 Code
    cells.append(create_cell("code", """X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Training:", len(X_train))
print("Testing:", len(X_test))
"""))

    # Cell 15: Section 7 Markdown
    cells.append(create_cell("markdown", "## 7. Syllabus-Aligned Models"))

    # Cell 16: Section 7 Code
    cells.append(create_cell("code", """models = {
    "Logistic Regression": Pipeline([
        ("features", CountVectorizer(lowercase=True, stop_words="english", max_features=10000)),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ]),
    "k-NN": Pipeline([
        ("features", CountVectorizer(lowercase=True, stop_words="english", max_features=10000)),
        ("scale", StandardScaler(with_mean=False)),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Decision Tree": Pipeline([
        ("features", CountVectorizer(lowercase=True, stop_words="english", max_features=10000)),
        ("model", DecisionTreeClassifier(max_depth=16, random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("features", CountVectorizer(lowercase=True, stop_words="english", max_features=10000)),
        ("model", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1))
    ]),
    "SVM": Pipeline([
        ("features", CountVectorizer(lowercase=True, stop_words="english", max_features=10000)),
        ("model", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE))
    ])
}
print("Initialized Models:", list(models.keys()))
"""))

    # Cell 17: Section 8 Markdown
    cells.append(create_cell("markdown", "## 8. Train and Evaluate All Models"))

    # Cell 18: Section 8 Code
    cells.append(create_cell("code", """results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, pred, average="weighted", zero_division=0)
    })
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
display(results_df)
"""))

    # Cell 19: Section 9 Markdown
    cells.append(create_cell("markdown", "## 9. Five-Fold Stratified Cross Validation"))

    # Cell 20: Section 9 Code
    cells.append(create_cell("code", """cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
    cv_rows.append({
        "Model": name,
        "Mean CV F1": scores.mean(),
        "CV Std": scores.std()
    })

cv_df = pd.DataFrame(cv_rows).sort_values("Mean CV F1", ascending=False).reset_index(drop=True)
display(cv_df)
"""))

    # Cell 21: Section 10 Markdown
    cells.append(create_cell("markdown", "## 10. Confusion Matrix — Current Best Model"))

    # Cell 22: Section 10 Code
    cells.append(create_cell("code", """best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_pred = best_model.predict(X_test)

labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.show()

print(classification_report(y_test, best_pred, zero_division=0))
"""))

    # Cell 23: Section 11 Markdown
    cells.append(create_cell("markdown", "## 11. ROC Curve and AUC"))

    # Cell 24: Section 11 Code
    cells.append(create_cell("code", """classes = sorted(y.unique())
y_bin = label_binarize(y_test, classes=classes)

if hasattr(best_model, "predict_proba"):
    proba = best_model.predict_proba(X_test)

    if len(classes) > 2:
        auc_score = roc_auc_score(y_bin, proba, multi_class="ovr", average="weighted")
    else:
        auc_score = roc_auc_score(y_bin, proba[:, 1])

    print("Multiclass Weighted ROC-AUC:", auc_score)

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multiclass ROC — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("ROC-AUC is not available for the selected model configuration.")
"""))

    # Cell 25: Section 12 Markdown
    cells.append(create_cell("markdown", "## 12. Overfitting / Underfitting Check"))

    # Cell 26: Section 12 Code
    cells.append(create_cell("code", """fit_rows = []

for name, model in models.items():
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_f1 = f1_score(y_train, train_pred, average="weighted", zero_division=0)
    test_f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    fit_rows.append({
        "Model": name,
        "Train F1": train_f1,
        "Test F1": test_f1,
        "Train-Test Gap": train_f1 - test_f1
    })

fit_df = pd.DataFrame(fit_rows).sort_values("Test F1", ascending=False)
display(fit_df)
"""))

    # Cell 27: Section 13 Markdown
    cells.append(create_cell("markdown", "## 13. Hyperparameter Tuning"))

    # Cell 28: Section 13 Code
    cells.append(create_cell("code", """param_grids = {
    "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
    "k-NN": {"model__n_neighbors": [3, 5, 7, 9]},
    "Decision Tree": {"model__max_depth": [5, 10, 16, None]},
    "Random Forest": {
        "model__n_estimators": [100, 150],
        "model__max_depth": [10, 16, None]
    },
    "SVM": {"model__C": [0.1, 1.0, 10.0]}
}

tuned_models = {}
tuned_rows = []

for name, grid in param_grids.items():
    search = GridSearchCV(
        models[name], grid, scoring="f1_weighted", cv=cv, n_jobs=-1
    )
    search.fit(X_train, y_train)
    tuned_models[name] = search.best_estimator_
    tuned_rows.append({
        "Model": name,
        "Best CV F1": search.best_score_,
        "Best Parameters": search.best_params_
    })

tuned_df = pd.DataFrame(tuned_rows).sort_values("Best CV F1", ascending=False)
display(tuned_df)
"""))

    # Cell 29: Section 14 Markdown
    cells.append(create_cell("markdown", "## 14. Final Model Comparison"))

    # Cell 30: Section 14 Code
    cells.append(create_cell("code", """final_rows = []

for name, model in tuned_models.items():
    pred = model.predict(X_test)
    final_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1": f1_score(y_test, pred, average="weighted", zero_division=0)
    })

final_df = pd.DataFrame(final_rows).sort_values("F1", ascending=False).reset_index(drop=True)
display(final_df)

BEST_MODEL_NAME = final_df.iloc[0]["Model"]
BEST_MODEL = tuned_models[BEST_MODEL_NAME]

print("BEST MODEL:", BEST_MODEL_NAME)
"""))

    # Cell 31: Section 15 Markdown
    cells.append(create_cell("markdown", "## 15. New / Unseen User Input Prediction"))

    # Cell 32: Section 15 Code
    cells.append(create_cell("code", """def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide a non-empty news text.")

    prediction = BEST_MODEL.predict([text])[0]
    result = {"prediction": prediction}

    if hasattr(BEST_MODEL, "predict_proba"):
        probabilities = BEST_MODEL.predict_proba([text])[0]
        result["probabilities"] = dict(zip(BEST_MODEL.classes_, probabilities))

    return result

new_article = (
    "Diplomatic talks between the two countries have reduced tensions "
    "and markets reacted positively to the announcement."
)

result = predict_sentiment(new_article)
print("Predicted Sentiment:", result["prediction"])

if "probabilities" in result:
    print("\\nClass probabilities:")
    for cls, p in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {p:.4f}")
"""))

    # Cell 33: Section 16 Markdown
    cells.append(create_cell("markdown", "## 16. Save Final Model"))

    # Cell 34: Section 16 Code
    cells.append(create_cell("code", """# Run this cell after verifying the final model.
# The website should load this saved model instead of retraining on every request.

import joblib
models_dir = "models"
if not os.path.exists(models_dir):
    models_dir = os.path.join("..", "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "feature3_sentiment_model.joblib")
joblib.dump(BEST_MODEL, model_path)

print("Saved:", model_path)
"""))

    # Cell 35: Section 17 Markdown
    cells.append(create_cell("markdown", """## 17. Final Result
The completed Feature 3 workflow is:
Real Dataset → Preprocessing → Feature Preparation → Train/Test Split → 5 Syllabus Models → Cross Validation → Hyperparameter Tuning → Evaluation → Best Model → Save Model → New User Input → Actual Sentiment Prediction

No fake metrics or hardcoded predictions should be used.
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Save to notebooks/Feature_3_Sentiment_Analysis.ipynb
    out_dir = "notebooks"
    os.makedirs(out_dir, exist_ok=True)
    out_path_1 = os.path.join(out_dir, "Feature_3_Sentiment_Analysis.ipynb")
    with open(out_path_1, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    # Save to root Feature_3_Sentiment_Analysis.ipynb as well
    out_path_2 = "Feature_3_Sentiment_Analysis.ipynb"
    with open(out_path_2, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Notebook written to {out_path_1} and {out_path_2} with {len(cells)} cells.")

if __name__ == "__main__":
    build_notebook()
