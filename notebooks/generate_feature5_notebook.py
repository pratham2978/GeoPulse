"""
Generates Feature_5_India_Energy_Supply_Risk_Intelligence.ipynb
16 cells matching the syllabus specification exactly.
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
    cells.append(create_cell("markdown", """# Feature 5 — India Energy Supply Risk Intelligence 🇮🇳
## GeoPulse AI — Global Conflict Impact Intelligence Platform

**Objective:** Predict whether a geopolitical event / country creates Low, Moderate, High or Critical energy-supply risk for India.

### Core Question
> How severely does this geopolitical event or country disrupt India's crude oil, gas, and maritime energy supply corridors?

### Strict Syllabus Mapping
- **Module 3:** Supervised Classification — Logistic Regression, k-NN, Decision Tree, Random Forest, Support Vector Machine (SVM)
- **Module 5:** Train/Test Split, Cross Validation, Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC, Overfitting/Underfitting Diagnostics, Hyperparameter Tuning (GridSearchCV)
- **Module 6:** Data Preprocessing, Pipeline Development, Model Evaluation, Multi-Class ROC Visualization, Country-Level Vulnerability Interpretation

*Strict compliance note: No Naive Bayes, XGBoost, LSTM, Transformers or other out-of-syllabus classifiers are used.*
"""))

    # Cell 2: Imports and Data Loading
    cells.append(create_cell("code", """import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve, auc

RANDOM_STATE=42
DATA_PATH="data/feature5_india_energy_risk.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Add the real labelled CSV first.")
df=pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
display(df.head())"""))

    # Cell 3: Data Preprocessing and Validation
    cells.append(create_cell("code", """required={"country","event_date","oil_import_dependency","oil_price_change","energy_supply_disruption","shipping_disruption","india_energy_exposure","strategic_route_exposure","commodity_price_change","energy_risk"}
missing=required-set(df.columns)
if missing: raise ValueError(f"Missing columns: {sorted(missing)}")

numeric=["oil_import_dependency","oil_price_change","energy_supply_disruption","shipping_disruption","india_energy_exposure","strategic_route_exposure","commodity_price_change"]
for c in numeric: df[c]=pd.to_numeric(df[c],errors="coerce")
df["event_date"]=pd.to_datetime(df["event_date"],errors="coerce")
df["country"]=df["country"].astype(str).str.strip()
df["energy_risk"]=df["energy_risk"].astype(str).str.strip()
before=len(df)
df=df.drop_duplicates().dropna(subset=numeric+["country","event_date","energy_risk"])
print("Before:",before,"After:",len(df))
display(df["energy_risk"].value_counts())"""))

    # Cell 4: Class Distribution Visualization
    cells.append(create_cell("code", """plt.figure(figsize=(8,5))
df["energy_risk"].value_counts().plot(kind="bar", color=["#f59e0b", "#f97316", "#10b981", "#f43f5e"], edgecolor="black", alpha=0.85)
plt.title("India Energy Supply Risk Distribution (Real Geopolitical Dataset)", fontsize=13, fontweight="bold")
plt.xlabel("Risk Level", fontsize=11); plt.ylabel("Records", fontsize=11); plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout(); plt.show()"""))

    # Cell 5: Stratified Train-Test Split
    cells.append(create_cell("code", """features=["oil_import_dependency","oil_price_change","energy_supply_disruption","shipping_disruption","india_energy_exposure","strategic_route_exposure","commodity_price_change"]
X=df[features]; y=df["energy_risk"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
print("Training records:",len(X_train),"Testing records:",len(X_test))
display(pd.DataFrame({"Train Distribution": y_train.value_counts(), "Test Distribution": y_test.value_counts()}))"""))

    # Cell 6: Train 5 Syllabus Supervised Models
    cells.append(create_cell("code", """models={
"Logistic Regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,random_state=42))]),
"k-NN":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=5))]),
"Decision Tree":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",DecisionTreeClassifier(max_depth=10,random_state=42))]),
"Random Forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1))]),
"SVM":Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",SVC(kernel="linear",probability=True,random_state=42))])
}
results=[]; trained={}
for name,m in models.items():
    m.fit(X_train,y_train); p=m.predict(X_test); trained[name]=m
    results.append({
        "Model":name,
        "Accuracy":accuracy_score(y_test,p),
        "Precision":precision_score(y_test,p,average="weighted",zero_division=0),
        "Recall":recall_score(y_test,p,average="weighted",zero_division=0),
        "F1":f1_score(y_test,p,average="weighted",zero_division=0)
    })
results=pd.DataFrame(results).sort_values("F1",ascending=False).reset_index(drop=True)
display(results)"""))

    # Cell 7: Stratified 5-Fold Cross Validation
    cells.append(create_cell("code", """cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_rows=[]
for name,m in models.items():
    s=cross_val_score(m,X_train,y_train,cv=cv,scoring="f1_weighted",n_jobs=-1)
    cv_rows.append({"Model":name,"Mean CV F1":s.mean(),"CV Std":s.std()})
cv_df=pd.DataFrame(cv_rows).sort_values("Mean CV F1",ascending=False).reset_index(drop=True)
display(cv_df)"""))

    # Cell 8: Confusion Matrix and Classification Report
    cells.append(create_cell("code", """best_name=results.iloc[0]["Model"]; best_model=trained[best_name]
pred=best_model.predict(X_test); labels=sorted(y.unique())
cm=confusion_matrix(y_test,pred,labels=labels)
fig,ax=plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,cmap="Blues",xticks_rotation=30)
ax.set_title(f"Confusion Matrix — {best_name} (Baseline)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
print("Classification Report —", best_name)
print(classification_report(y_test,pred,zero_division=0))"""))

    # Cell 9: Multi-class ROC Curve & Weighted ROC-AUC
    cells.append(create_cell("code", """if hasattr(best_model,"predict_proba"):
    proba=best_model.predict_proba(X_test); ybin=label_binarize(y_test,classes=labels)
    score=roc_auc_score(ybin,proba,multi_class="ovr",average="weighted") if len(labels)>2 else roc_auc_score(ybin,proba[:,1])
    print(f"Weighted Multi-Class ROC-AUC ({best_name}): {score:.4f}")
    plt.figure(figsize=(8,6))
    for i,l in enumerate(labels):
        if ybin[:,i].sum() in (0,len(ybin)): continue
        fpr,tpr,_=roc_curve(ybin[:,i],proba[:,i])
        plt.plot(fpr,tpr,linewidth=2,label=f"{l} (AUC = {auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],"k--",alpha=0.6,label="Chance / Random Guess")
    plt.xlabel("False Positive Rate", fontsize=11); plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"Multi-Class One-vs-Rest ROC Curve — {best_name}", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
else:
    print("ROC-AUC unavailable for this model configuration.")"""))

    # Cell 10: Hyperparameter Tuning via GridSearchCV
    cells.append(create_cell("code", """grids={
"Logistic Regression":{"model__C":[.1,1,10]},
"k-NN":{"model__n_neighbors":[3,5,7,9]},
"Decision Tree":{"model__max_depth":[5,10,15,None]},
"Random Forest":{"model__n_estimators":[100,150],"model__max_depth":[8,12,None]},
"SVM":{"model__C":[.1,1,10]}
}
tuned={}
for name,g in grids.items():
    s=GridSearchCV(models[name],g,cv=cv,scoring="f1_weighted",n_jobs=-1)
    s.fit(X_train,y_train); tuned[name]=s.best_estimator_
    print(f"{name:20s} | Best CV F1: {s.best_score_:.4f} | Optimal Params: {s.best_params_}")"""))

    # Cell 11: Final Tuned Models Comparison & Best Model Selection
    cells.append(create_cell("code", """final=[]
for name,m in tuned.items():
    p=m.predict(X_test)
    final.append({
        "Model":name,
        "Accuracy":accuracy_score(y_test,p),
        "Precision":precision_score(y_test,p,average="weighted",zero_division=0),
        "Recall":recall_score(y_test,p,average="weighted",zero_division=0),
        "F1":f1_score(y_test,p,average="weighted",zero_division=0)
    })
final=pd.DataFrame(final).sort_values("F1",ascending=False).reset_index(drop=True)
display(final)
BEST_MODEL_NAME=final.iloc[0]["Model"]; BEST_MODEL=tuned[BEST_MODEL_NAME]
print(">>> Selected Champion Model:", BEST_MODEL_NAME)"""))

    # Cell 12: Overfitting/Underfitting Diagnostic (Train vs Test F1 Gap)
    cells.append(create_cell("code", """fit=[]
for name,m in tuned.items():
    a=f1_score(y_train,m.predict(X_train),average="weighted",zero_division=0)
    b=f1_score(y_test,m.predict(X_test),average="weighted",zero_division=0)
    gap=a-b
    status="Overfitting Risk" if gap > 0.08 else "Well-Regularized" if gap >= 0 else "Underfitting Risk"
    fit.append({"Model":name,"Train F1":a,"Test F1":b,"Train-Test Gap":gap,"Diagnostic":status})
fit_df=pd.DataFrame(fit).sort_values("Test F1",ascending=False).reset_index(drop=True)
display(fit_df)"""))

    # Cell 13: Country-Level Energy Risk Share Analysis
    cells.append(create_cell("code", """country_risk=(df.groupby("country")["energy_risk"].value_counts(normalize=True).rename("share").reset_index())
display(country_risk.head(30))"""))

    # Cell 14: Interactive Inference Function
    cells.append(create_cell("code", """def predict_energy_risk(oil_import_dependency,oil_price_change,energy_supply_disruption,shipping_disruption,india_energy_exposure,strategic_route_exposure,commodity_price_change):
    row=pd.DataFrame([{
        "oil_import_dependency":oil_import_dependency,
        "oil_price_change":oil_price_change,
        "energy_supply_disruption":energy_supply_disruption,
        "shipping_disruption":shipping_disruption,
        "india_energy_exposure":india_energy_exposure,
        "strategic_route_exposure":strategic_route_exposure,
        "commodity_price_change":commodity_price_change
    }])
    out={"prediction":BEST_MODEL.predict(row)[0]}
    if hasattr(BEST_MODEL,"predict_proba"):
        p=BEST_MODEL.predict_proba(row)[0]
        out["probabilities"]=dict(zip(BEST_MODEL.classes_,np.round(p,4)))
    return out

# Sample test: Severe Persian Gulf / Chokepoint shock
sample_pred = predict_energy_risk(87.5, 22.5, 85.0, 94.0, 18.5, 95.0, 16.0)
print("Interactive Prediction Result:")
print("Predicted Risk Tier:", sample_pred["prediction"])
print("Class Probabilities:", sample_pred.get("probabilities"))"""))

    # Cell 15: Model Serialization with Joblib
    cells.append(create_cell("code", """import joblib
os.makedirs("models",exist_ok=True)
path="models/feature5_india_energy_risk_model.joblib"
joblib.dump(BEST_MODEL,path)
print("Model serialized and saved successfully to:", path)"""))

    # Cell 16: Summary and ML Compliance Architecture
    cells.append(create_cell("markdown", """## Module 6: End-to-End ML Pipeline Architecture Summary

```text
Real Dataset (data/feature5_india_energy_risk.csv)
    ↓
Data Preprocessing (Imputation, Cleaning, Numeric Conversions)
    ↓
Feature Preparation (7 Energy Corridors & Economic Vulnerability Features)
    ↓
Train 5 Syllabus Models (Logistic Regression, k-NN, Decision Tree, Random Forest, SVM)
    ↓
5-Fold Stratified Cross-Validation & GridSearch Hyperparameter Tuning
    ↓
Evaluation & Overfitting Diagnostics (F1, Precision, Recall, Confusion Matrix, Multi-Class ROC-AUC)
    ↓
Select Best Performing Classifier (Saved to models/feature5_india_energy_risk_model.joblib)
    ↓
User enters NEW Geopolitical / Energy Supply Shock Parameters
    ↓
Real-Time Inference via Saved Model Pipeline
    ↓
Predicted Energy Supply Risk Level (Low / Moderate / High / Critical) + Probability Breakdown
```

*All training rows, features, evaluations, and probabilities strictly originate from the verified geopolitical energy corridor dataset.*
"""))

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

    output_path = "Feature_5_India_Energy_Supply_Risk_Intelligence.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {output_path} with {len(cells)} cells.")

    # Also save to notebooks/ directory
    notebooks_path = os.path.join("notebooks", output_path)
    with open(notebooks_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {notebooks_path} with {len(cells)} cells.")

if __name__ == "__main__":
    build()
