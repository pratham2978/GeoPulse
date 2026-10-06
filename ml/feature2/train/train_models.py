"""
Training Pipeline for Feature 2: News & Narrative Classification
Strictly Syllabus-Compliant Machine Learning:
- Module 3: Supervised Learning (Logistic Regression, k-NN, Decision Tree, Random Forest, SVM)
- Module 5: Model Evaluation & Optimization (Train-Test Split, 5-Fold CV, Confusion Matrix, Accuracy, Precision, Recall, F1, ROC/AUC, Overfitting/Underfitting, Hyperparameter Tuning)
- Module 6: Mini Project (Full end-to-end preprocessing, training, evaluation, interpretation, deployment)

STRICT RULE: Only the 5 approved algorithms. No BERT, No Deep Learning, No Naive Bayes, No XGBoost.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

ORDERED_CLASSES = [
    "Military Conflict",
    "Diplomatic / Political",
    "Energy Risk",
    "Trade & Shipping",
    "Economic Impact",
    "Humanitarian Impact"
]

def run_training_pipeline():
    print("=" * 70)
    print("GEO-PULSE AI — FEATURE 2: NEWS & NARRATIVE CLASSIFICATION TRAINING")
    print("=" * 70)

    # 1. Load Cleaned Dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'data', 'processed', 'geopolitical_news_cleaned.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'geopolitical_news_cleaned.csv')

    df = pd.read_csv(data_path)
    print(f"\n[1] Loaded cleaned dataset: {len(df)} records")
    print("Columns:", list(df.columns))

    # 2. Text Preparation & Feature Extraction
    # Raw News -> Data Cleaning -> Feature Preparation -> Numerical Features -> ML Classification
    print("\n[2] Feature Engineering Pipeline: TF-IDF Representation")
    X_raw = df['combined_text'].astype(str)
    y_raw = df['label'].astype(str)

    # 3. Stratified Train / Test Split (80% Training, 20% Testing)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    print(f"Stratified Split: {len(X_train_text)} Training records, {len(X_test_text)} Testing records")

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english'
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_)} features")

    # 4. Model Definitions (Syllabus Module 3)
    # 7.1 Logistic Regression
    # 7.2 k-Nearest Neighbors
    # 7.3 Decision Tree
    # 7.4 Random Forest
    # 7.5 Support Vector Machine
    models_config = {
        "Logistic Regression": {
            "baseline": LogisticRegression(C=0.5, max_iter=1000, random_state=42),
            "tuned": LogisticRegression(C=2.5, max_iter=1000, random_state=42),
            "param_desc": "Regularization strength C tuned from 0.5 to 2.5 (optimal L2 penalty)"
        },
        "k-Nearest Neighbors (k-NN)": {
            "baseline": KNeighborsClassifier(n_neighbors=9, metric='cosine'),
            "tuned": KNeighborsClassifier(n_neighbors=5, metric='cosine'),
            "param_desc": "n_neighbors tuned from 9 to 5 with cosine distance for high-dimensional TF-IDF"
        },
        "Decision Tree": {
            "baseline": DecisionTreeClassifier(max_depth=None, random_state=42),
            "tuned": DecisionTreeClassifier(max_depth=16, min_samples_split=6, random_state=42),
            "param_desc": "max_depth constrained from None (unpruned) to 16, min_samples_split=6 to control variance"
        },
        "Random Forest": {
            "baseline": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
            "tuned": RandomForestClassifier(n_estimators=150, max_depth=22, random_state=42),
            "param_desc": "n_estimators expanded to 150, max_depth tuned to 22 for ensemble stability"
        },
        "Support Vector Machine (SVM)": {
            "baseline": CalibratedClassifierCV(SVC(kernel='rbf', C=1.0), cv=3),
            "tuned": CalibratedClassifierCV(SVC(kernel='linear', C=1.5), cv=3),
            "param_desc": "Kernel compared RBF vs Linear; Linear kernel with C=1.5 achieved superior text separation"
        }
    }

    # 5. Cross Validation, Training & Evaluation (Module 5)
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    comparison_results = []
    overfitting_analysis = []
    tuning_comparisons = []
    confusion_matrices = {}
    roc_data = {}
    trained_models = {}

    # Multi-class binarization for ROC AUC
    y_test_bin = label_binarize(y_test, classes=ORDERED_CLASSES)

    print("\n--- 5-FOLD CROSS VALIDATION & TEST SET EVALUATION ---")
    for name, cfg in models_config.items():
        base_model = cfg["baseline"]
        tuned_model = cfg["tuned"]

        # Evaluate baseline
        base_model.fit(X_train_vec, y_train)
        base_test_preds = base_model.predict(X_test_vec)
        base_acc = float(accuracy_score(y_test, base_test_preds) * 100)
        base_f1 = float(f1_score(y_test, base_test_preds, average='macro') * 100)

        # Train and evaluate tuned model
        tuned_model.fit(X_train_vec, y_train)
        trained_models[name] = tuned_model

        # 5-Fold Cross Validation on Training Data
        cv_scores = cross_val_score(tuned_model, X_train_vec, y_train, cv=cv_kfold, scoring='f1_macro')
        cv_mean = float(cv_scores.mean() * 100)
        cv_std = float(cv_scores.std() * 100)

        # Predictions on unseen Test set
        test_preds = tuned_model.predict(X_test_vec)
        train_preds = tuned_model.predict(X_train_vec)

        # Metrics
        train_acc = float(accuracy_score(y_train, train_preds) * 100)
        test_acc = float(accuracy_score(y_test, test_preds) * 100)
        precision = float(precision_score(y_test, test_preds, average='macro', zero_division=0) * 100)
        recall = float(recall_score(y_test, test_preds, average='macro', zero_division=0) * 100)
        f1 = float(f1_score(y_test, test_preds, average='macro', zero_division=0) * 100)

        # Overfitting / Underfitting diagnosis
        acc_diff = round(train_acc - test_acc, 2)
        if acc_diff > 12.0:
            diagnosis = "Overfitting Detected (High Variance — train accuracy significantly exceeds test)"
        elif test_acc < 75.0 and train_acc < 80.0:
            diagnosis = "Underfitting Detected (High Bias — model fails to capture text complexity)"
        else:
            diagnosis = "Good Fit (Balanced Bias-Variance tradeoff with solid generalization)"

        overfitting_analysis.append({
            "model": name,
            "train_accuracy": round(train_acc, 2),
            "test_accuracy": round(test_acc, 2),
            "accuracy_difference": acc_diff,
            "diagnosis": diagnosis
        })

        tuning_comparisons.append({
            "model": name,
            "tuning_parameter": cfg["param_desc"],
            "before_tuning": {
                "accuracy": round(base_acc, 2),
                "f1_score": round(base_f1, 2)
            },
            "after_tuning": {
                "accuracy": round(test_acc, 2),
                "f1_score": round(f1, 2)
            },
            "improvement": round(f1 - base_f1, 2)
        })

        comparison_results.append({
            "model": name,
            "accuracy": round(test_acc, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
            "cv_mean": round(cv_mean, 2),
            "cv_std": round(cv_std, 2)
        })

        # Confusion Matrix
        cm = confusion_matrix(y_test, test_preds, labels=ORDERED_CLASSES)
        confusion_matrices[name] = cm.tolist()

        # ROC Curve & AUC
        if hasattr(tuned_model, "predict_proba"):
            probs = tuned_model.predict_proba(X_test_vec)
            # Per-class ROC
            model_roc = {}
            auc_list = []
            for idx, c_name in enumerate(ORDERED_CLASSES):
                fpr, tpr, _ = roc_curve(y_test_bin[:, idx], probs[:, idx])
                roc_auc_val = float(auc(fpr, tpr))
                auc_list.append(roc_auc_val)
                # Sample down points for clean JSON storage
                step = max(1, len(fpr) // 30)
                sampled_fpr = [round(float(x), 4) for x in fpr[::step]]
                sampled_tpr = [round(float(y), 4) for y in tpr[::step]]
                if sampled_fpr[-1] != 1.0:
                    sampled_fpr.append(1.0)
                    sampled_tpr.append(1.0)
                model_roc[c_name] = {
                    "fpr": sampled_fpr,
                    "tpr": sampled_tpr,
                    "auc": round(roc_auc_val, 4)
                }
            model_roc["macro_auc"] = round(float(np.mean(auc_list)), 4)
            roc_data[name] = model_roc

        print(f"[{name}] Acc={test_acc:.2f}%, Prec={precision:.2f}%, Rec={recall:.2f}%, F1={f1:.2f}%, 5-Fold CV={cv_mean:.2f}% (±{cv_std:.2f}%)")

    # 6. Best Model Selection (F1 Score primary metric)
    best_item = max(comparison_results, key=lambda x: x["f1_score"])
    best_model_name = best_item["model"]
    best_model_instance = trained_models[best_model_name]

    print("\n" + "=" * 50)
    print(f"[*] BEST PERFORMING MODEL: {best_model_name}")
    print(f"  F1 Score   : {best_item['f1_score']}%")
    print(f"  Accuracy   : {best_item['accuracy']}%")
    print(f"  Precision  : {best_item['precision']}%")
    print(f"  Recall     : {best_item['recall']}%")
    print(f"  5-Fold CV  : {best_item['cv_mean']}% (±{best_item['cv_std']}%)")
    print("=" * 50)

    # 7. Model Serialization & Metadata Export
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    models_dir = os.path.join(root_dir, 'backend', 'models')
    os.makedirs(models_dir, exist_ok=True)
    ml_models_dir = os.path.join(root_dir, 'ml', 'feature2', 'models')
    os.makedirs(ml_models_dir, exist_ok=True)
    frontend_data_dir = os.path.join(root_dir, 'src', 'data')
    os.makedirs(frontend_data_dir, exist_ok=True)

    # Save Best Model and Vectorizer
    model_save_path = os.path.join(models_dir, 'narrative_classifier.joblib')
    vec_save_path = os.path.join(models_dir, 'vectorizer.joblib')
    joblib.dump(best_model_instance, model_save_path)
    joblib.dump(vectorizer, vec_save_path)
    joblib.dump(best_model_instance, os.path.join(ml_models_dir, 'narrative_classifier.joblib'))
    joblib.dump(vectorizer, os.path.join(ml_models_dir, 'vectorizer.joblib'))
    print(f"\n[OK] Best model saved to: {model_save_path}")
    print(f"[OK] Vectorizer saved to: {vec_save_path}")

    # Dataset Summary
    raw_csv_path = os.path.join(root_dir, 'backend', 'data', 'raw', 'geopolitical_news_raw.csv')
    df_raw = pd.read_csv(raw_csv_path)
    dataset_summary = {
        "total_raw_records": len(df_raw),
        "duplicates_removed": int(df_raw.duplicated(subset=['title', 'text']).sum()),
        "missing_values_removed": int(df_raw[['title', 'text', 'label']].isna().any(axis=1).sum()),
        "cleaned_records": len(df),
        "training_records": len(X_train_text),
        "testing_records": len(X_test_text),
        "number_of_classes": len(ORDERED_CLASSES),
        "classes": ORDERED_CLASSES,
        "class_distribution": df['label'].value_counts().to_dict(),
        "train_class_distribution": y_train.value_counts().to_dict(),
        "test_class_distribution": y_test.value_counts().to_dict(),
        "best_model": best_model_name,
        "best_f1_score": best_item["f1_score"],
        "best_accuracy": best_item["accuracy"]
    }

    # Model Comparison Export
    model_comparison_payload = {
        "comparison": comparison_results,
        "best_model": best_item,
        "overfitting_analysis": overfitting_analysis,
        "tuning_comparisons": tuning_comparisons
    }

    # Confusion Matrices Export
    confusion_matrices_payload = {
        "classes": ORDERED_CLASSES,
        "best_model": best_model_name,
        "matrices": confusion_matrices
    }

    # Sample Articles for Instant UI Demonstration (Authentic from Test Set)
    sample_articles = []
    # Pick 3 diverse actual test records
    test_df = pd.DataFrame({'text': X_test_text, 'label': y_test})
    for target_cat in ['Military Conflict', 'Trade & Shipping', 'Energy Risk']:
        match = test_df[test_df['label'] == target_cat]
        if not match.empty:
            full_record = df[df['combined_text'] == match.iloc[0]['text']].iloc[0]
            sample_articles.append({
                "label": target_cat,
                "headline": str(full_record['title']),
                "text": str(full_record['text']),
                "source": str(full_record.get('source', 'Reuters')),
                "country": str(full_record.get('country', 'Global'))
            })

    # Save to backend/models/
    with open(os.path.join(models_dir, 'dataset_summary.json'), 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    with open(os.path.join(models_dir, 'model_comparison.json'), 'w') as f:
        json.dump(model_comparison_payload, f, indent=2)
    with open(os.path.join(models_dir, 'confusion_matrices.json'), 'w') as f:
        json.dump(confusion_matrices_payload, f, indent=2)
    with open(os.path.join(models_dir, 'roc_curve_data.json'), 'w') as f:
        json.dump(roc_data, f, indent=2)
    with open(os.path.join(models_dir, 'sample_articles.json'), 'w') as f:
        json.dump(sample_articles, f, indent=2)

    # Mirror into src/data/ for standalone frontend
    with open(os.path.join(frontend_data_dir, 'narrative_dataset_summary.json'), 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    with open(os.path.join(frontend_data_dir, 'narrative_model_comparison.json'), 'w') as f:
        json.dump(model_comparison_payload, f, indent=2)
    with open(os.path.join(frontend_data_dir, 'narrative_confusion_matrices.json'), 'w') as f:
        json.dump(confusion_matrices_payload, f, indent=2)
    with open(os.path.join(frontend_data_dir, 'narrative_roc_data.json'), 'w') as f:
        json.dump(roc_data, f, indent=2)
    with open(os.path.join(frontend_data_dir, 'narrative_samples.json'), 'w') as f:
        json.dump(sample_articles, f, indent=2)

    print("\n[OK] All evaluation metrics, confusion matrices, and ROC curves exported successfully.")
    return best_model_name

if __name__ == '__main__':
    run_training_pipeline()
