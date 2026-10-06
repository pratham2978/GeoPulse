"""
Evaluation Metrics Module for Feature 2: News & Narrative Classification
Strictly Syllabus-Compliant Evaluation:
- Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 Score (Macro/Weighted)
- 5-Fold Cross Validation
- Multiclass Confusion Matrix
- Multiclass ROC & AUC
- Overfitting / Underfitting Diagnostics
"""

import os
import json
import pandas as pd
import numpy as np

def load_evaluation_metrics():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    models_dir = os.path.join(root_dir, 'backend', 'models')

    comparison_path = os.path.join(models_dir, 'model_comparison.json')
    cm_path = os.path.join(models_dir, 'confusion_matrices.json')
    roc_path = os.path.join(models_dir, 'roc_curve_data.json')
    summary_path = os.path.join(models_dir, 'dataset_summary.json')

    metrics = {}
    if os.path.exists(comparison_path):
        with open(comparison_path, 'r') as f:
            metrics['comparison'] = json.load(f)
    if os.path.exists(cm_path):
        with open(cm_path, 'r') as f:
            metrics['confusion_matrices'] = json.load(f)
    if os.path.exists(roc_path):
        with open(roc_path, 'r') as f:
            metrics['roc_curves'] = json.load(f)
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            metrics['dataset_summary'] = json.load(f)

    return metrics

if __name__ == '__main__':
    data = load_evaluation_metrics()
    print("Loaded evaluation metrics successfully:")
    if 'comparison' in data:
        print("Best Model:", data['comparison'].get('best_model', {}))
        print("Models evaluated:", len(data['comparison'].get('comparison', [])))
