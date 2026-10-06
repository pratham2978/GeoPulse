"""
Data Preprocessing for Feature 2: News & Narrative Classification
Syllabus Modules: Module 3, Module 5, Module 6

Performs:
1. Dataset Inspection: record count, columns, types, missing values, duplicates, class distribution
2. Missing Value Handling: identifies and removes unclassifiable records (missing title, text, or label)
3. Duplicate Detection & Removal: removes exact title+text duplicates
4. Text Preparation: concatenates Title + Text, normalizes punctuation, case folding
5. Exports clean dataset to processed CSV
"""

import os
import re
import json
import pandas as pd
import numpy as np

def clean_text(text: str) -> str:
    """Standard undergraduate NLP text normalization without black-box embeddings."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    t = text.lower()
    # Normalize punctuation/special characters
    t = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', t)
    # Collapse multiple whitespaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def preprocess_news_dataset():
    raw_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'data', 'raw', 'geopolitical_news_raw.csv')
    if not os.path.exists(raw_path):
        raw_path = os.path.join(os.path.dirname(__file__), 'data', 'geopolitical_news_raw.csv')

    print("=" * 60)
    print("SECTION 4: DATA PREPROCESSING — NEWS & NARRATIVE CLASSIFICATION")
    print("=" * 60)

    # 4.1 Dataset Inspection
    df_raw = pd.read_csv(raw_path)
    total_raw_records = len(df_raw)
    total_columns = len(df_raw.columns)
    
    print("\n--- 4.1 DATASET INSPECTION ---")
    print(f"Total Raw Records: {total_raw_records}")
    print(f"Total Columns: {total_columns}")
    print("Columns:", list(df_raw.columns))
    print("Data Types:\n", df_raw.dtypes)
    
    null_counts = df_raw.isnull().sum()
    print("\nMissing Values per Column:\n", null_counts)
    
    raw_duplicates = df_raw.duplicated(subset=['title', 'text']).sum()
    print(f"\nRaw Duplicate Records (Title + Text): {raw_duplicates}")
    print("\nClass Distribution (Raw):\n", df_raw['label'].value_counts(dropna=False))

    # 4.2 Missing Values Handling
    print("\n--- 4.2 MISSING VALUES HANDLING ---")
    # Check for missing title, text, or target label
    invalid_mask = df_raw['title'].isna() | df_raw['text'].isna() | df_raw['label'].isna()
    missing_count = int(invalid_mask.sum())
    print(f"Records with missing title, text, or label: {missing_count}")
    
    df_cleaned = df_raw[~invalid_mask].copy()
    print(f"Records remaining after missing value removal: {len(df_cleaned)}")

    # 4.3 Duplicate Removal
    print("\n--- 4.3 DUPLICATE REMOVAL ---")
    records_before_dedup = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates(subset=['title', 'text']).copy()
    records_after_dedup = len(df_cleaned)
    duplicates_removed = records_before_dedup - records_after_dedup

    print(f"Records Before Cleaning : {total_raw_records}")
    print(f"Records After Cleaning  : {records_after_dedup}")
    print(f"Duplicates Removed      : {duplicates_removed}")
    print(f"Missing Values Removed  : {missing_count}")

    # 4.4 Text Preparation
    print("\n--- 4.4 TEXT PREPARATION ---")
    # Meaningful text fields: Title + Article Text
    df_cleaned['clean_title'] = df_cleaned['title'].apply(clean_text)
    df_cleaned['clean_text'] = df_cleaned['text'].apply(clean_text)
    df_cleaned['combined_text'] = df_cleaned['clean_title'] + " " + df_cleaned['clean_text']

    print(f"Prepared clean combined_text for {len(df_cleaned)} records.")
    print("Sample prepared text snippet:")
    print("  ", df_cleaned['combined_text'].iloc[0][:160], "...")

    # Class distribution after cleaning
    class_dist = df_cleaned['label'].value_counts().to_dict()
    print("\nFinal Balanced Class Distribution:\n", class_dist)

    # Export clean dataset
    processed_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    out_csv = os.path.join(processed_dir, 'geopolitical_news_cleaned.csv')
    df_cleaned.to_csv(out_csv, index=False)
    print(f"\n[OK] Cleaned dataset saved to: {out_csv}")

    ml_proc_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(ml_proc_dir, exist_ok=True)
    df_cleaned.to_csv(os.path.join(ml_proc_dir, 'geopolitical_news_cleaned.csv'), index=False)

    # Return inspection dictionary
    return {
        'total_raw_records': total_raw_records,
        'records_after_cleaning': records_after_dedup,
        'duplicates_removed': duplicates_removed,
        'missing_values_removed': missing_count,
        'class_distribution': class_dist
    }

if __name__ == '__main__':
    preprocess_news_dataset()
