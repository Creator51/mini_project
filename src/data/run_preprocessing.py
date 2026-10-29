"""
Phase 2 Runner Script: Data Cleaning & Preprocessing Execution.
Performs leak-free stratified splitting, feature cleaning with documented justifications,
fits ColumnTransformer strictly on X_train, transforms X_test, and exports processed datasets.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure src path is in Python path
sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_raw_dataset, split_dataset
from src.data.cleaner import clean_dataframe, convert_numeric_strings
from src.data.preprocessing import (
    build_preprocessing_pipeline,
    fit_and_transform_pipeline,
    save_preprocessor
)


def run_phase_2(
    raw_file_path: str = "ML-EdgeIIoT-dataset.csv",
    target_col: str = "Attack_label",
    test_size: float = 0.2,
    random_state: int = 42
):
    """Execute complete Phase 2 Data Cleaning & Preprocessing pipeline."""
    print("=" * 70)
    print("PHASE 2: DATA CLEANING AND PREPROCESSING PIPELINE")
    print("=" * 70)

    # 1. Load Raw Dataset
    df_raw = load_raw_dataset(raw_file_path)

    # 2. Stratified Train-Test Split BEFORE any feature transformation (Zero Leakage Rule)
    X_train_raw, X_test_raw, y_train, y_test = split_dataset(
        df_raw, target_col=target_col, test_size=test_size, random_state=random_state
    )

    # 3. Clean Features (Drop Constant & Leakage columns)
    print("\n[STEP 1] Cleaning Training Features & Documenting Removals...")
    X_train_clean, dropped_cols, removal_reasons = clean_dataframe(
        X_train_raw, target_col=target_col, drop_leakage=True
    )
    
    # Apply identical feature drop schema & numeric conversion to X_test (without re-fitting rule logic on test)
    X_test_clean = convert_numeric_strings(X_test_raw.drop(columns=dropped_cols, errors='ignore'))

    # Save Feature Removal Justifications Report
    os.makedirs("results/experiments", exist_ok=True)
    justification_path = "results/experiments/feature_removal_justifications.json"
    with open(justification_path, "w") as f:
        json.dump({
            "target_column": target_col,
            "total_raw_features": X_train_raw.shape[1],
            "total_cleaned_features": X_train_clean.shape[1],
            "dropped_count": len(dropped_cols),
            "removed_features_and_justifications": removal_reasons
        }, f, indent=4)
    print(f"[STEP 1] Saved removal justifications to: [feature_removal_justifications.json](file:///{os.path.abspath(justification_path)})")

    # 4. Build Preprocessing Pipeline (ColumnTransformer)
    print("\n[STEP 2] Building sklearn ColumnTransformer Pipeline...")
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(X_train_clean)

    # 5. Fit strictly on X_train_clean and transform X_train_clean & X_test_clean
    print("\n[STEP 3] Fitting Preprocessor strictly on X_train (Zero Data Leakage)...")
    X_train_proc, X_test_proc, fitted_preprocessor, feature_names = fit_and_transform_pipeline(
        preprocessor, X_train_clean, X_test_clean
    )

    # 6. Save Processed Artifacts to disk
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n[STEP 4] Saving Processed Data & Fitted Pipeline...")
    save_preprocessor(fitted_preprocessor, "models/preprocessing.joblib")

    joblib.dump(X_train_proc, "data/processed/X_train.joblib")
    joblib.dump(X_test_proc, "data/processed/X_test.joblib")
    joblib.dump(y_train.values, "data/processed/y_train.joblib")
    joblib.dump(y_test.values, "data/processed/y_test.joblib")
    
    with open("data/processed/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)

    print(f"  - X_train processed shape : {X_train_proc.shape}")
    print(f"  - X_test processed shape  : {X_test_proc.shape}")
    print(f"  - y_train shape           : {y_train.shape}")
    print(f"  - y_test shape            : {y_test.shape}")
    print(f"  - Processed Feature Names : {len(feature_names)} features saved.")

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETED SUCCESSFULLY! PREPROCESSING PIPELINE READY.")
    print("=" * 70)


if __name__ == "__main__":
    run_phase_2()
