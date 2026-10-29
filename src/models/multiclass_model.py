"""
Multiclass Classifier Module for Cyber Threat Detection Framework.
Trains LightGBM on 15 Attack_type categories (DDoS_HTTP, SQL_injection, Ransomware, XSS, etc.).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.abspath("."))

from src.data.loader import load_raw_dataset, split_dataset
from src.data.cleaner import clean_dataframe, convert_numeric_strings
from src.data.preprocessing import build_preprocessing_pipeline, fit_and_transform_pipeline, save_preprocessor


def train_multiclass_model(
    raw_file_path: str = "ML-EdgeIIoT-dataset.csv",
    models_dir: str = "models",
    results_dir: str = "results"
):
    """Train LightGBM multiclass model to identify specific attack types."""
    print("=" * 70)
    print("TRAINING MULTICLASS ATTACK TYPE CLASSIFIER (15 CATEGORIES)")
    print("=" * 70)

    # 1. Load Raw Dataset for Multiclass Target (Attack_type)
    df_raw = load_raw_dataset(raw_file_path)

    # 2. Encode Multiclass Target 'Attack_type'
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df_raw["Attack_type"])

    os.makedirs(models_dir, exist_ok=True)
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    joblib.dump(label_encoder, le_path)
    print(f"[MULTICLASS] Saved LabelEncoder artifact to {le_path}")
    print(f"[MULTICLASS] Classes ({len(label_encoder.classes_)}): {label_encoder.classes_.tolist()}")

    # 3. Clean features using Attack_type as target
    df_temp = df_raw.copy()
    df_temp["Attack_type_encoded"] = y_encoded
    
    X_train_raw, X_test_raw, y_train, y_test = split_dataset(
        df_temp, target_col="Attack_type_encoded", test_size=0.2, random_state=42
    )

    X_train_clean, dropped_cols, _ = clean_dataframe(
        X_train_raw, target_col="Attack_type_encoded", drop_leakage=True
    )
    X_test_clean = convert_numeric_strings(X_test_raw.drop(columns=dropped_cols, errors='ignore'))

    # 4. Preprocess features
    preprocessor, _, _ = build_preprocessing_pipeline(X_train_clean)
    X_train_proc, X_test_proc, fitted_prep, feature_names = fit_and_transform_pipeline(
        preprocessor, X_train_clean, X_test_clean
    )

    # Save multiclass preprocessor if different
    joblib.dump(fitted_prep, os.path.join(models_dir, "multiclass_preprocessing.joblib"))

    # 5. Train Multiclass LightGBM Classifier
    print("\n[MULTICLASS] Training LightGBM Multiclass Model...")
    mc_lgbm = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=15,
        objective="multiclass",
        num_class=len(label_encoder.classes_),
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    mc_lgbm.fit(X_train_proc, y_train)

    # 6. Evaluate
    y_pred = mc_lgbm.predict(X_test_proc)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[MULTICLASS SUMMARY] Test Accuracy across 15 Threat Categories: {acc:.4f} ({acc*100:.2f}%)")

    # 7. Save Artifacts
    mc_model_path = os.path.join(models_dir, "multiclass_lightgbm.joblib")
    joblib.dump(mc_lgbm, mc_model_path)
    print(f"[MULTICLASS] Saved pre-trained Multiclass LightGBM to: [multiclass_lightgbm.joblib](file:///{os.path.abspath(mc_model_path)})")
    print("=" * 70)


if __name__ == "__main__":
    train_multiclass_model()
