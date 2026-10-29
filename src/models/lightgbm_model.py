"""
Phase 5: LightGBM Classifier Module for Cyber Threat Detection Framework.
Trains LightGBM GBDT model, measures training/inference latency, saves artifacts, and generates publication plots.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, os.path.abspath("."))


def train_and_evaluate_lgbm(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    results_dir: str = "results",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 15,
    random_state: int = 42
):
    """Train LightGBM classifier and record comprehensive performance metrics."""
    print("=" * 70)
    print("PHASE 5: LIGHTGBM CLASSIFIER TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Load Processed Datasets
    print("\n[STEP 1] Loading preprocessed datasets...")
    X_train = joblib.load(os.path.join(data_dir, "X_train.joblib"))
    X_test = joblib.load(os.path.join(data_dir, "X_test.joblib"))
    y_train = joblib.load(os.path.join(data_dir, "y_train.joblib"))
    y_test = joblib.load(os.path.join(data_dir, "y_test.joblib"))

    with open(os.path.join(data_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    print(f"  - X_train shape: {X_train.shape}")
    print(f"  - X_test shape : {X_test.shape}")

    # 2. Initialize and Train LightGBM Model
    lgbm = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced',
        verbose=-1
    )

    print(f"\n[STEP 2] Training LightGBM (n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth})...")
    start_train_time = time.time()
    lgbm.fit(X_train, y_train)
    training_time = time.time() - start_train_time
    print(f"  - Training Time: {training_time:.2f} seconds")

    # 3. Evaluate Model & Measure Latency
    print("\n[STEP 3] Evaluating LightGBM on Test Set...")
    start_infer_time = time.time()
    y_pred = lgbm.predict(X_test)
    y_prob = lgbm.predict_proba(X_test)[:, 1]
    total_infer_time = time.time() - start_infer_time
    
    per_sample_latency_ms = (total_infer_time / len(X_test)) * 1000

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("\n--- LIGHTGBM PERFORMANCE METRICS ---")
    print(f"  Accuracy               : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision              : {prec:.4f}")
    print(f"  Recall (Detection Rate): {rec:.4f}")
    print(f"  F1-Score               : {f1:.4f}")
    print(f"  False Negative Rate    : {fnr:.4f} ({fnr*100:.2f}%) [CRITICAL FOR SECURITY]")
    print(f"  False Positive Rate    : {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  Training Time          : {training_time:.2f} sec")
    print(f"  Inference Latency      : {per_sample_latency_ms:.4f} ms/sample")

    # 4. Save Model Artifact
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lightgbm_model.joblib")
    joblib.dump(lgbm, model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n[STEP 4] Saved trained LightGBM model to: [lightgbm_model.joblib](file:///{os.path.abspath(model_path)}) ({model_size_mb:.2f} MB)")

    # 5. Save Confusion Matrix Plot
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cm_path = os.path.join(fig_dir, "05_lgbm_confusion_matrix.png")

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title("LightGBM Confusion Matrix", fontweight="bold", fontsize=14)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 6. Save Feature Importance Plot
    fi_path = os.path.join(fig_dir, "06_lgbm_feature_importance.png")
    importances = lgbm.feature_importances_
    top_indices = np.argsort(importances)[-15:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_indices)), importances[top_indices], color="#27ae60", edgecolor="black")
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.xlabel("Feature Importance (Split Count)", fontweight="bold")
    plt.title("LightGBM Top 15 Feature Importances", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fi_path, dpi=300)
    plt.close()

    # 7. Export Metrics JSON
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "lightgbm_metrics.json")

    metrics_data = {
        "model_name": "LightGBM",
        "parameters": {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "random_state": random_state},
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "false_negative_rate": float(fnr),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
        "training_time_sec": float(training_time),
        "inference_latency_ms_per_sample": float(per_sample_latency_ms),
        "model_size_mb": float(model_size_mb)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"[STEP 5] Exported LightGBM metrics to: [lightgbm_metrics.json](file:///{os.path.abspath(metrics_path)})")
    print("=" * 70)
    print("PHASE 5 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate_lgbm()
