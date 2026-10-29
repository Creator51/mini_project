"""
Phase 11: Feature Selection Impact Experiment.
Evaluates model accuracy, recall, model size, and inference speed across Top-K feature subsets.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath("."))


def run_feature_selection_experiment(
    data_dir: str = "data/processed",
    results_dir: str = "results",
    k_values: list = [5, 10, 15, 20, 30, 41]
):
    """Evaluate performance vs computational complexity across Top-K feature subsets."""
    print("=" * 70)
    print("PHASE 11: FEATURE SELECTION IMPACT EXPERIMENT")
    print("=" * 70)

    X_train = joblib.load(os.path.join(data_dir, "X_train.joblib"))
    X_test = joblib.load(os.path.join(data_dir, "X_test.joblib"))
    y_train = joblib.load(os.path.join(data_dir, "y_train.joblib"))
    y_test = joblib.load(os.path.join(data_dir, "y_test.joblib"))

    with open(os.path.join(data_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    # 1. Obtain feature importance rankings from baseline LightGBM
    baseline_lgbm = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    baseline_lgbm.fit(X_train, y_train)
    importances = baseline_lgbm.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    results = []
    print(f"\n[STEP 1] Testing Feature Subsets (K features)...")

    for k in k_values:
        selected_idx = sorted_indices[:k]
        X_train_k = X_train[:, selected_idx]
        X_test_k = X_test[:, selected_idx]

        # Train model on K features
        t0_train = time.time()
        model_k = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        model_k.fit(X_train_k, y_train)
        train_time = time.time() - t0_train

        # Measure inference latency
        t0_infer = time.time()
        y_pred = model_k.predict(X_test_k)
        infer_time_ms = ((time.time() - t0_infer) / len(X_test_k)) * 1000

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "k_features": k,
            "accuracy": float(acc),
            "recall": float(rec),
            "f1_score": float(f1),
            "training_time_sec": float(train_time),
            "inference_latency_ms_per_sample": float(infer_time_ms),
            "top_features": [feature_names[i] for i in selected_idx[:5]]
        })
        print(f"  - K = {k:2d} features | Acc: {acc:.4f} | Recall: {rec:.4f} | Train Time: {train_time:.2f}s | Latency: {infer_time_ms:.4f} ms")

    # 2. Save Plot
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ks = [r["k_features"] for r in results]
    accs = [r["accuracy"] for r in results]
    lats = [r["inference_latency_ms_per_sample"] for r in results]

    ax1.set_xlabel("Number of Selected Features (K)", fontweight='bold')
    ax1.set_ylabel("Accuracy / Recall", color='#27ae60', fontweight='bold')
    ax1.plot(ks, accs, marker='o', color='#27ae60', linewidth=2.5, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='#27ae60')
    ax1.set_ylim(0.95, 1.01)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Inference Latency (ms / sample)", color='#e74c3c', fontweight='bold')
    ax2.plot(ks, lats, marker='s', color='#e74c3c', linewidth=2.5, linestyle='--', label='Latency')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')

    plt.title("Feature Selection Impact: Accuracy vs Inference Latency Tradeoff", fontweight='bold')
    fig.tight_layout()
    plot_path = os.path.join(fig_dir, "11_feature_selection_impact.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # 3. Export JSON
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    exp_path = os.path.join(metrics_dir, "feature_selection_experiment.json")

    with open(exp_path, "w") as f:
        json.dump({"feature_selection_results": results}, f, indent=4)

    print(f"\n[STEP 2] Saved Feature Selection Experiment to: [feature_selection_experiment.json](file:///{os.path.abspath(exp_path)})")
    print("=" * 70)
    print("PHASE 11 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_feature_selection_experiment()
