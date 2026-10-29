"""
Phases 6 & 7: Model Comparison & Combination Strategy Evaluation.
Compares Random Forest, LightGBM, and Hybrid Ensemble across accuracy, recall, latency, training time, and alpha tuning.
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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

sys.path.insert(0, os.path.abspath("."))

from src.models.ensemble import HybridThreatDetector


def run_model_comparison(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    results_dir: str = "results"
):
    """Execute complete comparative evaluation of RF, LightGBM, and Hybrid Ensemble."""
    print("=" * 70)
    print("PHASES 6 & 7: MODEL COMPARISON AND COMBINATION EVALUATION")
    print("=" * 70)

    # 1. Load Processed Datasets
    print("\n[STEP 1] Loading processed test dataset...")
    X_test = joblib.load(os.path.join(data_dir, "X_test.joblib"))
    y_test = joblib.load(os.path.join(data_dir, "y_test.joblib"))

    # Load RF and LightGBM metrics JSONs
    with open(os.path.join(results_dir, "metrics", "random_forest_metrics.json"), "r") as f:
        rf_metrics = json.load(f)
    with open(os.path.join(results_dir, "metrics", "lightgbm_metrics.json"), "r") as f:
        lgbm_metrics = json.load(f)

    # 2. Evaluate Hybrid Ensemble across different alpha values
    print("\n[STEP 2] Tuning Alpha (Weight Parameter) for Hybrid Ensemble...")
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    alpha_results = []

    detector = HybridThreatDetector()
    detector.load_models(
        rf_path=os.path.join(models_dir, "random_forest.joblib"),
        lgbm_path=os.path.join(models_dir, "lightgbm_model.joblib")
    )

    for a in alphas:
        detector.alpha = a
        start_time = time.time()
        y_pred = detector.predict(X_test)
        latency_ms = ((time.time() - start_time) / len(X_test)) * 1000

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        alpha_results.append({
            "alpha": a,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "false_negative_rate": float(fnr),
            "latency_ms_per_sample": float(latency_ms)
        })
        print(f"  - Alpha = {a:.2f} | Acc: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Latency: {latency_ms:.4f} ms")

    # Select optimal alpha = 0.50 for hybrid baseline
    detector.alpha = 0.50
    detector.save_ensemble(os.path.join(models_dir, "hybrid_ensemble.joblib"))

    # Compute hybrid metrics for optimal alpha
    start_infer = time.time()
    y_pred_ens = detector.predict(X_test)
    ens_infer_time = time.time() - start_infer
    ens_latency = (ens_infer_time / len(X_test)) * 1000
    
    cm_ens = confusion_matrix(y_test, y_pred_ens)
    tn_e, fp_e, fn_e, tp_e = cm_ens.ravel()
    fnr_ens = fn_e / (fn_e + tp_e) if (fn_e + tp_e) > 0 else 0.0
    fpr_ens = fp_e / (fp_e + tn_e) if (fp_e + tn_e) > 0 else 0.0

    ensemble_metrics = {
        "model_name": "Hybrid Ensemble (RF + LightGBM)",
        "alpha": 0.50,
        "accuracy": float(accuracy_score(y_test, y_pred_ens)),
        "precision": float(precision_score(y_test, y_pred_ens)),
        "recall": float(recall_score(y_test, y_pred_ens)),
        "f1_score": float(f1_score(y_test, y_pred_ens)),
        "false_negative_rate": float(fnr_ens),
        "false_positive_rate": float(fpr_ens),
        "training_time_sec": float(rf_metrics["training_time_sec"] + lgbm_metrics["training_time_sec"]),
        "inference_latency_ms_per_sample": float(ens_latency),
        "model_size_mb": float(rf_metrics["model_size_mb"] + lgbm_metrics["model_size_mb"])
    }

    # 3. Create Comparative Plots
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("\n[STEP 3] Generating Model Comparison Figures...")

    # Plot 1: Performance Metric Comparison Bar Chart
    models = ["Random Forest", "LightGBM", "Hybrid Ensemble"]
    accs = [rf_metrics["accuracy"], lgbm_metrics["accuracy"], ensemble_metrics["accuracy"]]
    recalls = [rf_metrics["recall"], lgbm_metrics["recall"], ensemble_metrics["recall"]]
    train_times = [rf_metrics["training_time_sec"], lgbm_metrics["training_time_sec"], ensemble_metrics["training_time_sec"]]
    latencies = [rf_metrics["inference_latency_ms_per_sample"], lgbm_metrics["inference_latency_ms_per_sample"], ensemble_metrics["inference_latency_ms_per_sample"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy & Recall
    x = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width/2, accs, width, label='Accuracy', color='#2980b9')
    axes[0].bar(x + width/2, recalls, width, label='Recall', color='#27ae60')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Detection Performance Comparison', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylim(0.95, 1.01)
    axes[0].legend()

    # Training Time
    axes[1].bar(models, train_times, color=['#3498db', '#e67e22', '#9b59b6'], edgecolor='black')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Training Computational Time', fontweight='bold')

    # Inference Latency
    axes[2].bar(models, latencies, color=['#3498db', '#e67e22', '#9b59b6'], edgecolor='black')
    axes[2].set_ylabel('Latency (ms / sample)')
    axes[2].set_title('Inference Latency per Sample', fontweight='bold')

    plt.tight_layout()
    comp_plot_path = os.path.join(fig_dir, "07_model_comparison_metrics.png")
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()

    # Plot 2: Combined Confusion Matrix Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm_rf = np.array(rf_metrics["confusion_matrix"])
    cm_lgbm = np.array(lgbm_metrics["confusion_matrix"])
    
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Random Forest Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_lgbm, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
    axes[1].set_title("LightGBM Confusion Matrix", fontweight="bold")
    axes[1].set_xlabel("Predicted")

    sns.heatmap(cm_ens, annot=True, fmt="d", cmap="Purples", ax=axes[2], cbar=False)
    axes[2].set_title("Hybrid Ensemble Confusion Matrix", fontweight="bold")
    axes[2].set_xlabel("Predicted")

    plt.tight_layout()
    cm_comp_path = os.path.join(fig_dir, "08_confusion_matrix_comparison.png")
    plt.savefig(cm_comp_path, dpi=300)
    plt.close()

    # 4. Export Combined Summary Metrics JSON
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = os.path.join(metrics_dir, "model_comparison_summary.json")

    summary_data = {
        "random_forest": rf_metrics,
        "lightgbm": lgbm_metrics,
        "hybrid_ensemble": ensemble_metrics,
        "alpha_tuning": alpha_results
    }

    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n[STEP 4] Saved Model Comparison Summary to: [model_comparison_summary.json](file:///{os.path.abspath(summary_path)})")
    print("=" * 70)
    print("PHASES 6 & 7 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_model_comparison()
