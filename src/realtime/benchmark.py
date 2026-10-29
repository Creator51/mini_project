"""
Phase 10: Real-Time Latency & Throughput Benchmarking Module.
Measures preprocessing time, model inference times, end-to-end latency statistics (mean, median, p95, p99),
and batch throughput (events/sec).
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

sys.path.insert(0, os.path.abspath("."))

from src.realtime.inference import RealTimeThreatDetector


def run_latency_and_throughput_benchmark(
    data_dir: str = "data/processed",
    results_dir: str = "results",
    num_single_trials: int = 1000,
    batch_sizes: list = [1, 10, 50, 100, 500, 1000]
):
    """Perform rigorous latency and throughput benchmarks without fabricating data."""
    print("=" * 70)
    print("PHASE 10: REAL-TIME LATENCY AND THROUGHPUT BENCHMARKING")
    print("=" * 70)

    detector = RealTimeThreatDetector(model_type="ensemble")
    X_test_proc = joblib.load(os.path.join(data_dir, "X_test.joblib"))
    
    # 1. Component-level Latency Breakdown (Preprocessing, RF, LGBM, Combination)
    print("\n[STEP 1] Benchmarking Component-Level Latency Breakdown...")
    rf_times = []
    lgbm_times = []
    comb_times = []

    # Warmup
    for _ in range(50):
        detector.rf_model.predict_proba(X_test_proc[:1])
        detector.lgbm_model.predict_proba(X_test_proc[:1])

    for i in range(num_single_trials):
        idx = i % len(X_test_proc)
        sample = X_test_proc[idx:idx+1]

        # RF latency
        t0 = time.perf_counter()
        rf_p = detector.rf_model.predict_proba(sample)
        t1 = time.perf_counter()
        rf_times.append((t1 - t0) * 1000)

        # LightGBM latency
        t0 = time.perf_counter()
        lgbm_p = detector.lgbm_model.predict_proba(sample)
        t1 = time.perf_counter()
        lgbm_times.append((t1 - t0) * 1000)

        # Combination logic
        t0 = time.perf_counter()
        ens_p = (0.50 * rf_p) + (0.50 * lgbm_p)
        t1 = time.perf_counter()
        comb_times.append((t1 - t0) * 1000)

    rf_mean_ms = float(np.mean(rf_times))
    lgbm_mean_ms = float(np.mean(lgbm_times))
    comb_mean_ms = float(np.mean(comb_times))
    
    print(f"  - Random Forest Latency   : {rf_mean_ms:.4f} ms")
    print(f"  - LightGBM Latency        : {lgbm_mean_ms:.4f} ms")
    print(f"  - Combination Logic       : {comb_mean_ms:.4f} ms")

    # 2. End-to-End Latency Percentiles (Mean, Median, P95, P99)
    print("\n[STEP 2] Measuring End-to-End Single-Event Latency Percentiles...")
    e2e_latencies = []
    dummy_payload = {
        "arp.opcode": 0.0, "tcp.ack": 100.0, "tcp.srcport": "80", "http.request.method": "GET"
    }

    # Warmup
    for _ in range(50):
        detector.predict_single(dummy_payload)

    for _ in range(num_single_trials):
        t0 = time.perf_counter()
        detector.predict_single(dummy_payload)
        t1 = time.perf_counter()
        e2e_latencies.append((t1 - t0) * 1000)

    e2e_mean = float(np.mean(e2e_latencies))
    e2e_median = float(np.median(e2e_latencies))
    e2e_p95 = float(np.percentile(e2e_latencies, 95))
    e2e_p99 = float(np.percentile(e2e_latencies, 99))

    print(f"  - End-to-End Mean Latency   : {e2e_mean:.4f} ms")
    print(f"  - End-to-End Median Latency : {e2e_median:.4f} ms")
    print(f"  - End-to-End P95 Latency    : {e2e_p95:.4f} ms")
    print(f"  - End-to-End P99 Latency    : {e2e_p99:.4f} ms")

    # 3. Throughput Benchmarking across Batch Sizes
    print("\n[STEP 3] Benchmarking Throughput (events/sec) across Batch Sizes...")
    throughput_results = []
    
    for b_size in batch_sizes:
        batch_samples = X_test_proc[:b_size]
        
        t0 = time.perf_counter()
        # Vectorized batch prediction
        rf_p = detector.rf_model.predict_proba(batch_samples)
        lgbm_p = detector.lgbm_model.predict_proba(batch_samples)
        ens_p = (0.50 * rf_p) + (0.50 * lgbm_p)
        preds = (ens_p[:, 1] >= 0.5).astype(int)
        t1 = time.perf_counter()

        elapsed_sec = t1 - t0
        throughput = b_size / elapsed_sec if elapsed_sec > 0 else 0.0
        batch_latency_ms = elapsed_sec * 1000
        per_event_latency_ms = (elapsed_sec / b_size) * 1000

        throughput_results.append({
            "batch_size": b_size,
            "total_batch_latency_ms": float(batch_latency_ms),
            "per_event_latency_ms": float(per_event_latency_ms),
            "throughput_events_per_sec": float(throughput)
        })
        print(f"  - Batch Size = {b_size:4d} | Batch Latency: {batch_latency_ms:6.2f} ms | Per-event: {per_event_latency_ms:6.4f} ms | Throughput: {throughput:8.2f} events/sec")

    # 4. Generate Publication Figures
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Component-Level Latency Breakdown
    plt.figure(figsize=(8, 5))
    components = ["Random Forest", "LightGBM", "Combination"]
    comp_latencies = [rf_mean_ms, lgbm_mean_ms, comb_mean_ms]
    
    bars = plt.bar(components, comp_latencies, color=['#2980b9', '#27ae60', '#8e44ad'], edgecolor='black', width=0.5)
    plt.ylabel("Latency (ms)", fontweight='bold')
    plt.title("Component-Level Inference Latency Breakdown", fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.001, f"{h:.4f} ms", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "09_latency_breakdown.png"), dpi=300)
    plt.close()

    # Figure 2: Throughput vs Batch Size
    plt.figure(figsize=(9, 5))
    bs_vals = [r["batch_size"] for r in throughput_results]
    tp_vals = [r["throughput_events_per_sec"] for r in throughput_results]
    
    plt.plot(bs_vals, tp_vals, marker='o', linewidth=2.5, color='#e74c3c', markersize=8)
    plt.xlabel("Batch Size (Number of Network Events)", fontweight='bold')
    plt.ylabel("Throughput (Events / Second)", fontweight='bold')
    plt.title("Real-Time Threat Detection Throughput Scaling", fontweight='bold')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    for x, y in zip(bs_vals, tp_vals):
        plt.annotate(f"{y:,.0f} ev/s", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "10_throughput_vs_batch_size.png"), dpi=300)
    plt.close()

    # 5. Export Benchmark Results JSON
    metrics_dir = os.path.join(results_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    benchmark_path = os.path.join(metrics_dir, "benchmark_results.json")

    benchmark_data = {
        "component_latency_ms": {
            "random_forest_mean": rf_mean_ms,
            "lightgbm_mean": lgbm_mean_ms,
            "combination_logic_mean": comb_mean_ms
        },
        "end_to_end_single_event_latency_ms": {
            "mean": e2e_mean,
            "median": e2e_median,
            "p95": e2e_p95,
            "p99": e2e_p99,
            "trials": num_single_trials
        },
        "throughput_benchmarks": throughput_results
    }

    with open(benchmark_path, "w") as f:
        json.dump(benchmark_data, f, indent=4)

    print(f"\n[STEP 5] Exported Benchmark Metrics to: [benchmark_results.json](file:///{os.path.abspath(benchmark_path)})")
    print("=" * 70)
    print("PHASE 10 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_latency_and_throughput_benchmark()
