"""
Phase 3: Exploratory Data Analysis (EDA) Module.
Generates publication-grade figures for class distributions, correlation heatmaps, and feature distributions.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.cleaner import clean_dataframe

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})


def plot_class_distributions(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot publication-quality class distributions for Attack_label and Attack_type."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Binary Target (Attack_label)
    if "Attack_label" in df.columns:
        counts = df["Attack_label"].value_counts()
        labels = ["Attack (1)" if c == 1 else "Normal (0)" for c in counts.index]
        colors = ["#e74c3c", "#2ecc71"]
        
        bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", alpha=0.85)
        axes[0].set_title("Binary Class Distribution (Attack vs Normal)", fontweight="bold")
        axes[0].set_ylabel("Count")
        axes[0].set_ylim(0, max(counts.values) * 1.15)
        
        for bar in bars:
            height = bar.get_height()
            pct = (height / len(df)) * 100
            axes[0].annotate(f"{height:,}\n({pct:.1f}%)",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

    # Multiclass Target (Attack_type)
    if "Attack_type" in df.columns:
        type_counts = df["Attack_type"].value_counts()
        palette = sns.color_palette("viridis", len(type_counts))
        
        bars = axes[1].barh(type_counts.index[::-1], type_counts.values[::-1], color=palette, edgecolor="black", alpha=0.85)
        axes[1].set_title("Multiclass Threat Distribution (Edge-IIoTset)", fontweight="bold")
        axes[1].set_xlabel("Count")
        
        for bar in bars:
            width = bar.get_width()
            pct = (width / len(df)) * 100
            axes[1].annotate(f"{width:,} ({pct:.1f}%)",
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "01_class_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[EDA] Saved Class Distribution plot to: [01_class_distribution.png](file:///{os.path.abspath(plot_path)})")


def plot_correlation_heatmap(df_clean: pd.DataFrame, output_dir: str = "results/figures", top_n: int = 15):
    """Plot correlation heatmap of top numerical features."""
    os.makedirs(output_dir, exist_ok=True)

    num_df = df_clean.select_dtypes(include=[np.number])
    if num_df.empty:
        print("[EDA] No numerical columns available for correlation heatmap.")
        return

    # Calculate variance and filter constant/near-constant columns
    var = num_df.var()
    non_zero_var_cols = var[var > 1e-5].index.tolist()
    num_df = num_df[non_zero_var_cols]

    # Select top features with highest correlation
    corr_matrix = num_df.corr().abs()
    
    if len(corr_matrix) > top_n:
        top_cols = corr_matrix.sum().nlargest(top_n).index
        corr_matrix = num_df[top_cols].corr()
    else:
        corr_matrix = num_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                square=True, linewidths=0.5, annot_kws={"size": 8})
    plt.title(f"Top {len(corr_matrix)} Feature Correlation Matrix (Training Set)", fontweight="bold")
    
    plot_path = os.path.join(output_dir, "02_correlation_heatmap.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[EDA] Saved Correlation Heatmap to: [02_correlation_heatmap.png](file:///{os.path.abspath(plot_path)})")


def run_eda(raw_file_path: str = "ML-EdgeIIoT-dataset.csv"):
    """Execute complete Phase 3 EDA."""
    print("=" * 70)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    df_raw = pd.read_csv(raw_file_path, low_memory=False)
    
    print("\n[STEP 1] Generating Class Distribution Plot...")
    plot_class_distributions(df_raw)

    print("\n[STEP 2] Generating Correlation Heatmap...")
    df_clean, _, _ = clean_dataframe(df_raw, target_col="Attack_label")
    plot_correlation_heatmap(df_clean)

    # Save summary statistics
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    eda_summary = {
        "total_records": len(df_raw),
        "total_raw_features": df_raw.shape[1],
        "total_cleaned_features": df_clean.shape[1],
        "attack_label_distribution": df_raw["Attack_label"].value_counts().to_dict(),
        "attack_type_distribution": df_raw["Attack_type"].value_counts().to_dict(),
        "numerical_feature_count": len(num_cols)
    }

    summary_path = "results/metrics/eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(eda_summary, f, indent=4)
        
    print(f"\n[STEP 3] Saved EDA Summary JSON to: [eda_summary.json](file:///{os.path.abspath(summary_path)})")
    print("=" * 70)
    print("PHASE 3 COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
