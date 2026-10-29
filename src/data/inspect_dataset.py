"""
Phase 1: Dataset Inspection Script for Edge-IIoTset / Cybersecurity Datasets.
Identifies schema, data types, missing values, duplicates, target distribution, and leakage risks.
"""

import os
import sys
import argparse
import json
import yaml
import pandas as pd
import numpy as np

# Leakage keywords to watch out for in network security datasets
POTENTIAL_LEAKAGE_KEYWORDS = [
    "ip", "mac", "timestamp", "time", "date", "id", "index", 
    "src_ip", "dst_ip", "source_ip", "destination_ip", "payload"
]


def load_config(config_path="config.yaml"):
    """Load configuration YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def inspect_dataset(file_path: str, max_sample_rows: int = None):
    """
    Perform deep inspection of dataset file.
    
    Args:
        file_path: Path to dataset CSV file.
        max_sample_rows: Optional limit on rows to load for quick inspection.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] Dataset file not found at: {file_path}")
        print("Please ensure the CSV file is placed in data/raw/ or pass the correct path.")
        sys.exit(1)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print("=" * 70)
    print("PHASE 1: CYBERSECURITY DATASET INSPECTION REPORT")
    print("=" * 70)
    print(f"Dataset File Path : {file_path}")
    print(f"File Size         : {file_size_mb:.2f} MB")

    # Load dataset
    print("\n[1/6] Loading Dataset...")
    if max_sample_rows:
        df = pd.read_csv(file_path, nrows=max_sample_rows, low_memory=False)
        print(f"Loaded sample of {len(df):,} rows.")
    else:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Loaded full dataset of {len(df):,} rows.")

    rows, cols = df.shape
    print(f"\n[2/6] General Overview:")
    print(f"  - Total Rows    : {rows:,}")
    print(f"  - Total Columns : {cols}")

    # Memory usage
    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"  - Memory Usage  : {mem_usage_mb:.2f} MB")

    # Missing values
    print("\n[3/6] Missing Values & Duplicates:")
    missing_summary = df.isnull().sum()
    cols_with_missing = missing_summary[missing_summary > 0]
    if len(cols_with_missing) == 0:
        print("  - Missing Values: None found (0 missing entries).")
    else:
        print(f"  - Columns with Missing Values ({len(cols_with_missing)} columns):")
        for col, count in cols_with_missing.items():
            pct = (count / rows) * 100
            print(f"    * {col}: {count:,} ({pct:.2f}%)")

    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / rows) * 100
    print(f"  - Duplicate Rows: {duplicate_rows:,} ({duplicate_pct:.2f}%)")

    # Columns & Data Types
    print("\n[4/6] Column Types & Unique Counts:")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"  - Numerical Columns   ({len(num_cols)}): {num_cols[:10]}{'...' if len(num_cols)>10 else ''}")
    print(f"  - Categorical Columns ({len(cat_cols)}): {cat_cols[:10]}{'...' if len(cat_cols)>10 else ''}")

    # Search for Target Variable candidates
    print("\n[5/6] Target Variable Inspection:")
    target_candidates = [col for col in df.columns if any(t in col.lower() for t in ["target", "label", "attack", "class"])]
    print(f"  - Identified Target Candidates: {target_candidates}")

    target_info = {}
    for col in target_candidates:
        print(f"\n  --- Class Distribution for '{col}' ---")
        counts = df[col].value_counts(dropna=False)
        percentages = df[col].value_counts(normalize=True, dropna=False) * 100
        dist = {}
        for val, cnt in counts.items():
            pct = percentages[val]
            print(f"    * {val}: {cnt:,} ({pct:.2f}%)")
            dist[str(val)] = {"count": int(cnt), "percentage": float(pct)}
        target_info[col] = dist

    # Data Leakage Risks Identification
    print("\n[6/6] Data Leakage Analysis & High-Risk Columns:")
    leakage_suspects = []
    constant_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        unique_cnt = df[col].nunique(dropna=False)
        
        if unique_cnt <= 1:
            constant_cols.append(col)
            
        if any(keyword in col_lower for keyword in POTENTIAL_LEAKAGE_KEYWORDS) and col not in target_candidates:
            leakage_suspects.append({
                "column": col,
                "dtype": str(df[col].dtype),
                "unique_values": int(unique_cnt),
                "sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
            })

    if constant_cols:
        print(f"  - Constant Columns (Zero variance, should be dropped): {constant_cols}")
    else:
        print("  - Constant Columns: None detected.")

    if leakage_suspects:
        print(f"  - Potential Leakage/Identifier Columns ({len(leakage_suspects)} detected):")
        for s in leakage_suspects:
            print(f"    * {s['column']} (type: {s['dtype']}, unique: {s['unique_values']:,}) -> sample: {s['sample']}")
    else:
        print("  - No obvious identifier leakage columns found by keyword rules.")

    # Save detailed JSON summary
    output_dir = "results/metrics"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "dataset_inspection_report.json")
    
    report_data = {
        "file_path": file_path,
        "file_size_mb": file_size_mb,
        "rows": rows,
        "columns": cols,
        "missing_values_count": int(df.isnull().sum().sum()),
        "duplicate_rows": int(duplicate_rows),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "target_distributions": target_info,
        "constant_columns": constant_cols,
        "potential_leakage_columns": leakage_suspects
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=4)
        
    print(f"\nInspection summary successfully saved to: [dataset_inspection_report.json](file:///{os.path.abspath(report_path)})")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Cybersecurity Dataset for Phase 1")
    parser.add_argument("--file", type=str, default=None, help="Path to raw CSV dataset file")
    parser.add_argument("--sample", type=int, default=None, help="Limit number of rows to load for fast inspection")
    args = parser.parse_args()

    config = load_config()
    target_file = args.file or config.get("dataset", {}).get("primary_raw_file", "data/raw/Edge-IIoTset_dataset.csv")

    # If the file does not exist at default location, search data/raw for any .csv file
    if not os.path.exists(target_file) and os.path.exists("data/raw"):
        csv_files = [os.path.join("data/raw", f) for f in os.listdir("data/raw") if f.endswith(".csv")]
        if csv_files:
            target_file = csv_files[0]
            print(f"[INFO] Found CSV dataset in data/raw/: {target_file}")

    inspect_dataset(target_file, max_sample_rows=args.sample)
