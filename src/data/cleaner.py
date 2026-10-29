"""
Data Cleaning Module for Cyber Threat Detection Framework.
Removes constant columns, identifier leakage columns, and cleans infinite/invalid values with detailed justifications.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# Explicit list of leakage and identifier columns with cybersecurity justifications
EXPLICIT_LEAKAGE_COLUMNS: Dict[str, str] = {
    "frame.time": "Timestamp column overfits model to the specific experiment execution time.",
    "ip.src_host": "Source IP address causes identifier leakage (memorizing specific IP addresses rather than learning general network flow behavior).",
    "ip.dst_host": "Destination IP address causes identifier leakage.",
    "arp.dst.proto_ipv4": "Device IP protocol string encoding causing device identity leakage.",
    "arp.src.proto_ipv4": "Device IP protocol string encoding causing device identity leakage.",
    "tcp.payload": "Raw TCP payload string causes severe textual memorization leakage.",
    "tcp.options": "Raw TCP options hex string contains specific connection state bytes causing overfitting and leakage.",
    "http.file_data": "Raw HTTP response file payload string causes overfitting.",
    "http.request.full_uri": "Full HTTP URI string contains specific target domain names and paths.",
    "http.referer": "HTTP referer header string causes domain memorization leakage.",
    "http.request.uri.query": "HTTP URI query string overfits to specific attack payload syntax.",
    "mqtt.msg": "Raw MQTT message payload string overfits to specific application message text.",
}

# Target columns cross-leakage
TARGET_COLUMNS = ["Attack_label", "Attack_type"]

# Explicit list of columns that contain numeric port numbers or counts stored as strings/objects
KNOWN_NUMERIC_STRING_COLUMNS = [
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", 
    "http.request.version", "dns.qry.name.len", "mqtt.conack.flags",
    "http.response", "http.content_length"
]


def convert_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert explicit numeric string columns (e.g. port numbers, lengths) into float dtypes."""
    df_converted = df.copy()
    for col in KNOWN_NUMERIC_STRING_COLUMNS:
        if col in df_converted.columns:
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
    return df_converted


def get_constant_columns(df: pd.DataFrame) -> List[str]:
    """Find columns with zero variance (single unique value)."""
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    return constant_cols


def clean_dataframe(
    df: pd.DataFrame, 
    target_col: str = "Attack_label",
    drop_leakage: bool = True
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Clean dataset feature set by dropping constant columns, leakage columns, and handling invalid values.
    
    Args:
        df: Input DataFrame.
        target_col: Current target column being modeled.
        drop_leakage: Whether to drop explicit leakage/identifier columns.
        
    Returns:
        df_cleaned: Cleaned DataFrame.
        dropped_cols: List of removed column names.
        removal_reasons: Dict mapping removed column name to detailed justification.
    """
    df_cleaned = df.copy()
    dropped_cols = []
    removal_reasons = {}

    # 1. Drop other target column(s) to avoid target cross-leakage
    for t_col in TARGET_COLUMNS:
        if t_col != target_col and t_col in df_cleaned.columns:
            df_cleaned.drop(columns=[t_col], inplace=True)
            dropped_cols.append(t_col)
            removal_reasons[t_col] = f"Secondary target column dropped to prevent direct target leakage into '{target_col}'."

    # 2. Convert explicit string port numbers and numeric-like strings to numeric floats
    df_cleaned = convert_numeric_strings(df_cleaned)

    # 3. Identify & Drop Constant Columns
    constant_cols = get_constant_columns(df_cleaned)
    for col in constant_cols:
        if col in df_cleaned.columns:
            df_cleaned.drop(columns=[col], inplace=True)
            dropped_cols.append(col)
            removal_reasons[col] = "Constant feature (zero variance) provides zero predictive information."

    # 4. Drop Explicit Leakage & Identifier Columns
    if drop_leakage:
        for col, reason in EXPLICIT_LEAKAGE_COLUMNS.items():
            if col in df_cleaned.columns:
                df_cleaned.drop(columns=[col], inplace=True)
                dropped_cols.append(col)
                removal_reasons[col] = f"Data Leakage / Identifier Risk: {reason}"

    # 5. Handle Infinite values (replace inf / -inf with NaN for imputer)
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned[num_cols] = df_cleaned[num_cols].replace([np.inf, -np.inf], np.nan)

    print(f"[DATA CLEANER] Dropped {len(dropped_cols)} features out of {df.shape[1]} original features.")
    print(f"[DATA CLEANER] Remaining features: {df_cleaned.shape[1]}")
    
    return df_cleaned, dropped_cols, removal_reasons
