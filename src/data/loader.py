"""
Data Loader Module for Cyber Threat Detection Framework.
Handles raw dataset loading and leak-free stratified train/test splitting.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def load_raw_dataset(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV dataset.
    
    Args:
        file_path: Path to dataset CSV file.
    Returns:
        pd.DataFrame raw dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {file_path}")
    
    print(f"[DATA LOADER] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"[DATA LOADER] Dataset successfully loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def split_dataset(
    df: pd.DataFrame, 
    target_col: str = "Attack_label", 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform leak-free stratified train-test split before any preprocessing/fitting.
    
    Args:
        df: Input DataFrame containing features and targets.
        target_col: Column name to use as target.
        test_size: Fraction of dataset reserved for testing.
        random_state: Seed for reproducibility.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns.")
        
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print(f"[DATA LOADER] Performing Stratified Train-Test Split (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[DATA LOADER] Train set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")
    return X_train, X_test, y_train, y_test
