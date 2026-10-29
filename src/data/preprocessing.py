"""
Preprocessing Pipeline Module for Cyber Threat Detection Framework.
Creates sklearn ColumnTransformer for numerical scaling and categorical encoding fitted ONLY on training data.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def build_preprocessing_pipeline(
    X_train: pd.DataFrame
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build sklearn ColumnTransformer based on training feature data types.
    Uses OrdinalEncoder for categorical features to maintain compact memory footprint and tree compatibility.
    
    Args:
        X_train: Training DataFrame (untransformed).
        
    Returns:
        preprocessor: Unfitted ColumnTransformer.
        num_cols: List of numerical column names.
        cat_cols: List of categorical column names.
    """
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"[PREPROCESSING] Numerical features ({len(num_cols)}): {num_cols[:5]}...")
    print(f"[PREPROCESSING] Categorical features ({len(cat_cols)}): {cat_cols}...")

    # Pipeline for numerical features: Median imputation -> Standard scaling
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: Most frequent imputation -> OrdinalEncoder
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    transformers = []
    if num_cols:
        transformers.append(('num', num_pipeline, num_cols))
    if cat_cols:
        transformers.append(('cat', cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor, num_cols, cat_cols


def fit_and_transform_pipeline(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Fit preprocessor ONLY on X_train, then transform both X_train and X_test to enforce NO DATA LEAKAGE.
    
    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training features.
        X_test: Testing features.
        
    Returns:
        X_train_proc: Processed numpy array for train.
        X_test_proc: Processed numpy array for test.
        preprocessor: Fitted ColumnTransformer.
        feature_names: List of output feature names after transformation.
    """
    # Ensure categorical columns are explicitly string-typed across both splits
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    for c in cat_cols:
        X_train_copy[c] = X_train_copy[c].astype(str)
        X_test_copy[c] = X_test_copy[c].astype(str)

    print("[PREPROCESSING] Fitting preprocessor pipeline ONLY on X_train to prevent data leakage...")
    X_train_proc = preprocessor.fit_transform(X_train_copy)
    
    print("[PREPROCESSING] Transforming X_test using already-fitted pipeline...")
    X_test_proc = preprocessor.transform(X_test_copy)

    # Get feature names post-transformation
    feature_names = []
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    print(f"[PREPROCESSING] Transformation Complete. Output Feature Matrix Shape: {X_train_proc.shape[1]} features.")
    return X_train_proc, X_test_proc, preprocessor, feature_names


def save_preprocessor(preprocessor: ColumnTransformer, path: str = "models/preprocessing.joblib"):
    """Save fitted preprocessor artifact to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"[PREPROCESSING] Saved fitted preprocessing artifact to: [preprocessing.joblib](file:///{os.path.abspath(path)})")


def load_preprocessor(path: str = "models/preprocessing.joblib") -> ColumnTransformer:
    """Load fitted preprocessor artifact from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor artifact not found at: {path}")
    return joblib.load(path)
