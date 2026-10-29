"""
Phase 7: Model Combination & Ensemble Detector Module.
Combines Random Forest and LightGBM predictions using soft-voting probability weighting.
"""

import os
import joblib
import numpy as np
from typing import Tuple, Dict, Any


class HybridThreatDetector:
    """
    Hybrid Threat Detector combining Bagging (Random Forest) and Boosting (LightGBM).
    
    Formula:
        P_final(Attack) = α * P_RF(Attack) + (1 - α) * P_LGBM(Attack)
    """

    def __init__(self, rf_model=None, lgbm_model=None, alpha: float = 0.5):
        self.rf_model = rf_model
        self.lgbm_model = lgbm_model
        self.alpha = alpha

    def load_models(
        self, 
        rf_path: str = "models/random_forest.joblib", 
        lgbm_path: str = "models/lightgbm_model.joblib"
    ):
        """Load pre-trained Random Forest and LightGBM models."""
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"Random Forest model not found at {rf_path}")
        if not os.path.exists(lgbm_path):
            raise FileNotFoundError(f"LightGBM model not found at {lgbm_path}")

        self.rf_model = joblib.load(rf_path)
        self.lgbm_model = joblib.load(lgbm_path)
        print(f"[HYBRID DETECTOR] Successfully loaded Random Forest & LightGBM models (alpha={self.alpha}).")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate weighted ensemble probabilities.
        
        Returns:
            np.ndarray of shape (N, 2) containing [P(Normal), P(Attack)]
        """
        if self.rf_model is None or self.lgbm_model is None:
            raise ValueError("Models not initialized or loaded.")

        rf_probs = self.rf_model.predict_proba(X)
        lgbm_probs = self.lgbm_model.predict_proba(X)

        ensemble_probs = (self.alpha * rf_probs) + ((1.0 - self.alpha) * lgbm_probs)
        return ensemble_probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary attack class using threshold."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def save_ensemble(self, path: str = "models/hybrid_ensemble.joblib"):
        """Save hybrid ensemble configuration and references."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[HYBRID DETECTOR] Saved hybrid ensemble to: [hybrid_ensemble.joblib](file:///{os.path.abspath(path)})")
