"""
model.py
--------
Model definitions and training routines for candidate pair classification.
Supports Logistic Regression and LightGBM / Gradient Boosting.
"""

import pickle
import numpy as np
from typing import Any, Tuple, Optional

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


class EntityPairClassifier:
    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the classification model on training feature matrix and labels."""
        if self.model_type == "lightgbm" and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.08,
                num_leaves=31,
                random_state=self.random_state,
                verbose=-1
            )
        elif self.model_type in ("lightgbm", "gbdt"):
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.08,
                random_state=self.random_state
            )
        else:
            self.model = LogisticRegression(
                max_iter=500,
                C=1.0,
                random_state=self.random_state
            )

        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs probability of positive match for each candidate pair."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "EntityPairClassifier":
        with open(filepath, "rb") as f:
            return pickle.load(f)
