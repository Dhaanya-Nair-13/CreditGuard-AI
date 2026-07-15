"""
train_xgboost_oof.py

5-Fold Out-of-Fold XGBoost Training
"""

import joblib
import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import (
    MODEL_DIR,
    ARTIFACT_DIR,
    RANDOM_STATE,
)

from trainers.data_loader import DataLoader
from trainers.xgboost_trainer import XGBoostTrainer

from threshold_optimizer import ThresholdOptimizer
from evaluate import ModelEvaluator


def main():

    print("=" * 60)
    print("CreditGuard-AI")
    print("5-Fold Out-of-Fold XGBoost")
    print("=" * 60)

    loader = DataLoader()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = loader.load_data()

    print("\nPreparing 5-fold cross validation...")

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_predictions = np.zeros(len(y_train))

    test_predictions = np.zeros(
        (len(y_test), 5)
    )

    models_dir = MODEL_DIR / "oof_models"
    models_dir.mkdir(
        exist_ok=True,
        parents=True,
    )