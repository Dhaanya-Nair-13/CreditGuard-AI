"""
train_xgboost_cv.py

5-Fold Stratified Cross Validation
for CreditGuard-AI
"""

import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from config import (
    MODEL_DIR,
    RANDOM_STATE,
)

from preprocessing import CreditRiskPreprocessor

from trainers.data_loader import DataLoader
from trainers.xgboost_trainer import XGBoostTrainer


def main():

    print("=" * 60)
    print("CreditGuard-AI")
    print("5-Fold Cross Validation")
    print("=" * 60)

    loader = DataLoader()

    X, y = loader.load_full_data()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores = []

    print("\nStarting Cross Validation...\n")

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print("=" * 60)
        print(f"Fold {fold}")
        print("=" * 60)

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_valid = y.iloc[valid_idx].copy()

        preprocessor = CreditRiskPreprocessor()

        X_train = preprocessor.fit_transform(X_train)
        X_valid = preprocessor.transform(X_valid)

        trainer = XGBoostTrainer()

        trainer.fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
        )

        probabilities = trainer.predict_proba(
            X_valid
        )[:, 1]

        roc = roc_auc_score(
            y_valid,
            probabilities,
        )

        fold_scores.append(roc)

        print(f"Fold ROC-AUC : {roc:.6f}")
        print("\n" + "=" * 60)
    print("Cross Validation Complete")
    print("=" * 60)

    for i, score in enumerate(fold_scores, start=1):
        print(f"Fold {i} ROC-AUC : {score:.6f}")

    print("-" * 60)
    print(f"Mean ROC-AUC     : {np.mean(fold_scores):.6f}")
    print(f"Std Deviation    : {np.std(fold_scores):.6f}")

    # ==========================================================
    # Train Final Model on Full Dataset
    # ==========================================================

    print("\n" + "=" * 60)
    print("Training Final Model on Full Dataset")
    print("=" * 60)

    preprocessor = CreditRiskPreprocessor()

    X_processed = preprocessor.fit_transform(X)

    trainer = XGBoostTrainer()

    trainer.fit(
        X_processed,
        y,
    )

    model = trainer.model

    print("\nSaving model...")

    joblib.dump(
        model,
        MODEL_DIR / "xgboost_cv.pkl",
    )

    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor_cv.pkl",
    )

    print("Model Saved.")

    print("\n" + "=" * 60)
    print("Cross Validation Finished Successfully")
    print("=" * 60)

if __name__ == "__main__":
    main()