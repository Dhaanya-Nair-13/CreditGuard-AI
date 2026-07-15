"""
train_ensemble.py

Leak-free weighted XGBoost + LightGBM ensemble.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config import RANDOM_STATE

from trainers.data_loader import DataLoader
from trainers.xgboost_trainer import XGBoostTrainer
from trainers.lightgbm_trainer import LightGBMTrainer

from threshold_optimizer import ThresholdOptimizer


def main():

    print("=" * 60)
    print("CreditGuard-AI")
    print("Leak-Free Ensemble Training")
    print("=" * 60)

    ########################################################
    # Load Data
    ########################################################

    loader = DataLoader()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = loader.load_data()

    ########################################################
    # Validation Split
    ########################################################

    print("\nCreating validation split...\n")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    ########################################################
    # Train XGBoost
    ########################################################

    print("Training XGBoost...")

    xgb = XGBoostTrainer()

    xgb.fit(
        X_tr,
        y_tr,
        X_val,
        y_val,
    )

    xgb_val_prob = xgb.predict_proba(X_val)[:, 1]

    ########################################################
    # Train LightGBM
    ########################################################

    print("Training LightGBM...")

    lgbm = LightGBMTrainer()

    lgbm.fit(
        X_tr,
        y_tr,
        X_val,
        y_val,
    )

    lgbm_val_prob = lgbm.predict_proba(X_val)[:, 1]

    ########################################################
    # Weight Search (Validation Only)
    ########################################################

    print("\nSearching blend weights...\n")

    weights = np.arange(0.50, 1.001, 0.01)

    results = []

    for w in weights:

        ensemble = (
            w * xgb_val_prob +
            (1 - w) * lgbm_val_prob
        )

        auc = roc_auc_score(
            y_val,
            ensemble,
        )

        results.append({

            "weight": round(w, 2),

            "auc": auc,

        })

    results = pd.DataFrame(results)

    results = results.sort_values(
        by="auc",
        ascending=False,
    ).reset_index(drop=True)

    best_weight = results.iloc[0]["weight"]

    print(results.head(10))

    print("\nBest Weight:", best_weight)

    ########################################################
    # Threshold Search
    ########################################################

    optimizer = ThresholdOptimizer()

    validation_prob = (
        best_weight * xgb_val_prob +
        (1 - best_weight) * lgbm_val_prob
    )

    best_threshold, _ = optimizer.find_best_threshold(
        y_val,
        validation_prob,
    )

    print("Best Threshold:", best_threshold)
        ########################################################
    # Retrain BOTH models on FULL training data
    ########################################################

    print("\nRetraining models on full training set...\n")

    xgb_final = XGBoostTrainer()

    xgb_final.fit(
        X_train,
        y_train,
    )

    lgbm_final = LightGBMTrainer()

    lgbm_final.fit(
        X_train,
        y_train,
    )

    ########################################################
    # Predict on TEST (first and only evaluation)
    ########################################################

    xgb_test_prob = xgb_final.predict_proba(
        X_test
    )[:, 1]

    lgbm_test_prob = lgbm_final.predict_proba(
        X_test
    )[:, 1]

    final_prob = (
        best_weight * xgb_test_prob +
        (1 - best_weight) * lgbm_test_prob
    )

    predictions = (
        final_prob >= best_threshold
    ).astype(int)

    ########################################################
    # Final Metrics
    ########################################################

    final_auc = roc_auc_score(
        y_test,
        final_prob,
    )

    final_accuracy = accuracy_score(
        y_test,
        predictions,
    )

    final_precision = precision_score(
        y_test,
        predictions,
    )

    final_recall = recall_score(
        y_test,
        predictions,
    )

    final_f1 = f1_score(
        y_test,
        predictions,
    )

    ########################################################
    # Print Results
    ########################################################

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    print(f"Blend Weight (XGB) : {best_weight:.2f}")
    print(f"Blend Weight (LGB) : {1-best_weight:.2f}")
    print(f"Threshold          : {best_threshold:.2f}")
    print(f"Accuracy           : {final_accuracy:.4f}")
    print(f"Precision          : {final_precision:.4f}")
    print(f"Recall             : {final_recall:.4f}")
    print(f"F1 Score           : {final_f1:.4f}")
    print(f"ROC-AUC            : {final_auc:.6f}")

    ########################################################
    # Save Results
    ########################################################

    pd.DataFrame([{
        "xgb_weight": best_weight,
        "lgb_weight": 1 - best_weight,
        "threshold": best_threshold,
        "accuracy": final_accuracy,
        "precision": final_precision,
        "recall": final_recall,
        "f1": final_f1,
        "roc_auc": final_auc,
    }]).to_csv(
        "backend/ml/artifacts/final_ensemble_results.csv",
        index=False,
    )

    print("\nResults saved.")

if __name__ == "__main__":
    main()