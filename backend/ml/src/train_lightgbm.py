"""
train_lightgbm.py

Train LightGBM on the engineered CreditGuard-AI dataset.
"""

import joblib

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
)

from threshold_optimizer import ThresholdOptimizer
from evaluate import ModelEvaluator

from trainers.data_loader import DataLoader
from trainers.lightgbm_trainer import LightGBMTrainer


def main():

    print("=" * 60)
    print("CreditGuard-AI")
    print("LightGBM Training")
    print("=" * 60)

    loader = DataLoader()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = loader.load_data()

    print("\nTraining LightGBM...\n")

    trainer = LightGBMTrainer()

    trainer.fit(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    probabilities = trainer.predict_proba(
        X_test
    )[:, 1]

    optimizer = ThresholdOptimizer()

    best_threshold, threshold_table = (
        optimizer.find_best_threshold(
            y_test,
            probabilities,
        )
    )

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    threshold_table.to_csv(
        ARTIFACT_DIR / "threshold_results_lightgbm.csv",
        index=False,
    )

    evaluator = ModelEvaluator(
        ARTIFACT_DIR
    )

    evaluator.save_confusion_matrix(
        y_test,
        predictions,
    )

    evaluator.save_roc_curve(
        y_test,
        probabilities,
    )

    evaluator.save_precision_recall_curve(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Best Threshold : {best_threshold:.2f}")
    print(f"Accuracy       : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision      : {precision_score(y_test, predictions):.4f}")
    print(f"Recall         : {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score       : {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC        : {roc_auc_score(y_test, probabilities):.4f}")

    print("\nSaving model...")

    trainer.save(
        MODEL_DIR / "lightgbm.pkl"
    )

    joblib.dump(
        preprocessor,
        MODEL_DIR / "lightgbm_preprocessor.pkl",
    )

    print("Done!")


if __name__ == "__main__":
    main()