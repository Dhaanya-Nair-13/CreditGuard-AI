"""
Train XGBoost on enriched Home Credit dataset.
"""

import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
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
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_DATA,
    BUREAU_DATA,
    BUREAU_BALANCE_DATA,
    PREVIOUS_APPLICATION_DATA,
    INSTALLMENTS_DATA,
    CREDIT_CARD_DATA,
    POS_CASH_DATA,
)

from preprocessing import CreditRiskPreprocessor
from feature_engineering import FeatureEngineer
from feature_pipeline.merge_features import FeatureMerger

from threshold_optimizer import ThresholdOptimizer
from evaluate import ModelEvaluator


def main():

    print("=" * 60)
    print("CreditGuard-AI")
    print("=" * 60)

    print("\nLoading enriched dataset...\n")

    df = FeatureMerger(
        TRAIN_DATA,
        BUREAU_DATA,
        BUREAU_BALANCE_DATA,
        PREVIOUS_APPLICATION_DATA,
        INSTALLMENTS_DATA,
        CREDIT_CARD_DATA,
        POS_CASH_DATA,
    ).transform()

    print("\nDataset Loaded.")
    print(df.shape)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print("\nApplying Feature Engineering...")

    X = FeatureEngineer().transform(X)

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nPreprocessing...")

    preprocessor = CreditRiskPreprocessor()

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print("\nTraining XGBoost...\n")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",

        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,

        subsample=1.0,
        colsample_bytree=0.7,

        gamma=0.5,
        min_child_weight=3,

        reg_alpha=0,
        reg_lambda=5,

        scale_pos_weight=11.3,

        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    optimizer = ThresholdOptimizer()

    best_threshold, threshold_table = optimizer.find_best_threshold(
        y_test,
        probabilities,
    )

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    threshold_table.to_csv(
        ARTIFACT_DIR / "threshold_results_enriched.csv",
        index=False,
    )

    evaluator = ModelEvaluator(ARTIFACT_DIR)

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

    joblib.dump(
        model,
        MODEL_DIR / "xgboost_enriched.pkl",
    )

    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessor_enriched.pkl",
    )

    print("Done!")


if __name__ == "__main__":
    main()