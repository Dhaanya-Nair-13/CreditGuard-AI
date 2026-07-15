"""
Train machine learning model.
"""

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import (
    MODEL_DIR,
    ARTIFACT_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_DATA,
)
from preprocessing import CreditRiskPreprocessor
from sklearn.ensemble import RandomForestClassifier
from feature_engineering import FeatureEngineer
from xgboost import XGBClassifier
from evaluate import ModelEvaluator
from threshold_optimizer import ThresholdOptimizer



def main():

    print("Loading dataset...")

    df = pd.read_csv(TRAIN_DATA)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    engineer = FeatureEngineer()
    X = engineer.transform(X)

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Preprocessing data...")

    preprocessor = CreditRiskPreprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Training XGBoost...")

    model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    )

    model.fit(X_train_processed, y_train)

    
    probabilities = model.predict_proba(X_test_processed)[:, 1]

    optimizer = ThresholdOptimizer()

    best_threshold, threshold_results = optimizer.find_best_threshold(
    y_test,
    probabilities
    )
    predictions = (probabilities >= best_threshold).astype(int)

    print("\nBest Threshold:", best_threshold)
    print("\nThreshold Results")
    print(threshold_results)

    threshold_results.to_csv(
    ARTIFACT_DIR / "threshold_results.csv",
    index=False
    )

    evaluator = ModelEvaluator(ARTIFACT_DIR)
    evaluator.save_confusion_matrix(
    y_test,
    predictions
    )
    evaluator.save_roc_curve(
    y_test,
    probabilities
    )
    evaluator.save_precision_recall_curve(
    y_test,
    probabilities
    )


    print("\n========== RESULTS ==========\n")

    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions):.4f}")
    print(f"F1 Score : {f1_score(y_test, predictions):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, probabilities):.4f}")

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))

    print("\nSaving model...")

    joblib.dump(model, MODEL_DIR / "xgboost.pkl")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")

    print("Done!")


if __name__ == "__main__":
    main()