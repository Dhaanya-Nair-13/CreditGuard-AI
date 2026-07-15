"""
Fast Optuna Hyperparameter Tuning
CreditGuard-AI

Uses:
- Train / Validation split
- Early Stopping
- Optuna
- XGBoost
"""

import json
import joblib
import optuna

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TRAIN_DATA,
    BUREAU_DATA,
    BUREAU_BALANCE_DATA,
    PREVIOUS_APPLICATION_DATA,
    INSTALLMENTS_DATA,
    CREDIT_CARD_DATA,
    POS_CASH_DATA,
    MODEL_DIR,
    ARTIFACT_DIR,
)

from preprocessing import CreditRiskPreprocessor
from feature_engineering import FeatureEngineer
from feature_pipeline.merge_features import FeatureMerger


print("=" * 60)
print("CreditGuard-AI Optuna Tuning")
print("=" * 60)

print("\nLoading Dataset...\n")

df = FeatureMerger(
    TRAIN_DATA,
    BUREAU_DATA,
    BUREAU_BALANCE_DATA,
    PREVIOUS_APPLICATION_DATA,
    INSTALLMENTS_DATA,
    CREDIT_CARD_DATA,
    POS_CASH_DATA,
).transform()

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

print(df.shape)

print("\nFeature Engineering...")

X = FeatureEngineer().transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(

    X,

    y,

    test_size=0.20,

    stratify=y,

    random_state=RANDOM_STATE,

)

print("\nPreprocessing...")

preprocessor = CreditRiskPreprocessor()

X_train = preprocessor.fit_transform(X_train)

X_valid = preprocessor.transform(X_valid)

print("Dataset Ready.\n")


def objective(trial):

    params = {

        "objective": "binary:logistic",

        "eval_metric": "auc",

        "tree_method": "hist",

        "random_state": RANDOM_STATE,

        "n_jobs": -1,

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            8,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.10,
            log=True,
        ),

        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            900,
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.65,
            1.0,
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.0,
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0,
            4,
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            8,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            0,
            3,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1,
            8,
        ),

        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            8,
            13,
        ),

    }

    model = XGBClassifier(
        **params,
        early_stopping_rounds=50,
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[(X_valid, y_valid)],

        verbose=False,
        

    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    score = roc_auc_score(

        y_valid,

        probabilities,

    )

    return score
if __name__ == "__main__":

    print("=" * 60)
    print("Starting Optuna Optimization")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        study_name="CreditGuardAI_XGBoost"
    )

    study.optimize(

        objective,

        n_trials=75,

        show_progress_bar=True,

    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION FINISHED")
    print("=" * 60)

    print(f"\nBest ROC-AUC : {study.best_value:.6f}\n")

    print("Best Parameters\n")

    for key, value in study.best_params.items():

        print(f"{key:20} : {value}")

    # --------------------------------------------------
    # Save Parameters
    # --------------------------------------------------

    params_path = (
        ARTIFACT_DIR /
        "best_xgboost_params.json"
    )

    with open(params_path, "w") as f:

        json.dump(
            study.best_params,
            f,
            indent=4,
        )

    print("\nParameters Saved.")

    # --------------------------------------------------
    # Train Final Model
    # --------------------------------------------------

    print("\nTraining Final Model...\n")

    final_params = study.best_params.copy()

    final_params.update({

        "objective": "binary:logistic",

        "eval_metric": "auc",

        "tree_method": "hist",

        "random_state": RANDOM_STATE,

        "n_jobs": -1,

    })

    final_model = XGBClassifier(
        **final_params,
        early_stopping_rounds=50,
    )

    final_model.fit(

        X_train,

        y_train,

        eval_set=[(X_valid, y_valid)],

        verbose=False,
        

    )

    probabilities = final_model.predict_proba(
        X_valid
    )[:, 1]

    roc = roc_auc_score(
        y_valid,
        probabilities,
    )

    print(f"\nValidation ROC-AUC : {roc:.6f}")

    # --------------------------------------------------
    # Save Model
    # --------------------------------------------------

    joblib.dump(

        final_model,

        MODEL_DIR /
        "xgboost_optuna.pkl",

    )

    joblib.dump(

        preprocessor,

        MODEL_DIR /
        "preprocessor_optuna.pkl",

    )

    joblib.dump(

        study,

        ARTIFACT_DIR /
        "optuna_study.pkl",

    )

    print("\nModel Saved.")
    print("Study Saved.")
    print("Done.")
