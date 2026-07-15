"""
shap_explainer.py

SHAP explainability for CreditGuard-AI
"""

import joblib
import shap
import pandas as pd

from config import MODEL_DIR


class SHAPExplainer:

    def __init__(self):

        print("Loading model...")

        self.model = joblib.load(
            MODEL_DIR / "xgboost_optuna.pkl"
        )

        self.preprocessor = joblib.load(
            MODEL_DIR / "preprocessor_optuna.pkl"
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )

        print("SHAP Explainer Ready.")

    def explain(self, raw_dataframe: pd.DataFrame):

        processed = self.preprocessor.transform(
            raw_dataframe
        )

        shap_values = self.explainer.shap_values(
            processed
        )

        feature_names = self.preprocessor.get_feature_names()

        explanation = pd.DataFrame({

            "Feature": feature_names,

            "SHAP_Value": shap_values[0]

        })

        explanation["Abs_SHAP"] = (
            explanation["SHAP_Value"]
            .abs()
        )

        explanation = explanation.sort_values(

            by="Abs_SHAP",

            ascending=False,

        )

        return explanation.drop(
            columns="Abs_SHAP"
        )