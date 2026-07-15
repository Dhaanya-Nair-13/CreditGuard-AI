"""
prediction_pipeline.py

Production prediction pipeline for CreditGuard-AI.
"""

import joblib
import pandas as pd
import shap

from backend.ml.src.config import (
    MODEL_DIR,
    ARTIFACT_DIR,
)

DISPLAY_NAMES = {

    "num__EXT_SOURCE_MEAN": "External Credit Score (Mean)",
    "num__EXT_SOURCE_MAX": "External Credit Score (Maximum)",
    "num__EXT_SOURCE_MIN": "External Credit Score (Minimum)",
    "num__EXT_SOURCE_STD": "External Credit Score (Std Dev)",

    "num__EXT_SOURCE_1": "External Credit Score 1",
    "num__EXT_SOURCE_2": "External Credit Score 2",
    "num__EXT_SOURCE_3": "External Credit Score 3",

    "num__DAYS_BIRTH": "Age",

    "num__AMT_CREDIT": "Loan Amount",
    "num__AMT_ANNUITY": "Loan Annuity",
    "num__AMT_GOODS_PRICE": "Goods Price",

    "num__bureau_debt_credit_ratio": "Debt-to-Credit Ratio",

    "num__DEF_30_CNT_SOCIAL_CIRCLE":
        "30-Day Social Circle Defaults",
}


class PredictionPipeline:

    def __init__(self):

        print("=" * 60)
        print("Loading Prediction Pipeline")
        print("=" * 60)

        print("Loading Feature Store...")

        self.feature_store = pd.read_parquet(
            ARTIFACT_DIR / "demo_features.parquet"
        )

        print("Loading Model...")

        self.model = joblib.load(
            MODEL_DIR / "xgboost_optuna.pkl"
        )

        print("Loading Preprocessor...")

        self.preprocessor = joblib.load(
            MODEL_DIR / "preprocessor_optuna.pkl"
        )

        print("Initializing SHAP Explainer...")

        self.explainer = shap.TreeExplainer(
            self.model
        )

        print("Prediction Pipeline Ready.")

    def get_customer(
        self,
        customer_id,
    ):

        customer = self.feature_store[
            self.feature_store["SK_ID_CURR"] == customer_id
        ]

        if customer.empty:

            raise ValueError(
                f"Customer {customer_id} not found."
            )

        return customer.copy()

    def preprocess(
        self,
        customer,
    ):

        customer = customer.drop(
            columns=["TARGET"],
            errors="ignore",
        )

        processed = self.preprocessor.transform(
            customer
        )

        return processed

    def predict_probability(
        self,
        customer_id,
    ):

        customer = self.get_customer(
            customer_id
        )

        processed = self.preprocess(
            customer
        )

        probability = self.model.predict_proba(
            processed
        )[0][1]

        return probability

    def predict(
        self,
        customer_id,
    ):

        probability = self.predict_probability(
            customer_id
        )

        if probability >= 0.65:
            risk = "High Risk"

        elif probability >= 0.35:
            risk = "Medium Risk"

        else:
            risk = "Low Risk"

        return {

            "customer_id": customer_id,

            "probability": round(
                float(probability),
                4,
            ),

            "risk": risk,

        }

    def explain(
        self,
        customer_id,
        top_n=10,
    ):

        customer = self.get_customer(
            customer_id
        )

        processed = self.preprocess(
            customer
        )

        shap_values = self.explainer(
            processed
        )

        feature_names = list(
            self.preprocessor.preprocessor.get_feature_names_out()
        )

        explanation = pd.DataFrame(
            {
                "feature": feature_names,
                "impact": shap_values.values[0],
            }
        )

        explanation["abs"] = explanation[
            "impact"
        ].abs()

        explanation = explanation.sort_values(
            by="abs",
            ascending=False,
        ).head(top_n)

        explanation = explanation.drop(
            columns="abs"
        )

        explanation["feature"] = (
            explanation["feature"]
            .replace(DISPLAY_NAMES)
        )

        explanation["direction"] = explanation[
            "impact"
        ].apply(
            lambda x:
            "Increases Risk"
            if x > 0
            else "Reduces Risk"
        )

        explanation["impact"] = explanation[
            "impact"
        ].apply(
            lambda x: round(float(x), 4)
        )

        return explanation

    def predict_with_explanation(
        self,
        customer_id,
    ):

        prediction = self.predict(
            customer_id
        )

        prediction["top_factors"] = (
            self.explain(
                customer_id
            ).to_dict(
                orient="records"
            )
        )

        return prediction