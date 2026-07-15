"""
installment_features.py

Creates customer-level features from installments_payments.csv
"""

import pandas as pd

from feature_pipeline.utils import FeatureAggregator


class InstallmentFeatureEngineer:

    def __init__(self, installments_path):

        self.installments_path = installments_path

    def transform(self):

        print("Loading installments_payments.csv...")

        ins = pd.read_csv(self.installments_path)

        print("Generating installment features...")

        # ------------------------------------------------
        # Business Features
        # ------------------------------------------------

        ins["PAYMENT_DIFF"] = (
            ins["AMT_PAYMENT"] -
            ins["AMT_INSTALMENT"]
        )

        ins["PAYMENT_RATIO"] = (
            ins["AMT_PAYMENT"] /
            ins["AMT_INSTALMENT"].replace(0, 1)
        )

        ins["LATE_DAYS"] = (
            ins["DAYS_ENTRY_PAYMENT"] -
            ins["DAYS_INSTALMENT"]
        )

        ins["LATE_PAYMENT"] = (
            ins["LATE_DAYS"] > 0
        ).astype(int)

        features = pd.DataFrame({

            "SK_ID_CURR":
                ins["SK_ID_CURR"].unique()

        })

        # ------------------------------------------------
        # Installment Count
        # ------------------------------------------------

        count = (

            ins.groupby("SK_ID_CURR")
            .size()
            .reset_index(
                name="installment_count"
            )

        )

        features = FeatureAggregator.merge(

            features,
            count,
            "SK_ID_CURR"

        )

        # ------------------------------------------------
        # Numerical Statistics
        # ------------------------------------------------

        numeric_columns = [

            "AMT_INSTALMENT",
            "AMT_PAYMENT",

            "PAYMENT_DIFF",
            "PAYMENT_RATIO",

            "LATE_DAYS",

            "NUM_INSTALMENT_NUMBER",
            "NUM_INSTALMENT_VERSION",

            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT"

        ]

        stats = [

            "mean",
            "sum",
            "max",
            "min",
            "std"

        ]

        features = FeatureAggregator.add_stat_features(

            features,
            ins,
            "SK_ID_CURR",
            numeric_columns,
            stats

        )

        # ------------------------------------------------
        # Late Payment Count
        # ------------------------------------------------

        late = (

            ins.groupby("SK_ID_CURR")[
                "LATE_PAYMENT"
            ]
            .sum()
            .reset_index(
                name="late_payment_count"
            )

        )

        features = FeatureAggregator.merge(

            features,
            late,
            "SK_ID_CURR"

        )

        features = FeatureAggregator.fill_numeric(
            features
        )

        # ------------------------------------------------
        # Ratios
        # ------------------------------------------------

        denom = (
            features["installment_count"]
            .replace(0, 1)
        )

        features["late_payment_ratio"] = (

            features["late_payment_count"] /
            denom

        )

        features = FeatureAggregator.fill_numeric(
            features
        )

        print(
            f"Generated {features.shape[1]-1} installment features."
        )

        return features