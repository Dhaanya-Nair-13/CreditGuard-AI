"""
credit_card_features.py

Creates customer-level features from credit_card_balance.csv
"""

import pandas as pd

from feature_pipeline.utils import FeatureAggregator


class CreditCardFeatureEngineer:

    def __init__(self, credit_card_path):

        self.credit_card_path = credit_card_path

    def transform(self):

        print("Loading credit_card_balance.csv...")

        cc = pd.read_csv(self.credit_card_path)

        print("Generating credit card features...")

        # ------------------------------------------------
        # Business Features
        # ------------------------------------------------

        cc["CREDIT_UTILIZATION"] = (
            cc["AMT_BALANCE"] /
            cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, 1)
        )

        cc["PAYMENT_RATIO"] = (
            cc["AMT_PAYMENT_TOTAL_CURRENT"] /
            cc["AMT_TOTAL_RECEIVABLE"].replace(0, 1)
        )

        cc["DRAWING_RATIO"] = (
            cc["AMT_DRAWINGS_CURRENT"] /
            cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, 1)
        )

        cc["LATE_PAYMENT"] = (
            cc["SK_DPD"] > 0
        ).astype(int)

        features = pd.DataFrame({

            "SK_ID_CURR":
                cc["SK_ID_CURR"].unique()

        })

        # ------------------------------------------------
        # Credit Card Count
        # ------------------------------------------------

        count = (

            cc.groupby("SK_ID_CURR")
            .size()
            .reset_index(
                name="credit_card_records"
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

            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",

            "AMT_DRAWINGS_CURRENT",
            "AMT_DRAWINGS_ATM_CURRENT",
            "AMT_DRAWINGS_POS_CURRENT",

            "AMT_PAYMENT_CURRENT",
            "AMT_PAYMENT_TOTAL_CURRENT",

            "AMT_TOTAL_RECEIVABLE",

            "CNT_DRAWINGS_CURRENT",
            "CNT_INSTALMENT_MATURE_CUM",

            "SK_DPD",
            "SK_DPD_DEF",

            "CREDIT_UTILIZATION",
            "PAYMENT_RATIO",
            "DRAWING_RATIO"

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
            cc,
            "SK_ID_CURR",
            numeric_columns,
            stats

        )

        # ------------------------------------------------
        # Late Payment Count
        # ------------------------------------------------

        late = (

            cc.groupby("SK_ID_CURR")[
                "LATE_PAYMENT"
            ]
            .sum()
            .reset_index(
                name="credit_card_late_count"
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

        denom = (
            features["credit_card_records"]
            .replace(0, 1)
        )

        features["credit_card_late_ratio"] = (

            features["credit_card_late_count"] /
            denom

        )

        features = FeatureAggregator.fill_numeric(
            features
        )

        print(
            f"Generated {features.shape[1]-1} credit card features."
        )

        return features