"""
previous_application_features.py

Creates customer-level features from previous_application.csv
"""

import pandas as pd

from feature_pipeline.utils import FeatureAggregator


class PreviousApplicationFeatureEngineer:

    def __init__(self, previous_application_path):

        self.previous_application_path = previous_application_path

    def transform(self):

        print("Loading previous_application.csv...")

        prev = pd.read_csv(self.previous_application_path)

        print("Generating previous application features...")

        features = pd.DataFrame({
            "SK_ID_CURR": prev["SK_ID_CURR"].unique()
        })

        # -------------------------------------------------
        # Previous application count
        # -------------------------------------------------

        count = (
            prev.groupby("SK_ID_CURR")
            .size()
            .reset_index(name="previous_application_count")
        )

        features = FeatureAggregator.merge(
            features,
            count,
            "SK_ID_CURR"
        )

        # -------------------------------------------------
        # Contract status counts
        # -------------------------------------------------

        statuses = [
            "Approved",
            "Refused",
            "Canceled",
            "Unused offer"
        ]

        for status in statuses:

            temp = (
                prev[
                    prev["NAME_CONTRACT_STATUS"] == status
                ]
                .groupby("SK_ID_CURR")
                .size()
                .reset_index(
                    name=f"previous_{status.lower().replace(' ','_')}"
                )
            )

            features = FeatureAggregator.merge(
                features,
                temp,
                "SK_ID_CURR"
            )

        features = FeatureAggregator.fill_numeric(features)

        # -------------------------------------------------
        # Approval / Refusal Ratios
        # -------------------------------------------------

        denom = (
            features["previous_application_count"]
            .replace(0, 1)
        )

        if "previous_approved" in features.columns:
            features["approval_rate"] = (
                features["previous_approved"] / denom
            )

        if "previous_refused" in features.columns:
            features["refusal_rate"] = (
                features["previous_refused"] / denom
            )

        # -------------------------------------------------
        # Numerical Statistics
        # -------------------------------------------------

        numeric_columns = [

            "AMT_APPLICATION",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "AMT_DOWN_PAYMENT",
            "CNT_PAYMENT",

            "DAYS_DECISION",
            "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE",
            "DAYS_TERMINATION"

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
            prev,
            "SK_ID_CURR",
            numeric_columns,
            stats

        )

        # -------------------------------------------------
        # Ratio Features
        # -------------------------------------------------

        prev["APPLICATION_CREDIT_RATIO"] = (
            prev["AMT_APPLICATION"] /
            prev["AMT_CREDIT"].replace(0, 1)
        )

        ratio = (
            prev.groupby("SK_ID_CURR")[
                "APPLICATION_CREDIT_RATIO"
            ]
            .mean()
            .reset_index(
                name="avg_application_credit_ratio"
            )
        )

        features = FeatureAggregator.merge(

            features,
            ratio,
            "SK_ID_CURR"

        )

        features = FeatureAggregator.fill_numeric(features)

        print(
            f"Generated {features.shape[1]-1} previous application features."
        )

        return features