"""
bureau_balance_features.py

Creates engineered features from bureau_balance.csv.
Each row returned represents ONE SK_ID_BUREAU.
"""

import pandas as pd

from feature_pipeline.utils import FeatureAggregator


class BureauBalanceFeatureEngineer:

    def __init__(self, bureau_balance_path):
        self.bureau_balance_path = bureau_balance_path

    def transform(self):

        print("Loading bureau_balance.csv...")

        bb = pd.read_csv(self.bureau_balance_path)

        print("Generating bureau balance features...")

        # -----------------------------
        # Base dataframe
        # -----------------------------
        features = pd.DataFrame({
            "SK_ID_BUREAU": bb["SK_ID_BUREAU"].unique()
        })

        # -----------------------------
        # Number of monthly records
        # -----------------------------
        month_count = (
            bb.groupby("SK_ID_BUREAU")
            .size()
            .reset_index(name="bb_month_count")
        )

        features = FeatureAggregator.merge(
            features,
            month_count,
            "SK_ID_BUREAU"
        )

        # -----------------------------
        # MONTHS_BALANCE statistics
        # -----------------------------
        month_stats = (
            bb.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"]
            .agg(
                bb_month_min="min",
                bb_month_max="max",
                bb_month_mean="mean",
                bb_month_std="std"
            )
            .reset_index()
        )

        features = FeatureAggregator.merge(
            features,
            month_stats,
            "SK_ID_BUREAU"
        )

        # -----------------------------
        # STATUS COUNTS
        # -----------------------------
        statuses = ["0", "1", "2", "3", "4", "5", "C", "X"]

        for status in statuses:

            temp = (
                bb[bb["STATUS"] == status]
                .groupby("SK_ID_BUREAU")
                .size()
                .reset_index(name=f"bb_status_{status}")
            )

            features = FeatureAggregator.merge(
                features,
                temp,
                "SK_ID_BUREAU"
            )

        # -----------------------------
        # Fill numeric NA
        # -----------------------------
        features = FeatureAggregator.fill_numeric(features)

        # -----------------------------
        # Ratio Features
        # -----------------------------
        denom = features["bb_month_count"].replace(0, 1)

        late_cols = [
            "bb_status_1",
            "bb_status_2",
            "bb_status_3",
            "bb_status_4",
            "bb_status_5",
        ]

        existing = [
            c for c in late_cols
            if c in features.columns
        ]

        features["bb_total_late"] = features[existing].sum(axis=1)

        features["bb_late_ratio"] = (
            features["bb_total_late"] / denom
        )

        if "bb_status_C" in features.columns:
            features["bb_closed_ratio"] = (
                features["bb_status_C"] / denom
            )

        if "bb_status_X" in features.columns:
            features["bb_unknown_ratio"] = (
                features["bb_status_X"] / denom
            )

        features = FeatureAggregator.fill_numeric(features)

        print(
            f"Generated {features.shape[1]-1} bureau balance features."
        )

        return features