"""
bureau_features.py

Creates customer-level bureau features from:
1. bureau.csv
2. bureau_balance engineered features

Returns one row per SK_ID_CURR.
"""

import pandas as pd

from feature_pipeline.bureau_balance_features import BureauBalanceFeatureEngineer
from feature_pipeline.utils import FeatureAggregator


class BureauFeatureEngineer:

    def __init__(self, bureau_path, bureau_balance_path):

        self.bureau_path = bureau_path
        self.bureau_balance_path = bureau_balance_path

    def transform(self):

        print("Loading bureau.csv...")

        bureau = pd.read_csv(self.bureau_path)

        # -----------------------------
        # Merge bureau balance features
        # -----------------------------

        bb_features = BureauBalanceFeatureEngineer(
            self.bureau_balance_path
        ).transform()

        print("Merging bureau balance features...")

        bureau = bureau.merge(
            bb_features,
            on="SK_ID_BUREAU",
            how="left"
        )

        # -----------------------------
        # Customer dataframe
        # -----------------------------

        features = pd.DataFrame({
            "SK_ID_CURR": bureau["SK_ID_CURR"].unique()
        })

        # -----------------------------
        # Loan Counts
        # -----------------------------

        loan_count = (
            bureau.groupby("SK_ID_CURR")
            .size()
            .reset_index(name="bureau_loan_count")
        )

        active_count = (
            bureau[bureau["CREDIT_ACTIVE"] == "Active"]
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="bureau_active_loans")
        )

        closed_count = (
            bureau[bureau["CREDIT_ACTIVE"] == "Closed"]
            .groupby("SK_ID_CURR")
            .size()
            .reset_index(name="bureau_closed_loans")
        )

        features = FeatureAggregator.merge(
            features,
            loan_count,
            "SK_ID_CURR"
        )

        features = FeatureAggregator.merge(
            features,
            active_count,
            "SK_ID_CURR"
        )

        features = FeatureAggregator.merge(
            features,
            closed_count,
            "SK_ID_CURR"
        )

        # -----------------------------
        # Numeric columns
        # -----------------------------

        numeric_columns = [

            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT",
            "AMT_CREDIT_SUM_OVERDUE",

            "DAYS_CREDIT",
            "DAYS_CREDIT_ENDDATE",
            "DAYS_ENDDATE_FACT",
            "CREDIT_DAY_OVERDUE",

            "bb_month_count",
            "bb_total_late",
            "bb_late_ratio",
            "bb_closed_ratio"

        ]

        stats = ["mean", "sum", "max", "min", "std"]

        features = FeatureAggregator.add_stat_features(
            features,
            bureau,
            "SK_ID_CURR",
            numeric_columns,
            stats
        )

        features = FeatureAggregator.fill_numeric(features)

        # -----------------------------
        # Ratios
        # -----------------------------

        denom = features["bureau_loan_count"].replace(0, 1)

        features["bureau_active_ratio"] = (
            features["bureau_active_loans"] / denom
        )

        if (
            "amt_credit_sum_debt_sum" in features.columns
            and
            "amt_credit_sum_sum" in features.columns
        ):

            credit = (
                features["amt_credit_sum_sum"]
                .replace(0, 1)
            )

            features["bureau_debt_credit_ratio"] = (
                features["amt_credit_sum_debt_sum"] /
                credit
            )

        features = FeatureAggregator.fill_numeric(features)

        print(
            f"Generated {features.shape[1]-1} bureau features."
        )

        return features