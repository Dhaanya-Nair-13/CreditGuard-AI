"""
pos_cash_features.py

Creates customer-level features from POS_CASH_balance.csv
"""

import pandas as pd

from feature_pipeline.utils import FeatureAggregator


class POSCashFeatureEngineer:

    def __init__(self, pos_cash_path):

        self.pos_cash_path = pos_cash_path

    def transform(self):

        print("Loading POS_CASH_balance.csv...")

        pos = pd.read_csv(self.pos_cash_path)

        print("Generating POS CASH features...")

        # -----------------------------------------
        # Business Features
        # -----------------------------------------

        pos["LATE_PAYMENT"] = (
            pos["SK_DPD"] > 0
        ).astype(int)

        pos["FUTURE_INSTALLMENT_RATIO"] = (
            pos["CNT_INSTALMENT_FUTURE"] /
            pos["CNT_INSTALMENT"].replace(0, 1)
        )

        features = pd.DataFrame({

            "SK_ID_CURR":
                pos["SK_ID_CURR"].unique()

        })

        # -----------------------------------------
        # Record Count
        # -----------------------------------------

        count = (

            pos.groupby("SK_ID_CURR")
            .size()
            .reset_index(
                name="pos_record_count"
            )

        )

        features = FeatureAggregator.merge(
            features,
            count,
            "SK_ID_CURR"
        )

        # -----------------------------------------
        # Contract Status Counts
        # -----------------------------------------

        statuses = [
            "Active",
            "Completed",
            "Signed",
            "Demand",
            "Returned to the store",
            "Approved",
            "Amortized debt"
        ]

        for status in statuses:

            temp = (
                pos[
                    pos["NAME_CONTRACT_STATUS"] == status
                ]
                .groupby("SK_ID_CURR")
                .size()
                .reset_index(
                    name=f"pos_{status.lower().replace(' ','_')}"
                )
            )

            features = FeatureAggregator.merge(
                features,
                temp,
                "SK_ID_CURR"
            )

        # -----------------------------------------
        # Statistics
        # -----------------------------------------

        numeric_columns = [

            "MONTHS_BALANCE",

            "CNT_INSTALMENT",

            "CNT_INSTALMENT_FUTURE",

            "SK_DPD",

            "SK_DPD_DEF",

            "FUTURE_INSTALLMENT_RATIO"

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
            pos,
            "SK_ID_CURR",
            numeric_columns,
            stats

        )

        # -----------------------------------------
        # Late Count
        # -----------------------------------------

        late = (

            pos.groupby("SK_ID_CURR")[
                "LATE_PAYMENT"
            ]
            .sum()
            .reset_index(
                name="pos_late_count"
            )

        )

        features = FeatureAggregator.merge(

            features,
            late,
            "SK_ID_CURR"

        )

        features = FeatureAggregator.fill_numeric(features)

        denom = (
            features["pos_record_count"]
            .replace(0, 1)
        )

        features["pos_late_ratio"] = (

            features["pos_late_count"] /
            denom

        )

        features = FeatureAggregator.fill_numeric(features)

        print(
            f"Generated {features.shape[1]-1} POS CASH features."
        )

        return features