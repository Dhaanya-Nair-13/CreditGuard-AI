"""
installment_features_v2.py

Advanced feature engineering for installments_payments.csv
"""

import numpy as np
import pandas as pd

from feature_pipeline.utils import FeatureAggregator


class InstallmentFeatureEngineerV2:

    def __init__(self, installments_path):
        self.installments_path = installments_path

    def transform(self):

        print("Loading installments_payments.csv...")

        ins = pd.read_csv(self.installments_path)

        print("Generating Installment V2 features...")

        # -------------------------------------------------
        # Business Features
        # -------------------------------------------------

        ins["PAYMENT_DIFF"] = (
            ins["AMT_PAYMENT"] -
            ins["AMT_INSTALMENT"]
        )

        ins["PAYMENT_RATIO"] = (
            ins["AMT_PAYMENT"] /
            ins["AMT_INSTALMENT"].replace(0, np.nan)
        )

        ins["PAYMENT_RATIO"] = (
            ins["PAYMENT_RATIO"]
            .replace([np.inf, -np.inf], np.nan)
        )

        ins["LATE_DAYS"] = (
            ins["DAYS_ENTRY_PAYMENT"] -
            ins["DAYS_INSTALMENT"]
        )

        ins["EARLY_DAYS"] = (
            ins["DAYS_INSTALMENT"] -
            ins["DAYS_ENTRY_PAYMENT"]
        )

        ins["UNDER_PAYMENT"] = (
            ins["PAYMENT_DIFF"] < 0
        ).astype(int)

        ins["OVER_PAYMENT"] = (
            ins["PAYMENT_DIFF"] > 0
        ).astype(int)

        ins["EXACT_PAYMENT"] = (
            ins["PAYMENT_DIFF"] == 0
        ).astype(int)

        ins["LATE_PAYMENT"] = (
            ins["LATE_DAYS"] > 0
        ).astype(int)

        ins["VERY_LATE_PAYMENT"] = (
            ins["LATE_DAYS"] > 30
        ).astype(int)

        ins["EXTREME_LATE_PAYMENT"] = (
            ins["LATE_DAYS"] > 90
        ).astype(int)

        ins["UNDERPAY_AMOUNT"] = (
            ins["PAYMENT_DIFF"]
            .clip(upper=0)
            .abs()
        )

        ins["OVERPAY_AMOUNT"] = (
            ins["PAYMENT_DIFF"]
            .clip(lower=0)
        )

        features = pd.DataFrame({

            "SK_ID_CURR":
                ins["SK_ID_CURR"].unique()

        })

        # -------------------------------------------------
        # Record Count
        # -------------------------------------------------

        cnt = (

            ins.groupby("SK_ID_CURR")
            .size()
            .reset_index(
                name="installment_record_count"
            )

        )

        features = FeatureAggregator.merge(
            features,
            cnt,
            "SK_ID_CURR"
        )

        # -------------------------------------------------
        # Behaviour Counts
        # -------------------------------------------------

        behaviour_columns = {

            "UNDER_PAYMENT":
                "underpayment_count",

            "OVER_PAYMENT":
                "overpayment_count",

            "EXACT_PAYMENT":
                "exactpayment_count",

            "LATE_PAYMENT":
                "latepayment_count",

            "VERY_LATE_PAYMENT":
                "verylatepayment_count",

            "EXTREME_LATE_PAYMENT":
                "extremelatepayment_count",

        }

        for column, name in behaviour_columns.items():

            temp = (

                ins.groupby("SK_ID_CURR")[column]
                .sum()
                .reset_index(
                    name=name
                )

            )

            features = FeatureAggregator.merge(
                features,
                temp,
                "SK_ID_CURR"
            )

        features = FeatureAggregator.fill_numeric(features)
        # -------------------------------------------------
        # Ratios
        # -------------------------------------------------

        denom = (
            features["installment_record_count"]
            .replace(0, 1)
        )

        ratio_map = {

            "underpayment_count":
                "underpayment_ratio",

            "overpayment_count":
                "overpayment_ratio",

            "exactpayment_count":
                "exactpayment_ratio",

            "latepayment_count":
                "latepayment_ratio",

            "verylatepayment_count":
                "verylatepayment_ratio",

            "extremelatepayment_count":
                "extremelatepayment_ratio",

        }

        for count_col, ratio_col in ratio_map.items():

            features[ratio_col] = (
                features[count_col] / denom
            )

        # -------------------------------------------------
        # Statistical Aggregations
        # -------------------------------------------------

        numeric_columns = [

            "AMT_INSTALMENT",
            "AMT_PAYMENT",

            "PAYMENT_DIFF",
            "PAYMENT_RATIO",

            "UNDERPAY_AMOUNT",
            "OVERPAY_AMOUNT",

            "LATE_DAYS",
            "EARLY_DAYS",

            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",

            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",

        ]

        stats = [

            "mean",
            "sum",
            "max",
            "min",
            "std",

        ]

        features = FeatureAggregator.add_stat_features(

            features,

            ins,

            "SK_ID_CURR",

            numeric_columns,

            stats,

        )

        # -------------------------------------------------
        # Median Features
        # -------------------------------------------------

        median_columns = [

            "AMT_PAYMENT",
            "AMT_INSTALMENT",
            "PAYMENT_RATIO",
            "PAYMENT_DIFF",
            "LATE_DAYS",

        ]

        for column in median_columns:

            temp = (

                ins.groupby("SK_ID_CURR")[column]
                .median()
                .reset_index(
                    name=f"{column.lower()}_median"
                )

            )

            features = FeatureAggregator.merge(

                features,
                temp,
                "SK_ID_CURR",

            )

        # -------------------------------------------------
        # Percentiles
        # -------------------------------------------------

        percentile_columns = [

            "AMT_PAYMENT",
            "PAYMENT_RATIO",
            "LATE_DAYS",

        ]

        for column in percentile_columns:

            q90 = (

                ins.groupby("SK_ID_CURR")[column]
                .quantile(0.90)
                .reset_index(
                    name=f"{column.lower()}_q90"
                )

            )

            q95 = (

                ins.groupby("SK_ID_CURR")[column]
                .quantile(0.95)
                .reset_index(
                    name=f"{column.lower()}_q95"
                )

            )

            features = FeatureAggregator.merge(
                features,
                q90,
                "SK_ID_CURR",
            )

            features = FeatureAggregator.merge(
                features,
                q95,
                "SK_ID_CURR",
            )
        # -------------------------------------------------
        # Recency Features
        # -------------------------------------------------

        recent = ins.sort_values(
            ["SK_ID_CURR", "DAYS_INSTALMENT"],
            ascending=[True, False]
        )

        last_payment = (
            recent.groupby("SK_ID_CURR")
            .first()
            .reset_index()
        )

        recent_cols = {
            "AMT_PAYMENT": "last_payment_amount",
            "AMT_INSTALMENT": "last_instalment_amount",
            "PAYMENT_RATIO": "last_payment_ratio",
            "PAYMENT_DIFF": "last_payment_diff",
            "LATE_DAYS": "last_late_days",
            "EARLY_DAYS": "last_early_days",
            "DAYS_ENTRY_PAYMENT": "last_entry_day",
            "DAYS_INSTALMENT": "last_due_day",
        }

        for original, new_name in recent_cols.items():

            temp = last_payment[
                ["SK_ID_CURR", original]
            ].rename(
                columns={
                    original: new_name
                }
            )

            features = FeatureAggregator.merge(
                features,
                temp,
                "SK_ID_CURR",
            )

        # -------------------------------------------------
        # Last 3 Payments
        # -------------------------------------------------

        last3 = (
            recent.groupby("SK_ID_CURR")
            .head(3)
        )

        last3_stats = (

            last3.groupby("SK_ID_CURR")
            .agg(
                last3_payment_mean=("AMT_PAYMENT", "mean"),
                last3_ratio_mean=("PAYMENT_RATIO", "mean"),
                last3_late_mean=("LATE_DAYS", "mean"),
                last3_underpay=("UNDERPAY_AMOUNT", "sum"),
                last3_overpay=("OVERPAY_AMOUNT", "sum"),
            )
            .reset_index()

        )

        features = FeatureAggregator.merge(
            features,
            last3_stats,
            "SK_ID_CURR",
        )

        # -------------------------------------------------
        # Last 5 Payments
        # -------------------------------------------------

        last5 = (
            recent.groupby("SK_ID_CURR")
            .head(5)
        )

        last5_stats = (

            last5.groupby("SK_ID_CURR")
            .agg(
                last5_payment_mean=("AMT_PAYMENT", "mean"),
                last5_ratio_mean=("PAYMENT_RATIO", "mean"),
                last5_late_mean=("LATE_DAYS", "mean"),
                last5_underpay=("UNDERPAY_AMOUNT", "sum"),
                last5_overpay=("OVERPAY_AMOUNT", "sum"),
            )
            .reset_index()

        )

        features = FeatureAggregator.merge(
            features,
            last5_stats,
            "SK_ID_CURR",
        )

        # -------------------------------------------------
        # Volatility
        # -------------------------------------------------

        volatility = (

            ins.groupby("SK_ID_CURR")
            .agg(

                payment_cv=(
                    "AMT_PAYMENT",
                    lambda x:
                    x.std() / x.mean()
                    if x.mean() != 0 else 0
                ),

                instalment_cv=(
                    "AMT_INSTALMENT",
                    lambda x:
                    x.std() / x.mean()
                    if x.mean() != 0 else 0
                ),

                ratio_cv=(
                    "PAYMENT_RATIO",
                    lambda x:
                    x.std() / x.mean()
                    if x.mean() != 0 else 0
                ),

                late_cv=(
                    "LATE_DAYS",
                    lambda x:
                    x.std() / (abs(x.mean()) + 1)
                ),

            )
            .reset_index()

        )

        features = FeatureAggregator.merge(
            features,
            volatility,
            "SK_ID_CURR",
        )
                # -------------------------------------------------
        # Trend Features
        # -------------------------------------------------

        recent_sorted = ins.sort_values(
            ["SK_ID_CURR", "DAYS_INSTALMENT"],
            ascending=[True, False]
        )

        recent_half = (
            recent_sorted.groupby("SK_ID_CURR")
            .head(5)
        )

        old_half = (
            recent_sorted.groupby("SK_ID_CURR")
            .tail(5)
        )

        recent_stats = (

            recent_half.groupby("SK_ID_CURR")
            .agg(

                recent_payment_mean=("AMT_PAYMENT", "mean"),

                recent_ratio_mean=("PAYMENT_RATIO", "mean"),

                recent_late_mean=("LATE_DAYS", "mean"),

                recent_underpay=("UNDERPAY_AMOUNT", "sum"),

                recent_overpay=("OVERPAY_AMOUNT", "sum"),

            )
            .reset_index()

        )

        old_stats = (

            old_half.groupby("SK_ID_CURR")
            .agg(

                old_payment_mean=("AMT_PAYMENT", "mean"),

                old_ratio_mean=("PAYMENT_RATIO", "mean"),

                old_late_mean=("LATE_DAYS", "mean"),

                old_underpay=("UNDERPAY_AMOUNT", "sum"),

                old_overpay=("OVERPAY_AMOUNT", "sum"),

            )
            .reset_index()

        )

        features = FeatureAggregator.merge(
            features,
            recent_stats,
            "SK_ID_CURR",
        )

        features = FeatureAggregator.merge(
            features,
            old_stats,
            "SK_ID_CURR",
        )

        # -------------------------------------------------
        # Trend Differences
        # -------------------------------------------------

        features["payment_trend"] = (
            features["recent_payment_mean"] -
            features["old_payment_mean"]
        )

        features["payment_ratio_trend"] = (
            features["recent_ratio_mean"] -
            features["old_ratio_mean"]
        )

        features["late_trend"] = (
            features["recent_late_mean"] -
            features["old_late_mean"]
        )

        features["underpay_trend"] = (
            features["recent_underpay"] -
            features["old_underpay"]
        )

        features["overpay_trend"] = (
            features["recent_overpay"] -
            features["old_overpay"]
        )

        # -------------------------------------------------
        # Interaction Features
        # -------------------------------------------------

        features["late_x_underpay"] = (

            features["latepayment_ratio"] *
            features["underpayment_ratio"]

        )

        features["late_x_payment_ratio"] = (

            features["latepayment_ratio"] *
            features["payment_ratio_mean"]

        )

        features["underpay_x_payment_ratio"] = (

            features["underpayment_ratio"] *
            features["payment_ratio_mean"]

        )

        features["overpay_x_payment_ratio"] = (

            features["overpayment_ratio"] *
            features["payment_ratio_mean"]

        )

        features["late_x_payment_cv"] = (

            features["latepayment_ratio"] *
            features["payment_cv"]

        )

        features["late_x_ratio_cv"] = (

            features["latepayment_ratio"] *
            features["ratio_cv"]

        )

        features["late_x_amount"] = (

            features["latepayment_ratio"] *
            features["amt_payment_mean"]

        )

        features["underpay_x_amount"] = (

            features["underpayment_ratio"] *
            features["amt_payment_mean"]

        )

        features["payment_consistency"] = (

            1 /
            (1 + features["payment_cv"])

        )

        features["payment_reliability"] = (

            features["payment_ratio_mean"] *
            features["payment_consistency"]

        )
                # -------------------------------------------------
        # Extreme Event Features
        # -------------------------------------------------

        extreme = (

            ins.groupby("SK_ID_CURR")
            .agg(

                max_underpay=("UNDERPAY_AMOUNT", "max"),

                max_overpay=("OVERPAY_AMOUNT", "max"),

                max_late_days=("LATE_DAYS", "max"),

                min_late_days=("LATE_DAYS", "min"),

                max_payment=("AMT_PAYMENT", "max"),

                min_payment=("AMT_PAYMENT", "min"),

                max_payment_ratio=("PAYMENT_RATIO", "max"),

                min_payment_ratio=("PAYMENT_RATIO", "min"),

            )
            .reset_index()

        )

        features = FeatureAggregator.merge(
            features,
            extreme,
            "SK_ID_CURR",
        )

        # -------------------------------------------------
        # Skewness
        # -------------------------------------------------

        skew = (

            ins.groupby("SK_ID_CURR")
            .agg(

                payment_skew=("AMT_PAYMENT", "skew"),

                ratio_skew=("PAYMENT_RATIO", "skew"),

                late_skew=("LATE_DAYS", "skew"),

            )
            .reset_index()

        )

        features = FeatureAggregator.merge(
            features,
            skew,
            "SK_ID_CURR",
        )

        # -------------------------------------------------
        # Recent vs Overall
        # -------------------------------------------------

        features["recent_vs_overall_payment"] = (

            features["last3_payment_mean"] /
            features["amt_payment_mean"].replace(0, 1)

        )

        features["recent_vs_overall_ratio"] = (

            features["last3_ratio_mean"] /
            features["payment_ratio_mean"].replace(0, 1)

        )

        features["recent_vs_overall_late"] = (

            features["last3_late_mean"] -
            features["late_days_mean"]

        )

        # -------------------------------------------------
        # Stability
        # -------------------------------------------------

        features["stable_customer"] = (

            (features["payment_cv"] < 0.20).astype(int)

        )

        features["chronic_late_customer"] = (

            (features["latepayment_ratio"] > 0.50).astype(int)

        )

        features["chronic_underpayer"] = (

            (features["underpayment_ratio"] > 0.50).astype(int)

        )

        features["perfect_customer"] = (

            (
                (features["latepayment_ratio"] == 0)
                &
                (features["underpayment_ratio"] == 0)
            ).astype(int)

        )

        # -------------------------------------------------
        # Missing Indicators
        # -------------------------------------------------

        important = [

            "payment_ratio_mean",

            "amt_payment_mean",

            "late_days_mean",

            "payment_cv",

        ]

        for col in important:

            features[f"{col}_missing"] = (

                features[col]
                .isna()
                .astype(int)

            )

        features = FeatureAggregator.fill_numeric(
            features
        )
                # -------------------------------------------------
        # Remove duplicate customer rows (safety)
        # -------------------------------------------------

        features = features.drop_duplicates(
            subset=["SK_ID_CURR"]
        )

        # -------------------------------------------------
        # Replace invalid numeric values
        # -------------------------------------------------

        numeric_cols = features.select_dtypes(
            include=["number"]
        ).columns

        features[numeric_cols] = (
            features[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
        )

        features = FeatureAggregator.fill_numeric(
            features
        )

        print(
            f"Generated {features.shape[1]-1} Installment V2 features."
        )

        return features