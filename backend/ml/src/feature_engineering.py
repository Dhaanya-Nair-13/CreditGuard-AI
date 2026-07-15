"""
Feature Engineering Module
"""

import numpy as np
import pandas as pd


class FeatureEngineer:

    def transform(self, df: pd.DataFrame):

        df = df.copy()

        # ----------------------------
        # Basic Financial Ratios
        # ----------------------------

        df["CREDIT_INCOME_RATIO"] = (
            df["AMT_CREDIT"] /
            (df["AMT_INCOME_TOTAL"] + 1)
        )

        df["ANNUITY_INCOME_RATIO"] = (
            df["AMT_ANNUITY"] /
            (df["AMT_INCOME_TOTAL"] + 1)
        )

        df["CREDIT_ANNUITY_RATIO"] = (
            df["AMT_CREDIT"] /
            (df["AMT_ANNUITY"] + 1)
        )

        # ----------------------------
        # Employment
        # ----------------------------

        df["AGE_YEARS"] = (
            -df["DAYS_BIRTH"] / 365
        )

        df["EMPLOYMENT_RATIO"] = (
            df["DAYS_EMPLOYED"] /
            (df["DAYS_BIRTH"] + 1)
        )

        # ----------------------------
        # Family
        # ----------------------------

        df["INCOME_PER_FAMILY"] = (
            df["AMT_INCOME_TOTAL"] /
            (df["CNT_FAM_MEMBERS"] + 1)
        )

        # ----------------------------
        # External Scores
        # ----------------------------

        ext_cols = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ]

        df["EXT_SOURCE_MEAN"] = (
            df[ext_cols].mean(axis=1)
        )

        df["EXT_SOURCE_STD"] = (
            df[ext_cols].std(axis=1)
        )

        # ----------------------------
        # Replace infinite values
        # ----------------------------

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

        # -------------------------------------------------
        # Income / Credit Features
        # -------------------------------------------------

        df["CREDIT_PER_PERSON"] = (
        df["AMT_CREDIT"] /
        (df["CNT_FAM_MEMBERS"] + 1)
        )

        df["ANNUITY_PER_PERSON"] = (
        df["AMT_ANNUITY"] /
        (df["CNT_FAM_MEMBERS"] + 1)
        )

        df["GOODS_INCOME_RATIO"] = (
        df["AMT_GOODS_PRICE"] /
        (df["AMT_INCOME_TOTAL"] + 1)
        )
        # -------------------------------------------------
        # # Employment
        # # -------------------------------------------------

        df["EMPLOYMENT_YEARS"] = (
            -df["DAYS_EMPLOYED"] / 365
        )

        df["CAR_AGE_RATIO"] = (
            df["OWN_CAR_AGE"] /
            (df["AGE_YEARS"] + 1)
        )

        # -------------------------------------------------
        # # Family
        # # -------------------------------------------------

        df["CHILDREN_RATIO"] = (
            df["CNT_CHILDREN"] /
            (df["CNT_FAM_MEMBERS"] + 1)
        )
        # -------------------------------------------------
        # # External Sources
        # # -------------------------------------------------

        df["EXT_SOURCE_MAX"] = df[
            [
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
            ]
        ].max(axis=1)

        df["EXT_SOURCE_MIN"] = df[
            [
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
            ]
        ].min(axis=1)

        # -------------------------------------------------
        # # Missing Values
        # # -------------------------------------------------
        df["MISSING_COUNT"] = df.isnull().sum(axis=1)

        return df