"""
utils.py

Common utilities for feature engineering.
"""

import pandas as pd


class FeatureAggregator:

    @staticmethod
    def merge(base_df, feature_df, on_column):
        """
        Merge two dataframes.
        """
        return base_df.merge(
            feature_df,
            on=on_column,
            how="left"
        )

    @staticmethod
    def fill_numeric(df):
        """
        Fill only numeric columns with 0.
        """
        numeric_cols = df.select_dtypes(include="number").columns

        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    @staticmethod
    def add_stat_features(
        base_df,
        source_df,
        group_column,
        value_columns,
        statistics,
    ):
        """
        Automatically create aggregation features.

        Example:
            mean
            sum
            max
            min
            std
        """

        for column in value_columns:

            if column not in source_df.columns:
                continue

            grouped = (
                source_df
                .groupby(group_column)[column]
                .agg(statistics)
                .reset_index()
            )

            rename_dict = {}

            for stat in statistics:
                rename_dict[stat] = f"{column.lower()}_{stat}"

            grouped.rename(
                columns=rename_dict,
                inplace=True,
            )

            base_df = base_df.merge(
                grouped,
                on=group_column,
                how="left",
            )

        return base_df

    @staticmethod
    def count(
        df,
        group_column,
        feature_name,
    ):

        return (
            df.groupby(group_column)
            .size()
            .reset_index(name=feature_name)
        )

    @staticmethod
    def nunique(
        df,
        group_column,
        value_column,
        feature_name,
    ):

        return (
            df.groupby(group_column)[value_column]
            .nunique()
            .reset_index(name=feature_name)
        )