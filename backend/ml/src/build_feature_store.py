"""
build_feature_store.py

Creates the offline feature store used by
FastAPI and Streamlit.
"""

import pandas as pd

from config import (
    ARTIFACT_DIR,
    TARGET_COLUMN,
    TRAIN_DATA,
    BUREAU_DATA,
    BUREAU_BALANCE_DATA,
    PREVIOUS_APPLICATION_DATA,
    INSTALLMENTS_DATA,
    CREDIT_CARD_DATA,
    POS_CASH_DATA,
)

from feature_pipeline.merge_features import FeatureMerger
from feature_engineering import FeatureEngineer


def main():

    print("=" * 60)
    print("Building Offline Feature Store")
    print("=" * 60)

    #####################################################
    # Merge Features
    #####################################################

    df = FeatureMerger(
        TRAIN_DATA,
        BUREAU_DATA,
        BUREAU_BALANCE_DATA,
        PREVIOUS_APPLICATION_DATA,
        INSTALLMENTS_DATA,
        CREDIT_CARD_DATA,
        POS_CASH_DATA,
    ).transform()

    #####################################################
    # Feature Engineering
    #####################################################

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    X = FeatureEngineer().transform(X)

    feature_store = X.copy()

    feature_store[TARGET_COLUMN] = y

    #####################################################
    # Save
    #####################################################

    output_path = ARTIFACT_DIR / "demo_features.parquet"

    feature_store.to_parquet(
        output_path,
        index=False,
    )

    print("\nSaved Feature Store")

    print(output_path)

    print("\nShape:")

    print(feature_store.shape)


if __name__ == "__main__":
    main()