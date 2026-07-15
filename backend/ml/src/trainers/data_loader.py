"""
data_loader.py

Loads and preprocesses the complete engineered dataset.
Shared by every ML model.
"""

from sklearn.model_selection import train_test_split

from config import (
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
    TRAIN_DATA,
    BUREAU_DATA,
    BUREAU_BALANCE_DATA,
    PREVIOUS_APPLICATION_DATA,
    INSTALLMENTS_DATA,
    CREDIT_CARD_DATA,
    POS_CASH_DATA,
)

from preprocessing import CreditRiskPreprocessor
from feature_engineering import FeatureEngineer
from feature_pipeline.merge_features import FeatureMerger


class DataLoader:

    def load_data(self):

        print("=" * 60)
        print("Loading Engineered Dataset")
        print("=" * 60)

        df = FeatureMerger(
            TRAIN_DATA,
            BUREAU_DATA,
            BUREAU_BALANCE_DATA,
            PREVIOUS_APPLICATION_DATA,
            INSTALLMENTS_DATA,
            CREDIT_CARD_DATA,
            POS_CASH_DATA,
        ).transform()

        print("\nDataset Loaded.")
        print(df.shape)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        print("\nApplying Feature Engineering...")

        X = FeatureEngineer().transform(X)

        print("\nSplitting Dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        print("\nPreprocessing...")

        preprocessor = CreditRiskPreprocessor()

        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
        )