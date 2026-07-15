"""
Production preprocessing module.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import TARGET_COLUMN, ID_COLUMN


class CreditRiskPreprocessor:

    def __init__(self):
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []

    def fit(self, X: pd.DataFrame):

        X = X.copy()

        # Remove ID column
        if ID_COLUMN in X.columns:
            X = X.drop(columns=[ID_COLUMN])

        # Detect feature types
        self.numeric_features = (
            X.select_dtypes(include=["int64", "float64"])
            .columns
            .tolist()
        )

        self.categorical_features = (
            X.select_dtypes(include=["object", "string", "category"])
            .columns
            .tolist()
        )

        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )


            ]
        )

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    self.numeric_features
                ),
                (
                    "cat",
                    categorical_pipeline,
                    self.categorical_features
                )
            ]
        )

        self.preprocessor.fit(X)

        return self

    def transform(self, X: pd.DataFrame):

        X = X.copy()

        if ID_COLUMN in X.columns:
            X = X.drop(columns=[ID_COLUMN])

        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame):

        return self.fit(X).transform(X)