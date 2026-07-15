"""
feature_importance_analysis.py

Analyzes feature importance of the trained XGBoost model.
"""

import pandas as pd
import joblib

from config import (
    MODEL_DIR,
    ARTIFACT_DIR,
)

print("=" * 60)
print("CreditGuard-AI")
print("Feature Importance Analysis")
print("=" * 60)

print("\nLoading model...")

model = joblib.load(
    MODEL_DIR / "xgboost_enriched.pkl"
)

preprocessor = joblib.load(
    MODEL_DIR / "preprocessor_enriched.pkl"
)

print("Done.")

print("\nExtracting feature names...")

feature_names = preprocessor.get_feature_names()

print(f"Features : {len(feature_names)}")

print("\nExtracting feature importance...")

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False,

).reset_index(drop=True)

importance_df["Rank"] = (

    importance_df.index + 1

)

importance_df.to_csv(

    ARTIFACT_DIR / "feature_importance.csv",

    index=False,

)

print("\nTop 20 Features\n")

print(

    importance_df.head(20)

)

print("\n")

zero_features = (

    importance_df["Importance"] == 0

).sum()

print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total Features        : {len(importance_df)}")
print(f"Zero Importance       : {zero_features}")
print(f"Non-zero Importance   : {len(importance_df)-zero_features}")

print("\nFeature importance saved to:")

print(

    ARTIFACT_DIR /
    "feature_importance.csv"

)

print("\nDone.")