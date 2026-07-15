"""
merge_features.py

Merge all engineered features into the application dataset.
"""

import pandas as pd

from feature_pipeline.utils import FeatureAggregator
from feature_pipeline.bureau_features import BureauFeatureEngineer
from feature_pipeline.previous_application_features import (
    PreviousApplicationFeatureEngineer,
)
from feature_pipeline.installment_features_v2 import (
    InstallmentFeatureEngineerV2,
)
from feature_pipeline.credit_card_features import CreditCardFeatureEngineer
from feature_pipeline.pos_cash_features import POSCashFeatureEngineer



class FeatureMerger:

    def __init__(
        self,
        application_path,
        bureau_path,
        bureau_balance_path,
        previous_application_path,
        installments_path,
        credit_card_path,
        pos_cash_path,
    ):

        self.application_path = application_path
        self.bureau_path = bureau_path
        self.bureau_balance_path = bureau_balance_path
        self.previous_application_path = previous_application_path
        self.installments_path = installments_path
        self.credit_card_path = credit_card_path
        self.pos_cash_path = pos_cash_path

    def transform(self):

        print("=" * 60)
        print("Loading application dataset...")
        print("=" * 60)

        application = pd.read_csv(self.application_path)

        # ==================================================
        # Bureau Features
        # ==================================================

        bureau_features = BureauFeatureEngineer(
            self.bureau_path,
            self.bureau_balance_path,
        ).transform()

        print("\nMerging bureau features...")

        application = application.merge(
            bureau_features,
            on="SK_ID_CURR",
            how="left",
        )

        # ==================================================
        # Previous Application Features
        # ==================================================

        previous_features = PreviousApplicationFeatureEngineer(
            self.previous_application_path
        ).transform()

        print("\nMerging previous application features...")

        application = application.merge(
            previous_features,
            on="SK_ID_CURR",
            how="left",
        )

        

        

        
    
        # ==================================================
        # Installment Features
        # ==================================================


        installment_features = InstallmentFeatureEngineerV2(
            self.installments_path
        ).transform()
        
        print("\nMerging installment features...")
        application = application.merge(
            installment_features,
            on="SK_ID_CURR",
            how="left",
        )

        # ==================================================
        # # Credit Card Features
        # # ==================================================
        
        credit_card_features = CreditCardFeatureEngineer(
            self.credit_card_path
        ).transform()
        print("\nMerging credit card features...")
        
        application = application.merge(
            credit_card_features,
            on="SK_ID_CURR",
            how="left",
        )

        # ==================================================
        # POS CASH Features
        # # ==================================================
        pos_cash_features = POSCashFeatureEngineer(
            self.pos_cash_path
        ).transform()
        
        print("\nMerging POS CASH features...")
        
        application = application.merge(
            pos_cash_features,
            on="SK_ID_CURR",
            how="left",
        )

        application = FeatureAggregator.fill_numeric(application)
        print("\nMerge Complete!")

        print("Final Dataset Shape:", application.shape)
        return application