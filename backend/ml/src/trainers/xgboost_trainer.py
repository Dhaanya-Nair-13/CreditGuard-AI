"""
xgboost_trainer.py

Reusable XGBoost trainer.
"""

from xgboost import XGBClassifier

from config import RANDOM_STATE


class XGBoostTrainer:

    def __init__(self, params=None):

        if params is None:

            params = {

                "objective": "binary:logistic",

                "eval_metric": "auc",

                "tree_method": "hist",

                "random_state": RANDOM_STATE,

                "n_jobs": -1,

                "max_depth": 5,

                "learning_rate": 0.04144396109308082,

                "n_estimators": 771,

                "subsample": 0.7487250650014305,

                "colsample_bytree": 0.9069947065143011,

                "gamma": 0.8786947680818691,

                "min_child_weight": 6,

                "reg_alpha": 2.020921196164734,

                "reg_lambda": 2.488598705699846,

                "scale_pos_weight": 10.89343686455616,

                "early_stopping_rounds": 50,

            }

        self.model = XGBClassifier(
            **params
        )

    def fit(

        self,

        X_train,

        y_train,

        X_valid=None,

        y_valid=None,

    ):

        if X_valid is None:

            self.model.fit(
                X_train,
                y_train,
            )

        else:

            self.model.fit(

                X_train,

                y_train,

                eval_set=[
                    (X_valid, y_valid)
                ],

                verbose=False,

            )

        return self

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def save(self, path):

        import joblib

        joblib.dump(
            self.model,
            path,
        )