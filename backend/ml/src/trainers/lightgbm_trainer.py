"""
lightgbm_trainer.py

Reusable LightGBM trainer.
"""

from lightgbm import LGBMClassifier

from config import RANDOM_STATE


class LightGBMTrainer:

    def __init__(self, params=None):

        if params is None:

            params = {

                "objective": "binary",

                "metric": "auc",

                "boosting_type": "gbdt",

                "random_state": RANDOM_STATE,

                "n_jobs": -1,

                "learning_rate": 0.05,

                "n_estimators": 800,

                "max_depth": 6,

                "num_leaves": 31,

                "subsample": 0.8,

                "colsample_bytree": 0.8,

                "reg_alpha": 0.0,

                "reg_lambda": 2.0,

                "min_child_samples": 20,

                "scale_pos_weight": 10.9,

                "verbosity": -1,

            }

        self.model = LGBMClassifier(
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

                eval_metric="auc",

                callbacks=[],

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