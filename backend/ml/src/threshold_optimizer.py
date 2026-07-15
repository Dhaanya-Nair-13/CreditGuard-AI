"""
Threshold Optimization
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


class ThresholdOptimizer:

    def find_best_threshold(self, y_true, probabilities):

        thresholds = np.arange(0.05, 0.96, 0.05)

        results = []

        best_threshold = 0.50
        best_f1 = 0

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_true,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_true,
                predictions
            )

            f1 = f1_score(
                y_true,
                predictions
            )

            results.append([
                threshold,
                precision,
                recall,
                f1
            ])

            if f1 > best_f1:

                best_f1 = f1
                best_threshold = threshold

        results = pd.DataFrame(
            results,
            columns=[
                "Threshold",
                "Precision",
                "Recall",
                "F1"
            ]
        )

        return best_threshold, results