"""
Model evaluation utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class ModelEvaluator:

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir

    def save_confusion_matrix(self, y_true, predictions):

        cm = confusion_matrix(y_true, predictions)

        disp = ConfusionMatrixDisplay(cm)

        disp.plot()

        plt.tight_layout()

        plt.savefig(
            self.artifact_dir / "confusion_matrix.png",
            dpi=300
        )

        plt.close()

    def save_roc_curve(self, y_true, probabilities):

        fpr, tpr, _ = roc_curve(
            y_true,
            probabilities
        )

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6,6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.3f}"
        )

        plt.plot([0,1],[0,1],"--")

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            self.artifact_dir / "roc_curve.png",
            dpi=300
        )

        plt.close()

    def save_precision_recall_curve(
        self,
        y_true,
        probabilities
    ):

        precision, recall, _ = precision_recall_curve(
            y_true,
            probabilities
        )

        plt.figure(figsize=(6,6))

        plt.plot(
            recall,
            precision
        )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision Recall Curve")

        plt.tight_layout()

        plt.savefig(
            self.artifact_dir / "precision_recall_curve.png",
            dpi=300
        )

        plt.close()