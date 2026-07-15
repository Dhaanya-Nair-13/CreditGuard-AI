from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "backend" / "ml" / "models"
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "ml" / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATA = DATASET_DIR / "application_train.csv"
TEST_DATA = DATASET_DIR / "application_test.csv"

# -----------------------------
# Dataset
# -----------------------------

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# -----------------------------
# Training
# -----------------------------

TEST_SIZE = 0.2
RANDOM_STATE = 42

BUREAU_DATA = DATASET_DIR / "bureau.csv"
PREVIOUS_APPLICATION_DATA = DATASET_DIR / "previous_application.csv"
INSTALLMENTS_DATA = DATASET_DIR / "installments_payments.csv"
POS_CASH_DATA = DATASET_DIR / "POS_CASH_balance.csv"
CREDIT_CARD_DATA = DATASET_DIR / "credit_card_balance.csv"
BUREAU_BALANCE_DATA = DATASET_DIR / "bureau_balance.csv"