from pathlib import Path
import sys

from fastapi import APIRouter, HTTPException

# Make backend/ml/src importable exactly like when you ran the scripts
ML_SRC = Path(__file__).resolve().parents[1] / "ml" / "src"

if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from prediction_pipeline import PredictionPipeline

router = APIRouter()

pipeline = PredictionPipeline()


@router.get("/")
def home():
    return {
        "message": "CreditGuard-AI API is running."
    }


@router.get("/customers/{customer_id}")
def predict_customer(customer_id: int):
    try:
        return pipeline.predict_with_explanation(customer_id)

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )