from pydantic import BaseModel


class PredictionResponse(BaseModel):

    customer_id: int

    probability: float

    risk: str


class Explanation(BaseModel):

    feature: str

    impact: float

    direction: str


class CustomerResponse(PredictionResponse):

    top_factors: list[Explanation]