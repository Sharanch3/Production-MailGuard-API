from typing import Dict, Literal

from pydantic import BaseModel, Field


class EmailInput(BaseModel):
    text: str = Field(..., min_length=3, description="Email text to classify")


class PredictionOutput(BaseModel):
    prediction: Literal["Spam", "Not Spam"]
    confidence: float = Field(description="Confidence score from 0-100")
    probabilities: Dict[str, float] = Field(description="Probability for each class")
    cleaned_text: str = Field(description="Processed email text")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    nlp_loaded: bool
