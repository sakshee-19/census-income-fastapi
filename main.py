"""FastAPI application for Census income classification."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load_model_artifacts
from starter.train_model import CATEGORICAL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"

model, encoder, label_binarizer, metadata = load_model_artifacts(
    MODEL_DIR
)

app = FastAPI(
    title="Census Income Classification API",
    description=(
        "Predict whether an individual's annual income exceeds $50K "
        "using Census demographic and employment features."
    ),
    version="1.0.0",
)


class CensusInput(BaseModel):
    """Validate Census features submitted for income prediction."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )

    age: Annotated[int, Field(ge=17, le=100)]
    workclass: str
    fnlgt: Annotated[int, Field(gt=0)]
    education: str
    education_num: Annotated[int, Field(alias="education-num", ge=1, le=16)]
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: Annotated[int, Field(alias="capital-gain", ge=0)]
    capital_loss: Annotated[int, Field(alias="capital-loss", ge=0)]
    hours_per_week: Annotated[
        int, Field(alias="hours-per-week", ge=1, le=99)
    ]
    native_country: str = Field(alias="native-country")


class PredictionResponse(BaseModel):
    """Structure the prediction response."""

    prediction: str
    prediction_label: str


@app.get("/")
async def root():
    """Return API welcome information."""
    return {
        "message": "Welcome to the Census Income Classification API!",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: CensusInput):
    """Predict the income class for one Census record."""
    try:
        input_df = pd.DataFrame(
            [input_data.model_dump(by_alias=True)]
        )

        processed_features, _, _, _ = process_data(
            input_df,
            categorical_features=CATEGORICAL_FEATURES,
            training=False,
            encoder=encoder,
            lb=label_binarizer,
        )

        prediction = inference(model, processed_features)

        predicted_class = label_binarizer.classes_[
            int(prediction[0])
        ]

        label = (
            "Income greater than $50K"
            if predicted_class == ">50K"
            else "Income at or below $50K"
        )

        return PredictionResponse(
            prediction=str(predicted_class),
            prediction_label=label,
        )

    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process the supplied Census data: {exc}",
        ) from exc
