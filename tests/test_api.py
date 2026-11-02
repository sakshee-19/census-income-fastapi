
"""Integration tests for the FastAPI application."""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

LOW_INCOME = {
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

HIGH_INCOME = {
    "age": 50,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 83311,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 13,
    "native-country": "United-States",
}


def test_get_root_returns_welcome_message():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Welcome to the Census Income Classification API!"
    )
    assert response.json()["docs"] == "/docs"
    assert response.json()["prediction_endpoint"] == "/predict"


def test_post_model_predicts_at_most_50k():
    response = client.post("/predict", json=LOW_INCOME)

    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"
    assert response.json()["prediction_label"] == (
        "Income at or below $50K"
    )


def test_post_model_predicts_above_50k():
    response = client.post("/predict", json=HIGH_INCOME)

    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
    assert response.json()["prediction_label"] == (
        "Income greater than $50K"
    )


def test_post_model_rejects_invalid_age():
    invalid_record = LOW_INCOME.copy()
    invalid_record["age"] = 10

    response = client.post("/predict", json=invalid_record)

    assert response.status_code == 422
