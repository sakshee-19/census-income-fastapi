"""Send a sample Census record to the deployed prediction API."""

import os
import sys

import requests

API_URL = os.getenv("API_URL", "").rstrip("/")

if not API_URL:
    sys.exit(
        "Error: Set API_URL to your deployed API URL before running."
    )

PREDICTION_URL = f"{API_URL}/predict"

SAMPLE_RECORD = {
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

try:
    response = requests.post(
        PREDICTION_URL,
        json=SAMPLE_RECORD,
        timeout=30,
    )

    print(f"HTTP Status Code: {response.status_code}")
    print(f"Response: {response.text}")

    response.raise_for_status()

    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Prediction Label: {result['prediction_label']}")

except requests.exceptions.RequestException as exc:
    print(f"API request failed: {exc}")
    sys.exit(1)
