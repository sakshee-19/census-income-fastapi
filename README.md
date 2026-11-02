# Census Income Classification API

A machine learning project that predicts whether an individual's annual income is above or at or below $50K using demographic and employment information from the U.S. Census dataset.

The project includes data preprocessing, model training, evaluation across categorical feature slices, a trained Random Forest model, a FastAPI prediction service, automated tests, and GitHub Actions CI.

**GitHub Repository:** https://github.com/sakshee-19/census-income-fastapi

## Project Overview

This project uses the Census Income dataset to train a binary classification model.

The model predicts one of two classes:

- `<=50K` — Annual income at or below $50K.
- `>50K` — Annual income above $50K.

The trained model and preprocessing artifacts are saved in the repository. A fresh clone can load these artifacts and serve predictions without retraining.

This is an educational demonstration, not a system intended for employment, lending, insurance, or other consequential decisions.

## Technology Stack

- Python 3.13
- pandas and NumPy
- scikit-learn
- FastAPI and Pydantic
- Uvicorn
- pytest
- Flake8
- GitHub Actions

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── census.csv
├── model/
│   ├── model.joblib
│   ├── encoder.joblib
│   ├── label_binarizer.joblib
│   └── metadata.joblib
├── screenshots/
├── starter/
│   ├── ml/
│   │   ├── data.py
│   │   └── model.py
│   └── train_model.py
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── live_post.py
├── main.py
├── model_card.md
├── requirements.txt
├── sanitycheck.py
├── setup.py
└── slice_output.txt
```

## Environment Setup

Python 3.13 is used for development and CI.

### 1. Clone the repository

```bash
git clone https://github.com/sakshee-19/census-income-fastapi.git
cd census-income-fastapi
```

### 2. Create and activate a virtual environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Model Training

The training script loads `data/census.csv`, preprocesses the data, trains a Random Forest classifier, evaluates its performance, and saves the fitted model and preprocessing artifacts.

Run:

```bash
python -m starter.train_model
```

The command generates or updates:

- `model/model.joblib`
- `model/encoder.joblib`
- `model/label_binarizer.joblib`
- `model/metadata.joblib`
- `slice_output.txt`

The training script uses a stratified train/test split with a fixed random seed for reproducibility.

## Model Evaluation

The latest training run produced the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 83.69% |
| Precision | 62.15% |
| Recall | 82.53% |
| F1 score | 70.90% |

The `>50K` class is treated as the positive class.

### Feature-Slice Evaluation

Precision, recall, and F1 are also calculated for individual values across the eight categorical features.

The generated results are available in:

```text
slice_output.txt
```

The model card documents the evaluation methodology, performance, intended use, limitations, and ethical considerations:

```text
model_card.md
```

## Running the API

Start the FastAPI application from the repository root:

```bash
uvicorn main:app --reload
```

The API will be available at:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API welcome information |
| `/predict` | POST | Predict an income category |
| `/docs` | GET | Interactive Swagger API documentation |

### Example prediction request

Send a POST request to `/predict` with the following JSON:

```json
{
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
  "native-country": "United-States"
}
```

Example response:

```json
{
  "prediction": "<=50K",
  "prediction_label": "Income at or below $50K"
}
```

The API validates incoming data using Pydantic and returns a validation error for invalid inputs.

## Running Tests

Run the complete test suite:

```bash
python -m pytest -v
```

The tests cover model training, metric calculation, inference, artifact persistence, API health, and predictions for both income classes.

Run the Udacity sanity checker:

```bash
python sanitycheck.py tests
```

## Code Quality

Run Flake8 while excluding the virtual environment and generated directories:

```bash
python -m flake8 . --count --show-source --statistics --exclude=.venv,.git,__pycache__,build,dist
```

## GitHub Actions — Continuous Integration

The project uses GitHub Actions to automate testing and code-quality checks.

Workflow location:

```text
.github/workflows/ci.yml
```

The workflow is configured to run on pushes and pull requests targeting `main` or `master`.

It performs the following steps:

1. Checks out the repository.
2. Sets up Python 3.13.
3. Installs project dependencies.
4. Runs pytest.
5. Runs Flake8.

Both tests and lint checks must pass for the CI workflow to succeed.

## Live API Request

The `live_post.py` script sends a sample Census record to a deployed API.

Set the API URL and execute:

```bash
API_URL="https://YOUR-DEPLOYED-API-URL" python live_post.py
```

The script prints the HTTP status code and prediction response.

## Screenshots

The `screenshots/` directory is intended to contain the required project evidence:

- `example.png` — Project implementation or example output.
- `continuous_deployment.png` — Successful CI/CD workflow.
- `live_get.png` — Live API GET endpoint response.
- `live_post.png` — Live API prediction response.

## Limitations and Responsible Use

The model learns patterns from historical Census data. Its predictions may reflect biases and limitations present in the original dataset.

Performance varies across demographic and other categorical groups. Some groups have limited test samples, making their measured performance less reliable.

The model should be used for educational and experimental purposes only. It should not be used to make decisions about individuals.

Refer to [model_card.md](model_card.md) for the complete model documentation and evaluation details.

## License

See [LICENSE.txt](LICENSE.txt).