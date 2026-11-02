"""Train the Census income classifier and evaluate its performance."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    inference,
    save_model_artifacts,
    train_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "census.csv"
MODEL_DIR = PROJECT_ROOT / "model"

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def compute_slice_metrics(
    data: pd.DataFrame,
    feature: str,
    model,
    encoder,
    label_binarizer,
) -> list[dict[str, object]]:
    """Compute precision, recall, and F1 for each value of a category."""
    results: list[dict[str, object]] = []
    for category_value in sorted(data[feature].dropna().unique(), key=str):
        subset = data.loc[data[feature] == category_value]
        if subset.empty:
            continue
        X_slice, y_slice, _, _ = process_data(
            subset,
            categorical_features=CATEGORICAL_FEATURES,
            label=LABEL,
            training=False,
            encoder=encoder,
            lb=label_binarizer,
        )
        predictions = inference(model, X_slice)
        precision, recall, f1 = compute_model_metrics(y_slice, predictions)
        results.append(
            {
                "feature": feature,
                "value": str(category_value),
                "rows": int(len(subset)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return results


def main() -> None:
    """Train and evaluate the model, then save artifacts and slice metrics."""
    data = pd.read_csv(DATA_PATH, skipinitialspace=True)
    # Normalize surrounding whitespace without changing the dataset's columns.
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].str.strip()

    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data[LABEL]
    )
    X_train, y_train, encoder, label_binarizer = process_data(
        train,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=label_binarizer,
    )

    model = train_model(X_train, y_train)
    predictions = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, predictions)
    accuracy = float((y_test == predictions).mean())
    print("Test metrics")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    metadata = {
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": [
            column for column in data.columns if column != LABEL
        ],
        "label_column": LABEL,
        "classes": [str(value) for value in label_binarizer.classes_],
        "test_size": 0.20,
        "random_state": 42,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }
    save_model_artifacts(model, encoder, label_binarizer, MODEL_DIR, metadata)

    # Report slices for every categorical feature, as required by the rubric.
    slice_rows: list[dict[str, object]] = []
    for feature in CATEGORICAL_FEATURES:
        slice_rows.extend(
            compute_slice_metrics(
                test, feature, model, encoder, label_binarizer
            )
        )
    slice_output = PROJECT_ROOT / "slice_output.txt"
    with slice_output.open("w", encoding="utf-8") as output:
        output.write("Census income model: test-set slice metrics\n")
        output.write(
            "Metric values are computed on the held-out test split.\n\n"
        )
        output.write("feature,value,rows,precision,recall,f1\n")
        for row in slice_rows:
            output.write(
                f"{row['feature']},{row['value']},{row['rows']},"
                f"{row['precision']:.4f},{row['recall']:.4f},{row['f1']:.4f}\n"
            )
    print(f"Saved model artifacts to {MODEL_DIR}")
    print(f"Saved slice metrics to {slice_output}")


if __name__ == "__main__":
    main()
