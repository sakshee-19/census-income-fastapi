
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from starter.ml.model import (
    compute_model_metrics,
    inference,
    load_model_artifacts,
    save_model_artifacts,
    train_model,
)


def test_train_model_returns_fitted_classifier():
    """The training function should return a fitted classifier."""
    X_train = np.array([
        [20, 1],
        [30, 0],
        [40, 1],
        [50, 0],
        [60, 1],
        [70, 0],
    ])
    y_train = np.array([0, 0, 1, 1, 1, 0])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    assert model.n_features_in_ == 2
    assert set(model.classes_) == {0, 1}


def test_compute_model_metrics():
    """Precision, recall, and F1 should match expected values."""
    y_true = np.array([1, 1, 0, 0])
    predictions = np.array([1, 0, 1, 0])

    precision, recall, f1 = compute_model_metrics(
        y_true, predictions
    )

    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_compute_model_metrics_when_no_positive_predictions():
    """Metrics should safely handle zero positive predictions."""
    y_true = np.array([1, 1, 0, 0])
    predictions = np.array([0, 0, 0, 0])

    precision, recall, f1 = compute_model_metrics(
        y_true, predictions
    )

    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test_inference_returns_expected_predictions():
    """Inference should return predictions with the expected shape."""
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = train_model(X_train, y_train)
    predictions = inference(model, X_train[:2])

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (2,)
    assert set(predictions).issubset({0, 1})


def test_save_and_load_model_artifacts(tmp_path):
    """All model artifacts and metadata should survive persistence."""
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = train_model(X_train, y_train)
    metadata = {"test": True, "version": 1}

    save_model_artifacts(
        model=model,
        encoder={"feature": "test"},
        label_binarizer={"classes": [0, 1]},
        output_dir=tmp_path,
        metadata=metadata,
    )

    loaded_model, encoder, label_binarizer, loaded_metadata = (
        load_model_artifacts(tmp_path)
    )

    assert isinstance(loaded_model, RandomForestClassifier)
    assert encoder == {"feature": "test"}
    assert label_binarizer == {"classes": [0, 1]}
    assert loaded_metadata == metadata

    original_predictions = inference(model, X_train)
    loaded_predictions = inference(loaded_model, X_train)

    np.testing.assert_array_equal(
        original_predictions, loaded_predictions
    )
