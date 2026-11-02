"""Model training, inference, evaluation, and artifact persistence."""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    """Train and return a reproducible random-forest classifier."""
    model = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(
    y: np.ndarray, preds: np.ndarray
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 (F-beta with beta=1)."""
    fbeta = fbeta_score(y, preds, beta=1, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    return float(precision), float(recall), float(fbeta)


def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """Run model inference and return the predicted class labels."""
    return np.asarray(model.predict(X))


def save_model_artifacts(
    model: Any,
    encoder: Any,
    label_binarizer: Any,
    output_dir: str | Path = "model",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the fitted estimator and preprocessing objects for inference."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path / "model.joblib")
    joblib.dump(encoder, output_path / "encoder.joblib")
    joblib.dump(label_binarizer, output_path / "label_binarizer.joblib")
    if metadata is not None:
        joblib.dump(metadata, output_path / "metadata.joblib")


def load_model_artifacts(
    model_dir: str | Path = "model",
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load all persisted artifacts needed to make predictions."""
    model_path = Path(model_dir)
    model = joblib.load(model_path / "model.joblib")
    encoder = joblib.load(model_path / "encoder.joblib")
    label_binarizer = joblib.load(model_path / "label_binarizer.joblib")
    metadata_path = model_path / "metadata.joblib"
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
    return model, encoder, label_binarizer, metadata
