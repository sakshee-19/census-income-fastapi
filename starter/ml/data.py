
"""Data preprocessing for Census income classification."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """Preprocess features and optionally transform target labels.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features, optionally including the target label.
    categorical_features : list[str] or None
        Names of categorical feature columns.
    label : str or None
        Target column name. None for inference without labels.
    training : bool
        Fit preprocessing objects when True; reuse them otherwise.
    encoder : OneHotEncoder or None
        Fitted encoder required during inference.
    lb : LabelBinarizer or None
        Fitted label binarizer required during labeled processing.

    Returns
    -------
    tuple
        Processed features, labels, encoder, and label binarizer.
    """
    if categorical_features is None:
        categorical_features = []

    X = X.copy()

    if label is not None:
        if label not in X.columns:
            raise ValueError(f"Target column '{label}' is missing.")
        y = X.pop(label)
    else:
        y = None

    missing_features = [
        column for column in categorical_features if column not in X.columns
    ]
    if missing_features:
        raise ValueError(
            f"Categorical feature columns are missing: {missing_features}"
        )

    X_categorical = X[categorical_features].to_numpy()
    X_continuous = X.drop(columns=categorical_features).to_numpy()

    if training:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
        X_categorical = encoder.fit_transform(X_categorical)

        if y is not None:
            lb = LabelBinarizer()
            y = lb.fit_transform(y.to_numpy()).ravel()
        else:
            y = np.array([])
    else:
        if encoder is None:
            raise ValueError("A fitted encoder is required for inference.")

        X_categorical = encoder.transform(X_categorical)

        if y is not None:
            if lb is None:
                raise ValueError(
                    "A fitted label binarizer is required "
                    "when labels are supplied."
                )
            y = lb.transform(y.to_numpy()).ravel()
        else:
            y = np.array([])

    processed_X = np.concatenate(
        [X_continuous, X_categorical],
        axis=1,
    )

    return processed_X, y, encoder, lb
