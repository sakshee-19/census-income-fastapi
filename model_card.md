# Model Card: Census Income Classification

## Model Details

- **Model type:** Supervised binary classification using a scikit-learn `RandomForestClassifier`.
- **Purpose:** Estimate which of two income categories (`<=50K` or `>50K`) is associated with a Census record, using the supplied demographic and employment-related features.
- **Implementation:** The model is trained by `starter/train_model.py`; reusable training, inference, metric, and artifact-persistence functions are in `starter/ml/model.py`.
- **Preprocessing:** The eight categorical features are one-hot encoded with a fitted `OneHotEncoder` (`handle_unknown="ignore"`). The target is encoded with a fitted `LabelBinarizer`. Continuous/integer columns are passed through without scaling.
- **Training configuration:** 150 trees, `min_samples_leaf=2`, `class_weight="balanced_subsample"`, `random_state=42`, and `n_jobs=-1`.
- **Artifact format:** The fitted estimator, encoder, label binarizer, and feature/metric metadata are saved separately as joblib files under `model/`.

## Intended Use

This model is an educational demonstration of a reproducible machine-learning workflow and a FastAPI inference service using the supplied Census income dataset. It can be used to demonstrate preprocessing, model training, evaluation, API integration, automated testing, and deployment.

It is **not** intended for decisions about an individual's employment, compensation, credit, housing, insurance, eligibility, or access to services. It should not be used to make or justify consequential decisions about real people.

## Training Data

The model uses the project-provided `data/census.csv` dataset. The supplied file contains 32,561 records and 15 columns: 14 input features and the `salary` target. The target classes in this file are `<=50K` (24,720 records) and `>50K` (7,841 records).

Leading/trailing whitespace is removed from string values before splitting. The target column is `salary`. The categorical input columns are `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`. The remaining input columns are treated as numeric features.

A stratified 80/20 train/test split is created with `random_state=42`. The encoder and label binarizer are fitted on the training partition only, then reused unchanged for the test partition and inference.

## Evaluation Data

Evaluation uses the held-out 20% test partition from the stratified split (6,513 records). The test partition is not used to fit the preprocessing objects or train the estimator. The reported metrics are from the run represented by the committed model artifacts and `slice_output.txt`.

## Metrics

The following metrics were calculated on the held-out test partition:

| Metric | Result |
|---|---:|
| Accuracy | 0.8369 (83.69%) |
| Precision | 0.6215 (62.15%) |
| Recall | 0.8253 (82.53%) |
| F1 score | 0.7090 (70.90%) |

Precision, recall, and F1 are calculated for the positive class (`>50K`) with zero-division results set to zero. Accuracy is the proportion of correct predictions. Slice-level precision, recall, and F1 are calculated separately for each observed value of each of the eight categorical features; the results are recorded in `slice_output.txt`.

## Ethical Considerations

The dataset contains sensitive or potentially sensitive demographic attributes, including age, sex, and race, as well as employment and relationship information. These attributes and their correlations with income can encode historical and structural inequalities. Model predictions may therefore reproduce or amplify disparities present in the source data.

The overall metrics do not establish fairness. Slice metrics are provided as a basic diagnostic, not as proof of equitable performance. Some categories have very few test examples, making their metrics unstable or uninformative. A zero precision/recall/F1 score for a small slice should not be interpreted as a reliable estimate of population-level performance.

The model should not be used to rank, screen, profile, or make consequential decisions about individuals or groups. Before any research or operational use, data provenance, consent and lawful basis, representativeness, privacy, subgroup performance, disparate impact, and domain-specific risks would require independent review. Avoid exposing individual-level predictions or input records unnecessarily.

## Caveats and Recommendations

- The supplied dataset is a historical snapshot and may not represent current populations, labor markets, income definitions, or economic conditions.
- The target is a coarse binary income label; it does not represent a person's complete financial circumstances or worth.
- The dataset is class-imbalanced. The reported precision and recall should be considered together rather than relying on accuracy alone.
- The model is a Random Forest baseline, not a tuned or externally validated production model. No claim is made that it is optimal.
- The evaluation is a single random holdout split. Additional cross-validation, temporal/external validation, calibration analysis, threshold analysis, and robust subgroup/fairness evaluation would be needed before drawing stronger conclusions.
- Slice metrics can vary substantially for rare categories. Always inspect each slice's sample count alongside its metrics; do not overinterpret tiny groups.
- One-hot encoding ignores previously unseen categories at inference time. This allows the API to accept unknown categories, but does not guarantee reliable predictions for them.
- Input validation checks basic types and numeric bounds; it does not guarantee that every categorical value is valid or representative of the training data.
- Re-train and re-evaluate when the source data, feature definitions, or intended deployment context changes. Keep the training data version, code version, model artifacts, and evaluation results together for reproducibility.
