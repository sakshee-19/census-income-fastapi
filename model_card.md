# Model Card: Census Income Classification

For additional information, see the Model Cards paper:
https://arxiv.org/abs/1810.03993

## Model Details

- **Model name:** Census Income Classification Model
- **Model type:** Supervised binary classification
- **Algorithm:** Random Forest Classifier
- **Framework:** Scikit-learn
- **Target variable:** `salary`
- **Prediction classes:** `<=50K` and `>50K`
- **Positive class:** `>50K`
- **Random state:** 42
- **Number of estimators:** 150
- **Minimum samples per leaf:** 2
- **Class weighting:** `balanced_subsample`

The model predicts whether an individual's annual income falls
above or at/below the $50,000 threshold using demographic and
employment-related Census features.

Categorical variables are transformed using a fitted
OneHotEncoder with `handle_unknown="ignore"`. The fitted encoder,
label binarizer, classifier, and training metadata are persisted
as model artifacts to support inference without retraining.

The model was developed using Python 3.13 and scikit-learn 1.8.0.

## Intended Use

The model is intended for educational and demonstration purposes,
including:

- Demonstrating an end-to-end machine learning workflow.
- Learning model training, evaluation, and preprocessing.
- Demonstrating prediction serving through a FastAPI application.
- Exploring model performance across categorical feature groups.
- Practicing automated testing and cloud deployment.

The model is not intended for making decisions about individuals.

It must not be used for employment, hiring, lending, credit
decisions, insurance eligibility, housing, public benefits, or
other consequential decisions.

Predictions should not be interpreted as verified information
about an individual's actual income.

## Training Data

The model uses the Census income dataset provided with this
project in `data/census.csv`.

- **Target column:** `salary`
- **Input features:** 14 columns
- **Categorical features:** 8
- **Continuous/numeric features:** 6

Categorical features:

- `workclass`
- `education`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `native-country`

The remaining input features are numeric.

Preprocessing includes:

- Loading the CSV while handling whitespace after delimiters.
- Trimming whitespace from categorical values.
- Separating the target variable from input features.
- One-hot encoding categorical features.
- Binarizing the target labels.

The dataset contains demographic and socioeconomic information.
Such information can reflect historical inequalities and
structural differences in access to income and opportunity.

The model was trained on 80% of the available dataset using a
stratified train/test split and random state 42.

## Evaluation Data

The remaining 20% of the dataset was held out for evaluation.

The same fitted preprocessing objects used during training were
applied to the evaluation data.

The evaluation data was not used to fit the model.

The held-out test set was also used to calculate performance
metrics across categorical feature slices.

## Metrics

The following metrics were measured on the held-out test set.

| Metric | Result |
|---|---:|
| Accuracy | 83.69% |
| Precision | 62.15% |
| Recall | 82.53% |
| F1 score | 70.90% |

Metrics are calculated with `>50K` treated as the positive class.

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are
  actually positive.
- **Recall:** Proportion of actual positive cases identified
  by the model.
- **F1 score:** Harmonic mean of precision and recall.

The model achieves 83.69% accuracy and 82.53% recall.
Its lower precision of 62.15% indicates that a meaningful
proportion of predicted higher-income cases are false positives.

### Feature-Slice Evaluation

Precision, recall, and F1 were also calculated separately for
values within each of the eight categorical features.

The results are recorded in `slice_output.txt`.

Some observations from the held-out test set include:

| Feature | Group | Rows | F1 |
|---|---|---:|---:|
| Race | Amer-Indian-Eskimo | 73 | 0.4444 |
| Race | Asian-Pac-Islander | 200 | 0.6719 |
| Race | Black | 662 | 0.6866 |
| Race | White | 5,533 | 0.7126 |
| Sex | Female | 2,158 | 0.7000 |
| Sex | Male | 4,355 | 0.7105 |
| Education | Bachelors | 1,096 | 0.7780 |
| Education | HS-grad | 2,120 | 0.5686 |
| Education | Prof-school | 114 | 0.8925 |
| Relationship | Own-child | 1,032 | 0.3810 |
| Relationship | Wife | 336 | 0.7423 |

These results show that model performance varies across
categorical groups.

Some groups have very small sample sizes. Their precision,
recall, and F1 scores may be unstable and should not be treated
as reliable estimates of real-world performance.

Slice metrics are descriptive evaluations. They do not, by
themselves, establish whether the model is fair or unbiased.

## Ethical Considerations

The dataset contains sensitive demographic attributes,
including race and sex, as well as socioeconomic and
employment-related information.

Potential risks include:

- Historical bias in the source data may be learned by the model.
- Differences in group representation may affect model quality.
- Performance may vary substantially across demographic groups.
- Small sample sizes can produce misleading subgroup metrics.
- Income classification can reinforce socioeconomic stereotypes.
- Predictions may be incorrectly treated as facts about
  individuals.

Although race and sex are included in the input data, excluding
these features alone would not guarantee fairness because other
features may act as proxies for sensitive attributes.

The model has not undergone a comprehensive fairness audit,
causal analysis, or assessment of real-world social impact.

Human oversight, additional validation, and fairness testing
would be necessary before considering any use beyond education.

## Caveats and Recommendations

1. This model is a learning project and should not be used for
   real-world individual assessment or consequential decisions.
2. The evaluation metrics describe performance on one held-out
   split and may not generalize to other populations or time
   periods.
3. The model predicts an income category, not an exact income.
4. Dataset labels and features may contain errors, omissions,
   and historical biases.
5. Performance varies across categorical feature groups.
6. Metrics for groups with very few records are particularly
   unreliable.
7. The model has not been independently audited for fairness,
   privacy, or security.
8. Changes in population characteristics or economic conditions
   may reduce prediction quality over time.
9. Further work should include cross-validation, calibration,
   fairness analysis, subgroup error analysis, and evaluation
   on representative external data.

The model should be treated as an educational demonstration
of machine learning deployment rather than a production-ready
income assessment system.
