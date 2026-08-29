# Model Card — Educational Heart Disease Classifier

## Intended use

Formative education: compare a learner’s initial classification with the output of a logistic-regression pipeline trained on the UCI Cleveland Heart Disease dataset.

## Prohibited use

Diagnosis, screening, triage, treatment selection, patient management, clinical decision support, or high-stakes assessment.

## Data and target

303 historical Cleveland records; 13 predictors. Original `num` target transformed to binary: 0 = absence, >0 = presence. Six missing values were handled within the leakage-safe pipeline.

## Selection

Logistic Regression selected by the highest mean five-fold cross-validation AUROC (0.9025 ± 0.0148), not by holdout-set optimisation.

## Holdout results

Accuracy 0.8689; sensitivity 0.9286; specificity 0.8182; F1 0.8667; AUROC 0.9665; AUPRC 0.9634; Brier score 0.0832. Confusion matrix: TN=27, FP=6, FN=2, TP=26.

## Limitations

Small, historical, single-source dataset; limited demographic representation; binary simplification of a multiclass label; no external or prospective validation; interface values may not map to current clinical practice. Probabilities are model outputs, not clinical risk estimates.

