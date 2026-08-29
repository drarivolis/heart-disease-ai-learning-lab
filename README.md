# Heart Disease AI Learning Lab

An educational Streamlit application developed for the book chapter **“Vibe Coding with No-Code and Low-Code AI for Healthcare Education and Clinical Training.”**
## Live Demonstration

Access the deployed educational application here:

[Heart Disease AI Learning Lab](https://heart-disease-ai-learning-lab-dqqdrrr3gx99byyflna6hg.streamlit.app/)

## Authors

- Arivoli Sundaramurthy, Department of Electrical and Electronics Engineering
- Chitra Vaithiyalingam, Department of Mathematics
- PSG Institute of Technology and Applied Research, Coimbatore, Tamil Nadu, India

## Important boundary

This application is exclusively for education and research demonstration. It must not be used for diagnosis, screening, treatment, or patient management. Do not enter real patient information.

## Dataset

UCI Heart Disease dataset, Cleveland subset: https://archive.ics.uci.edu/dataset/45/heart+disease

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C52P4X. License: CC BY 4.0.

## Verified experiment

Logistic Regression was selected using mean five-fold cross-validation AUROC.

| Metric | Result |
|---|---:|
| CV accuracy | 0.8552 ± 0.0238 |
| CV AUROC | 0.9025 ± 0.0148 |
| Holdout accuracy | 0.8689 |
| Holdout sensitivity | 0.9286 |
| Holdout specificity | 0.8182 |
| Holdout F1-score | 0.8667 |
| Holdout AUROC | 0.9665 |
| Holdout AUPRC | 0.9634 |
| Holdout Brier score | 0.0832 |

The holdout set contained 61 observations (TN=27, FP=6, FN=2, TP=26). Results are exploratory because the dataset is small and historical.

## Local execution

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Sign in to https://share.streamlit.io and connect GitHub.
3. Select the repository, `main` branch, and `streamlit_app.py`.
4. Deploy, test the public URL, and record the Git commit used for the chapter.

## Reproducibility note

The saved pipeline was created with scikit-learn 1.6.1; the same version is pinned in `requirements.txt` because scikit-learn model persistence is version-sensitive.

