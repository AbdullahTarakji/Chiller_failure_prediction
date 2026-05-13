# Chiller Failure Prediction

Machine learning project for predicting chiller failures from operational sensor data.

## Overview
This repository contains code and experiments to build a predictive model that detects (or forecasts) chiller failures using historical operational / sensor readings. The goal is to enable proactive maintenance by identifying elevated failure risk early.

## Dataset
- **Source:** Not specified in the repository yet.
- **Expected format:** A tabular dataset (CSV/Parquet) where each row is a timestamped observation with sensor/operational features.
- **Target:** A label/indicator representing chiller failure (binary) or failure class (multiclass).

> If you share the dataset link and column names, the README can be updated to be specific.

## Approach (recommended)
Typical workflow:
1. Data cleaning and validation (missing values, outliers, time alignment)
2. Feature engineering (rolling statistics, lag features, seasonality)
3. Model training (e.g., Logistic Regression, Random Forest, XGBoost, LightGBM)
4. Evaluation (precision/recall, ROC-AUC, PR-AUC; and time-aware validation)
5. Export model + inference script/notebook

## Installation
Create a virtual environment and install dependencies.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If `requirements.txt` is not present yet, you can generate it after installing your packages:

```bash
pip freeze > requirements.txt
```

## Usage
Because repository structure may evolve, the most common entry points are:

- **Notebooks**: run exploratory analysis and training notebooks (if present)
- **Scripts**: run training/inference scripts (if present)

Suggested commands (adjust paths/names once present):

```bash
# Example: train a model
python train.py --data data/chiller.csv --target failure

# Example: run inference
python predict.py --model artifacts/model.pkl --data data/new_readings.csv
```

## Results
Results/metrics are not documented yet. Recommended to report:
- Confusion matrix
- Precision / Recall / F1 (especially for the failure class)
- PR-AUC (often more informative with imbalanced failures)
- Calibration (optional)

## Project Structure (suggested)
```text
.
├── data/                 # local data (usually gitignored)
├── notebooks/            # exploration and experiments
├── src/                  # reusable code
├── artifacts/            # trained models/encoders
├── requirements.txt
└── README.md
```

## Future Work
- Add clear dataset description and a small sample schema
- Add reproducible training pipeline (config + CLI)
- Add time-series aware cross-validation
- Add model tracking (MLflow) and model export

## License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
