# Heart Failure Prediction — Reproducible ML Pipeline

This repository implements a reproducible pipeline for preprocessing, exploratory data analysis (EDA), hyperparameter optimization (Optuna), model training, evaluation, and a Streamlit-based deployment for predicting heart disease.

## Summary
- Dataset (processed): `data/processed/heart_clean.csv`  
- Core code: `src/models.py`, `src/visualizations.py`  
- Notebooks: `notebooks/models.ipynb`, `notebooks/processing.ipynb`, `notebooks/eda1.ipynb`  
- App: `app.py` (Streamlit)  
- Artifacts: recommended `model/modelo_heart_disease_stacking.pkl`  
- Reports: `reports/figures`, `reports/tables`  
- Dependencies: `requirements.txt`

## Purpose
Provide a transparent, reproducible workflow from raw data preprocessing through model development (including Optuna hyperparameter search) to a deployable ensemble model with interpretability outputs and a lightweight UI for demonstration.

## Repository structure (key files)
- `app.py` — Streamlit app and model loader.
- `src/models.py` — Model training, baseline evaluations, Optuna search functions, and stacking builder. Key functions: `baseline`, `randomforest_basemodel`, `svc_basemodel`, `decisiontree_basemodel`, `xgboost_basemodel`, `logistic_regression_optuna`, `random_forest_optuna`, `svc_optuna`, `create_triple_stacking`, `plot_ensemble_importance`.
- `src/visualizations.py` — EDA and plotting helpers.
- `notebooks/models.ipynb` — Full experiment flow (baselines → Optuna → stacking → artifact save).
- `data/processed/heart_clean.csv` — Cleaned dataset used for modeling.
- `model/` — Recommended location for saved model artifacts.
- `reports/` — Generated figures and tables.

## Environment & installation
1. Create a virtual environment (venv/conda) and activate it.  
2. Install dependencies:
```bash
pip install -r requirements.txt
```
Recommended Python version: 3.10+.

## End-to-end reproducible sequence (detailed)
1. Load data
```python
from pathlib import Path
import pandas as pd
ROOT = Path.cwd()
df = pd.read_csv(ROOT / "data" / "processed" / "heart_clean.csv")
```

2. Quick baseline (sanity checks)
- Run `src.models.baseline(df, "HeartDisease")` to obtain a baseline logistic regression performance and basic diagnostics (confusion matrix, classification report, ROC-AUC).

3. Preprocessing conventions
- Numeric features scaled with `StandardScaler`.
- Categorical features encoded with `OneHotEncoder(handle_unknown='ignore')`.
- These transformations are implemented in `ColumnTransformer` inside each training pipeline in `src/models.py`.

4. Train/test splitting
- Stratified splits used across experiments. Default `random_state=42` is applied for reproducibility. Typical test_size values: 0.2–0.3 (see function docstrings).

5. Baseline model evaluations
- Run base models to compare initial performance:
  - `randomforest_basemodel(df, "HeartDisease")`
  - `svc_basemodel(df, "HeartDisease")`
  - `decisiontree_basemodel(df, "HeartDisease")`
  - `xgboost_basemodel(df, "HeartDisease")`

6. Hyperparameter optimization (Optuna)
- Logistic Regression: `logistic_regression_optuna(df, "HeartDisease")` — typically 100 trials, scoring ROC-AUC, uses CV internally.
- Random Forest: `random_forest_optuna(df, "HeartDisease")` — typically 20 trials.
- SVC: `svc_optuna(df, "HeartDisease")` — typically 50 trials.
Implementation details: objective functions use stratified CV and `roc_auc` as the optimization metric.

7. Final model building / ensemble
- Obtain best parameter dicts from Optuna runs and build a stacking classifier:
```python
params_lr = logistic_regression_optuna(df, "HeartDisease")
params_rf = random_forest_optuna(df, "HeartDisease")
params_svc = svc_optuna(df, "HeartDisease")

modelo_ensamble_final = create_triple_stacking(
    df=df,
    target="HeartDisease",
    params_lr=params_lr,
    params_rf=params_rf,
    params_svc=params_svc
)
```
- Evaluate with cross-validation and report ROC-AUC, precision, recall, F1, and confusion matrices (functions print and return metrics).

8. Interpretability and diagnostics
- Permutation importance and ensemble importance are generated via `plot_ensemble_importance` and `sklearn.inspection.permutation_importance`. Save plots to `reports/figures`.

9. Artifact saving (standardized)
- Save final model artifact to `model/modelo_heart_disease_stacking.pkl` (repo-root `model` folder). Example:
```python
import joblib
joblib.dump(modelo_ensamble_final, "model/modelo_heart_disease_stacking.pkl")
```

10. Deployment (Streamlit)
- Run the app:
```bash
python -m streamlit run app.py
```
- The app loads the artifact `model/modelo_heart_disease_stacking.pkl`, accepts user inputs, shows predicted probability and a simple risk label, and displays basic explainability plots.

## Reported results (representative)
- Example runs (notebook outputs):
  - Random Forest baseline ROC-AUC: ≈ 0.95–0.96
  - XGBoost baseline ROC-AUC: ≈ 0.94–0.95
  - Optuna-tuned Logistic Regression ROC-AUC: ≈ 0.95
  - Optuna-tuned SVC ROC-AUC: ≈ 0.94–0.95
  - Final stacking ensemble ROC-AUC: ≈ 0.95 (see `notebooks/models.ipynb` for per-run prints and saved figures)
Note: these are representative values observed locally during development; re-runs may vary due to data splits and random seeds.

## Best practices and recommendations
- Standardize artifact paths: use `model/` at repo root for saved models and ensure notebooks and app reference the same path.
- Persist Optuna studies to disk (SQLite) for reproducibility: `optuna.create_study(storage="sqlite:///optuna.db", ...)`.
- Implement nested CV for unbiased model selection when reporting final performance.
- Add calibration plots (Platt scaling / isotonic) and SHAP analyses for per-sample explanations.
- Evaluate fairness metrics across demographic groups and include limitations / clinical disclaimer.
- Add CI, unit tests for preprocessors, and an environment file (environment.yml or requirements pinned).

## How to reproduce exact notebook run (concise)
1. From repo root:
```bash
pip install -r requirements.txt
jupyter lab  # or open notebooks in VS Code
```
2. Open `notebooks/models.ipynb` and run cells in order. After stacking, the notebook will save the model to `model/modelo_heart_disease_stacking.pkl`.


