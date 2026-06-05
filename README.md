# Customer Churn Prediction

Brief project to prepare data and train classifiers to predict customer churn.

## Files

- `Customer-Churn-Prediction.ipynb` — Jupyter notebook with data loading, preprocessing, encoding, and model training/evaluation.
- `Dataset/Churn_Modelling.csv` — Source CSV dataset.

## Setup

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib
```

2. Start Jupyter and open the notebook:

```bash
pip install jupyter
jupyter notebook Customer-Churn-Prediction.ipynb
```

## How to run

- Open the notebook and run cells in order. The notebook performs:
  - Loading `Dataset/Churn_Modelling.csv`
  - Dropping identifier columns (`RowNumber`, `CustomerId`, `Surname`)
  - Duplicate removal and basic missing-value checks
  - Numeric scaling and categorical one-hot encoding via a `ColumnTransformer`
  - 80/20 stratified train/test split
  - Training and evaluation of classifiers (Logistic Regression and Random Forest)

## Preprocessing summary

- Numeric features are imputed (median) and scaled (StandardScaler).
- Categorical features (`Geography`, `Gender`) are imputed (most frequent) and one-hot encoded.
- Encoded train/test DataFrames are available in the notebook as `X_train_enc` and `X_test_enc`.

## Models & Results (quick)

- Logistic Regression (test accuracy): ~0.8080
- Random Forest (test accuracy): ~0.8605

These values are printed in the notebook after training.

## Next steps / suggestions

- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for Random Forest.
- Plot feature importances and partial dependence for interpretability.
- Persist best model with `joblib` or `pickle` for deployment.

## Contact

If you want, I can add a requirements file, a saved model export, or visualizations in the notebook.
