# Bank Direct Marketing: Term Deposit Subscription Prediction

## Project Overview

This project addresses a classic marketing challenge: **predicting whether a bank client will subscribe to a term deposit** based on their demographic, historical, and socio-economic data. Accurate predictions can help banks optimize marketing campaigns, reduce costs, and improve customer targeting, ultimately increasing conversion rates and customer satisfaction.

---

## Problem Statement

Given a set of client and campaign features, build a machine learning model to predict if a client will subscribe to a term deposit (`subscribe = 1`) or not (`subscribe = 0`). The dataset is highly imbalanced, making robust evaluation and careful preprocessing essential.

---

## Dataset

- **Location:** `data/` directory
- **Files:**
  - `train.csv`: Training data (22 columns, 22,500 rows)
  - `test.csv`: Test data (21 columns, 7,500 rows)
  - `sample_submission.csv`: Submission format
  - `data description.txt`: Detailed feature descriptions

### Feature Summary

- **Client Data:** `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
- **Contact Data:** `contact`, `month`, `day_of_week`, `duration`
- **Campaign Data:** `campaign`, `pdays`, `previous`, `poutcome`
- **Socio-Economic Data:** `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`
- **Target:** `subscribe` (1 = yes, 0 = no)

See `data/data description.txt` for full details.

---

## Exploratory Data Analysis (EDA)

- **Class Imbalance:** The dataset is highly imbalanced, with far fewer positive (`subscribe = 1`) cases than negative.
- **Feature Distributions:** Categorical features such as `job`, `marital`, and `education` show significant variation in subscription rates.
- **Correlation:** Some features, like `duration` and `poutcome`, are strong predictors of subscription.
- **Missing Values:** Some categorical features include an `unknown` category, handled during preprocessing.

---

## Data Preprocessing & Feature Engineering

- **Label Encoding:** All categorical features are label-encoded for tree-based models.
- **One-Hot Encoding:** Alternative encoding for models that benefit from expanded feature space.
- **Handling Unknowns:** `unknown` values in categorical features are treated as a separate category.
- **Balancing:** Used `RandomOverSampler` from `imblearn` to address class imbalance, ensuring the model is not biased toward the majority class.
- **Feature Selection:** All features are retained after encoding, as tree-based models handle irrelevant features well.

---

## Model Selection & Training

- **Primary Model:** [CatBoostClassifier](https://catboost.ai/)  
  Chosen for its native handling of categorical features, robustness to overfitting, and strong performance on tabular data.
- **Alternative Model:** RandomForestClassifier (for benchmarking)
- **Hyperparameter Tuning:**  
  Used `GridSearchCV` to tune `iterations`, `depth`, and `learning_rate` for CatBoost.
- **Cross-Validation:**  
  5-fold cross-validation with F1-score as the main metric, due to class imbalance.

---

## Evaluation Metrics

- **F1 Score:**  
  Chosen for its balance between precision and recall, especially important for imbalanced datasets.
- **Other Metrics:**  
  Precision, recall, and confusion matrix were also considered during analysis.

---

## Results

- **Best Model:** CatBoostClassifier with:
  - `depth`: 8
  - `iterations`: 1000
  - `learning_rate`: 0.2
- **Final Mean F1 Score (cross-validation):**  
  **0.454** (rounded from 0.4537, as found at the end of the notebook)

### Interpretation

- The F1 score reflects the challenge of predicting rare events in imbalanced data.
- Further improvements could include advanced sampling, feature engineering, or ensemble methods.

---

## How to Use

### 1. Install Requirements

```bash
pip install catboost scikit-learn imbalanced-learn pandas numpy joblib
```

### 2. Run the Notebook

Open `Bank Direct Marketing.ipynb` and run all cells. The notebook will:
- Load and preprocess the data
- Train and evaluate models
- Generate predictions for the test set
- Save submission files in the `submissions/` directory

### 3. Make Predictions (Example)

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/df_label_encoded_overSample-model_CatBoostClassifer_1000_10_0.1.joblib')

# Prepare your data (example)
df_test = pd.read_csv('data/test.csv')
# ...apply the same preprocessing as in the notebook...

# Predict
predictions = model.predict(df_test)
```

---

## File Structure

```
.
├── Bank Direct Marketing.ipynb      # Main notebook
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── data description.txt
├── models/                         # Saved models
├── submissions/                    # Submission files
└── README.md
```

---

## Limitations & Future Work

- **Imbalanced Data:** Despite oversampling, predicting rare events remains challenging.
- **Feature Engineering:** More domain-specific features or interaction terms could improve performance.
- **Modeling:** Ensemble methods or stacking could be explored.
- **Deployment:** The current workflow is notebook-based; a production pipeline could be developed.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## References

- [CatBoost Documentation](https://catboost.ai/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [imbalanced-learn Documentation](https://imbalanced-learn.org/)
- [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) (original inspiration)

---

## Acknowledgements

- The dataset was taken from Kaggle, specifically from the [Bank Direct Marketing Kaggle Competition](https://www.kaggle.com/competitions/bankdirectmarketing).
- Data and problem inspired by real-world bank marketing campaigns.
- CatBoost and scikit-learn libraries for modeling.

---

**Final F1 Score:**  
**0.454** (CatBoost, cross-validated, see end of notebook for details)
