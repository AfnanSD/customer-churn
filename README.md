# Customer Churn Prediction

## Project Overview

This project explores customer activity data and applies several pre-built machine learning models from scikit-learn and other libraries to predict which users are most likely to cancel their subscription.
The focus is on data understanding, preparation, and evaluation rather than model architecture design.
Through exploratory data analysis (EDA), feature engineering, and model comparison, the notebook aims to identify which algorithms perform best in detecting potential churned users.

---

## Approach

### 1. Exploratory Data Analysis (EDA)

* Examined the structure and columns of both datasets (`customer_churn` and `customer_churn_mini`).
* Identified key features such as `page`, `level` and `length`.
* Determined that the unique identifier for each row is a combination of (`userId`, `ts`, `itemInSession`).
* Checked for missing values and inconsistent records.
* Created summaries by month and explored user activity trends.

### 2. Feature Engineering

* Generated user-level features such as:
  * Tenure days
  * Total listen time
  * Average song length
  * Total number of songs listened
* Defined a churn label to indicate whether a user cancelled their subscription.
* Encoded categorical features and normalized numeric ones.

### 3. Modeling and Evaluation

* Trained and compared multiple machine learning models:

  * Logistic Regression (LR)
  * Random Forest (RF)
  * LightGBM (LGBM)
  * XGBoost (XGB)
* Evaluated performance using the following metrics:

  * ROC-AUC
  * Precision-Recall AUC
  * Precision
  * Recall

Example of model performance:

| Model               | ROC-AUC | PR-AUC |
| ------------------- | ------- | ------ |
| Logistic Regression | 0.82    | 0.56   |
| **Random Forest **      | 0.86    | 0.66   |
| LightGBM            | 0.81    | 0.63   |
| XGBoost             | 0.82    | 0.64   |

---

## Tools and Libraries

* **Python**
* **pandas**, **numpy** – Data manipulation and analysis
* **matplotlib**, **seaborn** – Data visualization
* **scikit-learn**, **xgboost**, **lightgbm** – Machine learning and model evaluation

---

## Challenges Faced

* **Class imbalance:** The dataset contained far fewer churned users compared to active users.
* **Model tuning:** Required experimentation with hyperparameters to improve generalization.

---

## Future Improvements

* Apply **SMOTE** or other resampling techniques to handle class imbalance.
* Try **advanced models** such as CatBoost or neural networks.
* Explore **time-series analysis** to capture temporal churn patterns.
* Deploy the trained model as a **REST API** using FastAPI or Flask.
