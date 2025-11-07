# Churn Prediction for SaaS using Machine Learning

This project aims to build and compare several machine learning models (Softmax Regression, SVM, Random Forest) to predict a customer's churn risk score (1-5) and identify the key drivers of churn in a SaaS business context.

***
## Project Objective

The goal is to develop a predictive model that can:
- Predict the churn risk score (1-5) for each customer.
- Identify the most important features influencing churn.
- Compare the performance of multiple machine learning algorithms.

***

## Dataset Description

- The dataset contains customer information such as age, gender, region, membership category, joining date, referral status, transaction history, and feedback.
- The target variable is `churnriskscore`, which ranges from 1 to 5, representing the likelihood of churn.
- Key features include:
  - `age`, `gender`, `regioncategory`, `membershipcategory`
  - `avgtransactionvalue`, `avgfrequencylogindays`, `pointsinwallet`
  - `usedspecialdiscount`, `offerapplicationpreference`, `pastcomplaint`
  - `feedback` (categorical, processed using NLP techniques)

Dataset Used: [Churn Risk Dataset on Kaggle](https://www.kaggle.com/datasets/imsparsh/churn-risk-rate-hackerearth-ml/data)

***

## Methodology

1. **Data Preprocessing**
   - Handle missing values and drop irrelevant columns (e.g., `lastvisittime`).
   - Convert categorical features to numerical using encoding.
   - Use NLP to process textual feedback into numerical features.

2. **Exploratory Data Analysis (EDA)**
   - Visualize the distribution of churn risk scores.
   - Analyze correlations between features and churn.

3. **Feature Engineering**
   - Extract meaningful features from raw data.
   - Use SMOTE to balance the training dataset.

4. **Model Selection & Training**
   - Train and compare Softmax Regression, SVM, and Random Forest models.
   - Perform hyperparameter tuning using Optuna for Random Forest.

5. **Model Evaluation**
   - Evaluate models using accuracy, precision, recall, and F1-score.
   - Use cross-validation for robustness.

***

## Model Comparison

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Softmax Regression   | 0.58     | 0.56      | 0.58   | 0.55     |
| SVM                  | 0.73     | 0.73      | 0.73   | 0.72     |
| Random Forest        | 0.74     | 0.74      | 0.74   | 0.74     |

***

## Key Features

- **NLP Integration**: Feedback text is processed using lemmatization and stopword removal.
- **Hyperparameter Tuning**: Optuna is used for optimizing Random Forest hyperparameters.
- **Imbalanced Data Handling**: SMOTE is applied to balance the training dataset.
- **Comprehensive Evaluation**: Multiple metrics are used to assess model performance.
***
## Results

- The Random Forest model achieved the highest accuracy (74%) and F1-score.
- Key drivers of churn include `avgtransactionvalue`, `avgfrequencylogindays`, and `feedback`.
- The model can be used to identify at-risk customers and implement targeted retention strategies.



