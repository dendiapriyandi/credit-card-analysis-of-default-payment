# 💳 Credit Card Default Payment Analysis

As a data scientist in a banking environment, the goal of this project is to develop a predictive model that assesses the likelihood of new credit card clients defaulting on their payment in the next month. This project also includes a thorough exploratory data analysis (EDA) to uncover useful business insights.

---

## 🎯 Objective

- Identify the most suitable machine learning model to predict credit card default payments.
- Reduce false negatives, where a customer is wrongly predicted as non-defaulter.
- Support future business strategy through data-driven insights.

---

## 📊 Dataset Overview

The dataset is sourced from the UCI Machine Learning Repository and includes anonymized information on credit card clients in Taiwan.

| Feature | Description |
|--------|-------------|
| `limit_balance` | Total credit limit (individual + family) in NT dollars |
| `sex` | Gender (1 = male, 2 = female) |
| `education_level` | Education level (1 = grad school, 2 = university, etc.) |
| `marital_status` | Marital status (1 = married, 2 = single, etc.) |
| `age` | Age of client |
| `pay_0`, `pay_2`, ..., `pay_6` | Repayment status over previous months |
| `bill_amt1` - `bill_amt6` | Bill statement amounts |
| `pay_amt1` - `pay_amt6` | Payment amounts |
| `default_payment_next_month` | Default status (target variable) |

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked for null values and performed data cleaning
- Explored distribution of gender, education, marital status, and age
- Visualized trends in payment delays, bill amounts, and payment behaviors
- Analyzed correlation between variables and target class

---

## 🧠 Machine Learning Models

Evaluated various models to determine the best performance:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

### 📈 Metrics Used:

- Accuracy  
- Precision  
- Recall  
- F1 Score

---

## 🏆 Key Findings

- Payment history variables (`pay_0`, `pay_2`, etc.) strongly influence default prediction.
- The model with the best performance was Logistic Regression, achieving an accuracy of 78%.
- Significant imbalance in the dataset was addressed using resampling techniques.

---
