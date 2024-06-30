# Credit Card Fraud Detection

This project aims to build a machine learning model to identify fraudulent credit card transactions. The dataset used for this project is highly imbalanced, with a small percentage of fraudulent transactions.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview
In this project, we explore and preprocess the credit card transaction data, handle class imbalance, and train two classification algorithms: logistic regression and random forest. The models are evaluated using precision, recall, and F1-score.

## Dataset
The dataset contains 284,807 transactions with 30 feature columns and 1 target column (`Class`), where:
- `0` indicates a genuine transaction.
- `1` indicates a fraudulent transaction.

The dataset can be found [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Preprocessing
1. **Normalization**: The `Amount` and `Time` columns are normalized using `StandardScaler`.
2. **Handling Class Imbalance**: The majority class (genuine transactions) is undersampled to match the number of minority class (fraudulent transactions).
3. **Train-Test Split**: The data is split into training and testing sets with a 70-30 ratio.

## Modeling
Two models are trained and evaluated:
- **Logistic Regression**
- **Random Forest Classifier**

### Model Training and Evaluation
The models are trained on the undersampled dataset and evaluated on the test set. The following metrics are used for evaluation:
- Precision
- Recall
- F1-score

## Evaluation
### Logistic Regression Performance:
- **Accuracy**: 94%
- **Precision**: 95% for fraudulent transactions
- **Recall**: 93% for fraudulent transactions
- **F1-score**: 94% for fraudulent transactions

### Random Forest Performance:
- **Accuracy**: 94%
- **Precision**: 96% for fraudulent transactions
- **Recall**: 91% for fraudulent transactions
- **F1-score**: 94% for fraudulent transactions

Both models perform similarly, but logistic regression has a slightly higher recall, and random forest has a slightly higher precision.

## Usage
To run this project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your_username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Install the necessary packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the script:
    ```sh
    python fraud_detection.py
    ```

### Dependencies
- pandas
- scikit-learn


