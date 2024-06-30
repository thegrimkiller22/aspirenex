# Titanic Survival Prediction

This project aims to predict whether a passenger on the Titanic survived or not using the Titanic dataset. The dataset contains information about individual passengers such as their age, gender, ticket class, fare, cabin, and whether or not they survived. Two machine learning models, Random Forest Classifier and Gradient Boosting Classifier, are used to perform the predictions.

## Dataset

The dataset used in this project is the Titanic dataset, which can be found [here](https://www.kaggle.com/c/titanic/data).

## Project Structure

- `titanic.csv`: The dataset file.
- `titanic_survival.py`: The main script to preprocess the data, train models, and evaluate performance.
- `README.md`: Project documentation.

## Installation

To run this project, you need Python and the following libraries installed:

- pandas
- scikit-learn
- numpy

You can install these libraries using pip:

```bash
pip install pandas scikit-learn numpy
```

## Usage

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Place the `titanic.csv` file in the project directory.

3. Run the `titanic_survival.py` script:

```bash
python titanic_survival.py
```

## Model Evaluation

The script trains and evaluates two models: Random Forest Classifier and Gradient Boosting Classifier. The accuracy, classification report, and confusion matrix for each model are printed to the console.

### Random Forest Classifier

```plaintext
Random Forest Accuracy: 0.82
Random Forest Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.89      0.86        50
           1       0.79      0.71      0.75        34

    accuracy                           0.82        84
   macro avg       0.81      0.80      0.81        84
weighted avg       0.82      0.82      0.82        84

Random Forest Confusion Matrix:
[[45  5]
 [10 24]]
```

### Gradient Boosting Classifier

```plaintext
Gradient Boosting Accuracy: 0.83
Gradient Boosting Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.90      0.88        50
           1       0.79      0.74      0.76        34

    accuracy                           0.83        84
   macro avg       0.82      0.82      0.82        84
weighted avg       0.83      0.83      0.83        84

Gradient Boosting Confusion Matrix:
[[45  5]
 [ 9 25]]
```


