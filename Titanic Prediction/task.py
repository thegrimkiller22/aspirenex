import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

#plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

file_path = 'titanic.csv'
titanic_data = pd.read_csv(file_path)

# copy to modifying org value
titanic_data_copy = titanic_data.copy()

# Fill missing values
titanic_data_copy['Age'] = titanic_data_copy['Age'].fillna(titanic_data_copy['Age'].median())
titanic_data_copy['Fare'] = titanic_data_copy['Fare'].fillna(titanic_data_copy['Fare'].median())

# Drop the unused or useless column
titanic_data_copy = titanic_data_copy.drop(columns=['Cabin'])

# Encode categorical variables
label_encoder = LabelEncoder()
titanic_data_copy['Sex'] = label_encoder.fit_transform(titanic_data_copy['Sex'])
titanic_data_copy['Embarked'] = label_encoder.fit_transform(titanic_data_copy['Embarked'])

# Drop unnecessary columns
titanic_data_copy = titanic_data_copy.drop(columns=['PassengerId', 'Name', 'Ticket'])

# x=feature y=targets
X = titanic_data_copy.drop(columns=['Survived'])
y = titanic_data_copy['Survived']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_classification_report = classification_report(y_test, rf_y_pred)
rf_confusion_matrix = confusion_matrix(y_test, rf_y_pred)

# results  Random Forest Classifier
print('Random Forest Classifier:')
print(f'Accuracy: {rf_accuracy:.2f}')
print('Classification Report:')
print(rf_classification_report)
print('Confusion Matrix:')
print(rf_confusion_matrix)

# confusion matrix Random Forest Classifier
plot_confusion_matrix(rf_confusion_matrix, classes=['Not Survived', 'Survived'],
                      normalize=True, title='Normalized Confusion Matrix - Random Forest Classifier')
plt.show()

#  Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_y_pred)
gb_classification_report = classification_report(y_test, gb_y_pred)
gb_confusion_matrix = confusion_matrix(y_test, gb_y_pred)

#  Gradient Boosting Classifier
print('\nGradient Boosting Classifier:')
print(f'Accuracy: {gb_accuracy:.2f}')
print('Classification Report:')
print(gb_classification_report)
print('Confusion Matrix:')
print(gb_confusion_matrix)

# Plot confusion matrix for Gradient Boosting Classifier
plot_confusion_matrix(gb_confusion_matrix, classes=['Not Survived', 'Survived'],
                      normalize=True, title='Normalized Confusion Matrix - Gradient Boosting Classifier')
plt.show()
