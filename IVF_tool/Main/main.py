# Import Libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier, plot_importance


# Install Libraries
#pip install tensorflow
#pip install xgboost

#********************************************************
# ML Training and Testing
#........................
## 1.INIT
# Load Dataset
data = pd.read_csv('IVF_tool/Data/training_testing/processed-2017-2018.csv')

# Basic info of data
print(f"Dataset Info: {data.info()}")     # column types and non-null counts
print(f"Dataset Shape: {data.shape}")
print(f"Dataset Head: {data.head()}")
print(f"Summary Stat: {data.describe()}")

#........................
## 2. DATA SPLIT
# Split features(x) and labels(y)
X = data.drop('Live birth occurrence', axis=1)
y = data['Live birth occurrence']

# Train-Test split (80-20, 70-30, 50-50)?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Validation set (split training data to training and validation sets)
validation = pd.read_excel('IVF_tool/Data/validation/ar-2015-2016.xlsb')
#........................
## 3. DEFINE MODEL
def DecisionTree():
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")

    print(classification_report(y_test, y_pred))

def XGBoost():
    # Define and train the model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    # Plot
    plot_importance(model)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

def TensorFlow():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    history = model.fit(...)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.legend()
    plt.show()

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"TensorFlow Accuracy: {accuracy:.2f}")

def Ensemble():
    ensemble_model = VotingClassifier(estimators=[
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ], voting='soft')
    ensemble_model.fit(X_train, y_train)
    y_pred = ensemble_model.predict(X_test)
    print(f"Ensemble Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# Machine Learning models
DecisionTree()
XGBoost()
TensorFlow()


# Ensemble Learning models (Combined Basic ML Models)
Ensemble()