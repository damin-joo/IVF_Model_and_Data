# Import Libaries
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, roc_curve

# Install Libraries
#pip install tensorflow
#pip install xgboost

#********************************************************
# ML Training and Testing
#........................
## 1.INIT
# Load Dataset
# data = pd.read_csv('IVF_tool/Data/training_testing/processed-2017-2018.csv')
train_data = pd.read_csv('Data/training_testing/train_2017-2018.csv')
test_data = pd.read_csv('Data/training_testing/test_2017-2018.csv')

# Basic info of data
# print(f"Dataset Info: {data.info()}")     # column types and non-null counts
# print(f"Dataset Shape: {data.shape}")
# print(f"Dataset Head: {data.head()}")
# print(f"Summary Stat: {data.describe()}")

#........................
## 2. DATA SPLIT
# Split features(x) and labels(y)
# X = data.drop('Live birth occurrence', axis=1)
# y = data['Live birth occurrence']
# Define features (X) and target (y)
# Replace 'target_column' with the actual name of your target variable
X_train_full = train_data.drop(['success or not'], axis=1)
y_train_full = train_data['success or not']

X_test_full = test_data.drop(['success or not'], axis=1)
y_test_full = test_data['success or not']

# Train-Test split (80-20, 70-30, 50-50)?
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Validation set (split training data to training and validation sets)
validation = pd.read_excel('Data/validation/ar-2015-2016.xlsb', engine='pyxlsb')

#........................
## 2.5 EVALUATE MODEL
def evaluate_model(y_test, y_pred, y_pred_prob, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    logloss = log_loss(y_test, y_pred_prob)

    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend()
    plt.show()

#........................
## 3. DEFINE MODEL
def LR():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)  # Threshold for binary classification

    # Evaluate the model
    evaluate_model(y_test, y_pred, y_pred_prob, "Linear Regression")

def DecisionTree():
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_pred, y_pred_prob, "Decision Tree")

def XGBoost():
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_pred, y_pred_prob, "XGBoost")

def TensorFlow():
    # model = Sequential([
    #     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    #     Dense(64, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

    # y_pred_prob = model.predict(X_test).flatten()
    # y_pred = (y_pred_prob > 0.5).astype(int)

    # evaluate_model(y_test, y_pred, y_pred_prob, "TensorFlow")

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

    model.save("tensorflow_model.h5")  # Save in H5 format
    
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    evaluate_model(y_test, y_pred, y_pred_prob, "TensorFlow")


def Ensemble_DT_XGB():
    estimators = [
        ('dt', DecisionTreeClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]

    ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
    ensemble_model.fit(X_train, y_train)

    y_pred = ensemble_model.predict(X_test)
    y_pred_prob = ensemble_model.predict_proba(X_test)[:, 1]

    evaluate_model(y_test, y_pred, y_pred_prob, "Ensemble Model")


# Run Models
# LR()
# DecisionTree()
# XGBoost()
TensorFlow()
# Ensemble_DT_XGB()