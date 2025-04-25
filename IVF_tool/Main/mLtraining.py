import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Sklearn and other required modules
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ********************************************************
# ML Training and Testing
# ........................
## 1. INIT
# Load Dataset
train_data = pd.read_csv('IVF_tool/Data/training_testing/train_2017-2018.csv')
test_data = pd.read_csv('IVF_tool/Data/training_testing/test_2017-2018.csv')

# ........................
# 2. DATA SPLIT
# Define features (X) and target (y)
# X_train_full = train_data.drop(['success or not'], axis=1)
# y_train_full = train_data['success or not']
# X_test_full = test_data.drop(['success or not'], axis=1)
# y_test_full = test_data['success or not']

# v2
X_train_full = train_data.drop(['Live birth occurrence'], axis=1)
y_train_full = train_data['Live birth occurrence']
X_test_full = test_data.drop(['Live birth occurrence'], axis=1)
y_test_full = test_data['Live birth occurrence']

# Train-Test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# ........................
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

# ........................
# 4. RUN MODELS
def run():
    print("Running models...")
    LR()
    DecisionTree()
    XGBoost()
    TensorFlow()
    print("All models have been run.")
run()