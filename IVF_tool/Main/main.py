# Import Libaries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


#********************************************************
# ML Training and Testing
#........................
## 1.INIT
# Load Dataset
data = pd.read_csv('IVF_tool/Data/training_testing/cleaned_dataset.csv', sheet_name='Anonymised register')

# Basic info of data
print(f"Dataset Info: {data.info()}")     # column types and non-null counts
print(f"Dataset Shape: {data.shape}")
print(f"Dataset Head: {data.head()}")
print(f"Summary Stat: {data.describe}")

#........................
## 2. DATA SPLIT
# Train-Test split (80-20, 70-30, 50-50)?
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Validation set (split training data to training and validation sets)

#........................
## 3. DEFINE MODEL
