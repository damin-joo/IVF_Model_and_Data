# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Install Libraries

#********************************************************
# Data pre-processing
#........................
## 1.INIT
# Load file
data = pd.read_excel('IVF_tool/Data/original/ar-2017-2018.xlsx', sheet_name='Anonymised register')

# Basic info of data
print(f"Dataset Info: {data.info()}")     # column types and non-null counts
print(f"Dataset Shape: {data.shape}")
print(f"Dataset Head: {data.head()}")
print(f"Summary Stat: {data.describe}")

#........................
## 2.DATA CLEANING
# Handle Missing Values
## KNN imputation?
## Remove rows/columns with excessive missing values??


# Check for missing values in each column
print("\nMissing Values per Column:")
missing_count = data.isnull().sum()
missing_percentage = (missing_count / data.shape[0]) * 100

missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing Percentage (%)': missing_percentage
})

print(missing_summary)

# Some param only exists for certain circumstances
# missing_threshold = 0.3  # Define a threshold for missing values
# data = data.dropna(thresh=(1 - missing_threshold) * len(data), axis=1)  # Drop columns with too many missing values

# imputer = SimpleImputer(strategy="mean")  # Replace "mean" with "median" or "most_frequent" as needed
# data.iloc[:, :] = imputer.fit_transform(data)

#........................
# 3.DATA TRANSFORMATION
# Normalize data
scaler = StandardScaler()  # = Z-score normalization Or use MinMaxScaler()
scaled_features = scaler.fit_transform(data.select_dtypes(include=np.number))
data[data.select_dtypes(include=np.number).columns] = scaled_features

# Encoding categorical variables
encoder = OneHotEncoder(sparse=False)
categorical_columns = data.select_dtypes(include="object").columns
encoded_features = pd.DataFrame(encoder.fit_transform(data[categorical_columns]), columns=encoder.get_feature_names_out())
data = data.drop(columns=categorical_columns).join(encoded_features)

#........................
## 4. DATA REDUCTION

# Remove low-variance features
low_variance_threshold = 0.01
variance = data.var()
low_variance_cols = variance[variance < low_variance_threshold].index
data = data.drop(columns=low_variance_cols)

# Dimensionality reduction with PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
reduced_features = pca.fit_transform(data)
data = pd.DataFrame(reduced_features)

# Selecting only the required columns
required_columns = [
    'Patient age at treatment',
    'Total number of previous IVF cycles',
    'Total number of previous DI cycles',
    'Total number of previous pregnancies - IVF and DI',
    'Total number of previous live births - IVF or DI',
    'Causes of infertility - tubal disease',
    'Causes of infertility - ovulatory disorder',
    'Causes of infertility - male factor',
    'Causes of infertility - patient unexplained',
    'Causes of infertility - endometriosis',
    'Type of treatment - IVF or DI',
    'Specific treatment type',
    'Fresh cycle',
    'Frozen cycle',
    'Year of treatment',
    'Live birth occurrence',
    'Patient ethnicity',
    'Partner ethnicity',
    'Partner Type',
    'Partner age'
]

# Filter the dataframe to include only the required columns
data_filtered = data[required_columns]

#........................
## 5. DATA SPLIT
# Train-Test split (80-20, 70-30, 50-50)?
train, test = train_test_split(data, test_size=0.2, random_state=42)

# Validation set (split training data to training and validation sets)


#........................
## 6. SAVE DATA
# Save the cleaned dataset
data.to_csv("IVF_tool/Data/training_testing/cleaned_dataset.csv", index=False)
print("Data preparation complete. Cleaned dataset saved.")


