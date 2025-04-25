import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Sklearn and other required modules
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ***** LOAD DATA ************************************************
xls = pd.ExcelFile('IVF_tool/Data/original/ar-2017-2018.xlsx')
print("Available sheet names:", xls.sheet_names)

df = pd.read_excel(xls, sheet_name='Anonymised register')

# Print the number of columns
print(f"\nNumber of columns: {df.shape[1]}")

# ***** MISSING RATIO ************************************************
missing_data_ratio = df.isnull().mean() * 100
df_dropped = df.loc[:, missing_data_ratio <= 20]

print(f"\nColumns remaining after dropping columns with >20% missing data:")
print(df_dropped.columns)
print(f"\nNumber of columns remaining: {df_dropped.shape[1]}")

# ***** TARGET VARIABLE ************************************************
missing_target = df_dropped['Live birth occurrence'].isnull().sum()
print(f"\nMissing values in target column: {missing_target}")

df_model = df_dropped.dropna(subset=['Live birth occurrence'])
print(f"\nData shape after dropping rows with missing target: {df_model.shape}")
print("\nClass distribution in target variable:")
print(df_model['Live birth occurrence'].value_counts())

# ***** FILTER TREATMENT TYPE ************************************************
df_model = df_model[df_model['Type of treatment - IVF or DI'] == 'IVF']
df_model = df_model[df_model['Main reason for producing embroys storing eggs'] == 'Treatment - IVF']

# ***** ENCODING ************************************************
# Handle missing data in 'Total number of previous live births - IVF or DI'
categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns
print(f"\nCategorical columns:\n{categorical_cols.tolist()}")

df_encoded = df_model.copy()
label_encoder = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = label_encoder.fit_transform(df_encoded[col].astype(str))

# Combine cause of infertility into a single feature
col_cause = [
    'Causes of infertility - tubal disease',
    'Causes of infertility - ovulatory disorder',
    'Causes of infertility - male factor',
    'Causes of infertility - patient unexplained',
    'Causes of infertility - endometriosis'
]
conditions_cause = [
    (df_encoded[col_cause[0]] == 1),
    (df_encoded[col_cause[1]] == 1),
    (df_encoded[col_cause[2]] == 1),
    (df_encoded[col_cause[3]] == 1),
    (df_encoded[col_cause[4]] == 1)
]
values_cause = [1, 2, 3, 4, 5]
df_encoded['Cause of Infertility'] = np.select(conditions_cause, values_cause, default=0)

# ***** STANDARDIZATION ************************************************
X = df_encoded.drop(columns=['Live birth occurrence', 'Number of live births'], errors='ignore')
y = df_encoded['Live birth occurrence']

# Impute missing values for both features and target before splitting
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
y = imputer.fit_transform(y.values.reshape(-1, 1))
y = y.ravel()

# Check after imputation
print("Missing values in features after imputation:")
print(X.isnull().sum())

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Instead of subtracting overall minimum, subtract each column's minimum:
X = X - X.min()

print("✅ Standardization and non-negative transformation complete.")
print("Missing values in features after transformation:")
print(X.isnull().sum())

# ***** FEATURE SELECTION ************************************************
selector = SelectKBest(score_func=chi2, k=20)
selector.fit(X, y)
selected_features = X.columns[selector.get_support()]
chi2_scores = selector.scores_

chi2_scores_df = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_scores
}).sort_values(by='Chi2 Score', ascending=False)

chi2_scores_df['Rank'] = range(1, len(chi2_scores_df) + 1)

print("\nTop selected features ranked by Chi-Square score:")
print(chi2_scores_df[['Rank', 'Feature', 'Chi2 Score']].head(20))

plt.figure(figsize=(12, 6))
sns.barplot(data=chi2_scores_df.head(20).sort_values(by='Chi2 Score', ascending=False), x='Chi2 Score', y='Feature', palette='viridis')
plt.title('Top 20 Features by Chi-Square Score')
plt.xlabel('Chi2 Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ***** TRAIN/TEST SPLIT ************************************************
# Train-Test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Reset indices to ensure alignment
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = pd.Series(y_train, name='Live birth occurrence').reset_index(drop=True)
y_test = pd.Series(y_test, name='Live birth occurrence').reset_index(drop=True)

# print("Missing values in X_train:\n", X_train.isnull().sum())
# print("Missing values in y_train:\n", y_train.isnull().sum())

# Concatenate the DataFrames and Series
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# ***** SELECTED COLUMN FILTERING ************************************************
cols_to_keep_1 = [
    'Elective single embryo transfer',
    'Patient age at treatment', 
    'Partner age', 
    'Stimulation used',
    'Patient ethnicity',
    'Egg source',
    'Live birth occurrence'
]

cols_to_keep_2 = [
    'Elective single embryo transfer',
    'Patient age at treatment', 
    'Embryos transferred',   
    'Total embryos created',
    'Embryos stored for use by patient',
    'Partner age', 
    'Total embryos thawed',
    'Stimulation used',
    'Patient ethnicity',
    'Egg source',
    'Partner Type',
    'Partner ethnicity',
    'Live birth occurrence'
]

# Ensure train and test have the expected columns:
print("Train columns before filtering:", train.columns.tolist())

train = train[[col for col in cols_to_keep_1 if col in train.columns]]
test = test[[col for col in cols_to_keep_1 if col in test.columns]]

# # Final check before saving
# print("Missing values in train dataset by column:")
# print(train.isnull().sum())
# print("Missing values in test dataset by column:")
# print(test.isnull().sum())

# Overall check:
assert train.isnull().sum().sum() == 0, "Train dataset contains missing values!"
assert test.isnull().sum().sum() == 0, "Test dataset contains missing values!"

# ***** SAVE TO CSV ************************************************
train.to_csv('IVF_tool/Data/training_testing/train_2017-2018.csv', index=False)
test.to_csv('IVF_tool/Data/training_testing/test_2017-2018.csv', index=False)

print("✅ Filtered train and test datasets saved successfully.")

