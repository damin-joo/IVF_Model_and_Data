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

# Manually encode categorical variables
# Patient age at treatment
Patmap = {'18-34': 0, '35-37': 1, '38-39': 2, '40-42': 3, '43-44': 4, '45-50': 5, '999': 6}
df_encoded['Patient age at treatment'] = df_encoded['Patient age at treatment'].map(Patmap).fillna(6).astype('int')

# Total number of previous IVF cycles
TNPCmap = {0 : 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, '>5': 6}
df_encoded['Total number of previous IVF cycles'] = df_encoded['Total number of previous IVF cycles'].map(TNPCmap)

# Total number of previous DI cycles
TNPDCmap = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5 ,'>5':6}
df_encoded['Total number of previous DI cycles'] = df_encoded['Total number of previous DI cycles'].map(TNPDCmap)

# Specific treatment type
TSTTEmap = {'Unknown': 0, 'IVF': 1, 'IVF:Unknown': 1,  'ICSI:IVF': 2, 'ICSI:Unknown': 2, 'ICSI': 2, 'DI': 3}
df_encoded['Specific treatment type'] = df_encoded['Specific treatment type'].map(TSTTEmap)

# Egg source
TESmap = {'Patient':0,'Donor':1}
df_encoded['Egg source'] = df_encoded['Egg source'].map(TESmap)

# Sperm source
TSSmap = {'Partner':0,'Donor':1}
df_encoded['Sperm source'] = df_encoded['Sperm source'].fillna('Partner').map(TSSmap)

# Fresh eggs collected
TFECmap = {0: 0, '1-5': 1, '6-10': 2, '11-15': 3,'16-20': 4, '21-25': 5,'26-30': 6, '31-35': 7, '36-40': 8, '>40':9}
df_encoded['Fresh eggs collected'] = df_encoded['Fresh eggs collected'].fillna(0).map(TFECmap)

# Total eggs mixed
TTEMmap = {0 : 0, '1-5': 1, '6-10': 2, '11-15': 3,'16-20': 4, '21-25': 5,'26-30': 6, '31-35': 7, '36-40': 8, '>40':9}
df_encoded['Total eggs mixed'] = df_encoded['Total eggs mixed'].map(TTEMmap)

# Total embryos created
TTECmap = { 0 : 0, '1-5': 1, '6-10': 2, '11-15': 3,'16-20': 4, '21-25': 5,'26-30': 6, '>30':7}
df_encoded['Total embryos created'] = df_encoded['Total embryos created'].map(TTECmap)

# Total embryos thawed
TTETmap = {0: 0, '1-5': 1, '6-10': 2, '>10': 3}
df_encoded['Total embryos thawed'] = df_encoded['Total embryos thawed'].map(TTETmap)

# Embryos stored for use by patient
TESFUBPmap = { 0: 0, '1-5': 1, '6-10': 2, '11-15': 3,'16-20': 4, '>20':5}
df_encoded['Embryos stored for use by patient'] = df_encoded['Embryos stored for use by patient'].map(TESFUBPmap)

# Patient ethnicity
TPEmap = {'Other': 0, 'Black': 1, 'White': 2, 'Asian': 3, 'Mixed': 4}
df_encoded['Patient ethnicity'] = df_encoded['Patient ethnicity'].map(TPEmap).astype('int')

# Partner ethnicity
TParEmap = {'Other': 0, 'Black': 1, 'White': 2, 'Asian': 3, 'Mixed': 4, 'Any other ethnicity': 0}
df_encoded['Partner ethnicity'] = df_encoded['Partner ethnicity'].map(TParEmap)

# Partner Type
partner_type_map = {'Male': 0,'Female': 1,'Surrogate': 2,'Unknown': 3}
df_encoded['Partner Type'] = df_encoded['Partner Type'].fillna('Unknown').map(partner_type_map)

# Partner age
PARTmap = {'18-34': 0, '35-37': 1, '38-39': 2, '40-42': 3, '43-44': 4, '45-50': 5, '51-55': 6, '56-60':7, '999': 8}
df_encoded['Partner age'] = df_encoded['Partner age'].map(PARTmap).fillna(8).astype('int')

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
# Drop the original cause columns
df_encoded.drop(columns=['Main reason for producing embroys storing eggs', 'Type of treatment - IVF or DI', 'Number of live births'], inplace=True)

# Separate features and target variable
X = df_encoded.drop(['Live birth occurrence'], axis=1)
y = df_encoded['Live birth occurrence']

# Split into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reset indices to ensure proper alignment
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = pd.Series(y_train, name='Live birth occurrence').reset_index(drop=True)
y_test = pd.Series(y_test, name='Live birth occurrence').reset_index(drop=True)

# ***** FEATURE SELECTION ************************************************
selector = SelectKBest(score_func=chi2, k=20)
selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]
chi2_scores = selector.scores_

chi2_scores_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Chi2 Score': chi2_scores
}).sort_values(by='Chi2 Score', ascending=False)
chi2_scores_df['Rank'] = range(1, len(chi2_scores_df) + 1)

print("\nTop selected features ranked by Chi-Square score:")
print(chi2_scores_df[['Rank', 'Feature', 'Chi2 Score']].head(20))

plt.figure(figsize=(12, 6))
sns.barplot(data=chi2_scores_df.head(20).sort_values(by='Chi2 Score', ascending=False), 
            x='Chi2 Score', y='Feature', palette='viridis')
plt.title('Top 20 Features by Chi-Square Score')
plt.xlabel('Chi2 Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ***** TRAIN/TEST SPLIT ************************************************
# Fit the scaler on the training data and transform both train and test sets.
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# ----- Step 6: Concatenate Scaled Features with Target Variable -----
train_final = pd.concat([X_train, y_train], axis=1)
test_final = pd.concat([X_test, y_test], axis=1)

print("Final training set shape:", train_final.shape)
print("Final test set shape:", test_final.shape)

# ***** SELECTED COLUMN FILTERING ************************************************
cols_to_keep_1 = [
    'Patient age at treatment',
    'Partner age',
    'Elective single embryo transfer',
    'Total number of previous IVF cycles',
    'Stimulation Used',
    'Egg source',
    'Live birth occurrence'
]

cols_to_keep_2 = [
    'Patient age at treatment',
    'Partner age',
    'Elective single embryo transfer',
    'Embryos stored for use by patient',
    'Embryos transferred',   
    'Total embryos created',
    'Total eggs mixed',
    'Total embryos thawed', 
    'Total number of previous IVF cycles',
    'Stimulation Used',
    'Egg source',
    'Live birth occurrence'
]

cols_to_keep = [
    'Patient age at treatment',
    'Patient ethnicity',
    'Partner age',
    'Partner ethnicity',
    'Total number of previous IVF cycles',
    'Stimulation Used',
    'Elective single embryo transfer',  
    'Egg source',
    'Sperm source',
    'Live birth occurrence'
]

# Ensure train and test have the expected columns:
print("Train columns before filtering:", train_final.columns.tolist())

train = train_final[[col for col in cols_to_keep if col in test_final.columns]]
test = test_final[[col for col in cols_to_keep if col in test_final.columns]]

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

