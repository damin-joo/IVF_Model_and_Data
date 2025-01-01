# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# Install Libraries
#pip install pandas
#pip install numpy
#pip install matplotlib
#pip install seaborn
#pip install scikit-learn

#********************************************************
# Data pre-processing
#........................
# Load the dataset
file_path = 'IVF_tool/Data/original/ar-2017-2018.xlsx'
sheet_name = 'Anonymised register'  # Update with the correct sheet name if necessary

# Read the dataset
data = pd.read_excel(file_path, sheet_name=sheet_name)
print("Dataset loaded successfully!")

# Basic Info
print(f"Dataset Shape: {data.shape}")
print(f"Columns: {data.columns}")

# print(f"Dataset Info: {data.info()}")     # column types and non-null counts
# print(f"Dataset Head: {data.head()}")
# print(f"Summary Stat: {data.describe()}")

#........................
## 2. DATA REDUCTION

# Selecting only the required columns
cols_to_keep = [
    'Patient age at treatment', 'Total number of previous IVF cycles',
    'Total number of previous DI cycles',
    'Total number of previous pregnancies - IVF and DI',
    'Total number of previous live births - IVF or DI',
    'Causes of infertility - tubal disease',
    'Causes of infertility - ovulatory disorder',
    'Causes of infertility - male factor',
    'Causes of infertility - patient unexplained',
    'Causes of infertility - endometriosis',
    'Egg donor age at registration', 'Sperm donor age at registration',
    'Donated embryo', 'Type of treatment - IVF or DI',
    'Egg source', 'Sperm source',
    'Live birth occurrence', 'Number of live births',
    'Patient ethnicity', 'Partner ethnicity',
    'Partner Type', 'Partner age'
]

# Filter the dataframe to include only the required columns
data = data[cols_to_keep].copy()

# Only include IVF Treatment
data = data[data['Type of treatment - IVF or DI'] == 'IVF']

print("Data Reduction Finished")

#........................
# 3.DATA TRANSFORMATION
# Normalize data
# Normalize the numerical columns
scaler = StandardScaler()  # or MinMaxScaler()
numerical_columns = data.select_dtypes(include=[np.number]).columns

data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Encoding categorical variables
# Replace >3, >5 values
data['Total number of previous live births - IVF or DI'].replace('>3', 4, inplace=True)
data['Total number of previous IVF cycles'].replace('>5', 6, inplace=True)
data['Total number of previous DI cycles'].replace('>5', 6, inplace=True)

# Categorize Age into Groups
age_map = {'18-34': 0, '35-37': 1, '38-39': 2, '40-42': 3, '43-44': 4, '45-50': 5, '999': 6}
donor_age_map = {'<= 20': 0, 'Between 21 and 25': 0, 'Between 26 and 30': 0, 'Between 31 and 35': 0, 'Between 36 and 40': 1, 'Between 41 and 45': 3, '>45': 5, '>35': 1}

data['Patient age at treatment'] = data['Patient age at treatment'].map(age_map).fillna(6).astype('int')
data['Partner age'] = data['Partner age'].map(age_map).fillna(6).astype('int')
data['Egg donor age at registration'] = data['Egg donor age at registration'].map(age_map).fillna(6).astype('int')
data['Sperm donor age at registration'] = data['Sperm donor age at registration'].map(age_map).fillna(6).astype('int')
# print(data['Patient age at treatment'].unique())


# # Update egg age based on source
# data['Egg age'] = np.where(
#     data['Egg source'] == 'Donor',
#     data['Egg donor age at registration'],
#     data['Patient age at treatment']  # Use patient age if egg source is not a donor
# )

# # Update sperm age based on source
# data['Sperm age'] = np.where(
#     data['Partner Type'] == 'Donor',
#     data['Sperm donor age at registration'],
#     data['Partner age']  # Use partner age if sperm source is not a donor
# )

# Get rid of unncessary columns
columns_to_drop = ['Type of treatment - IVF or DI']     # Number of live births necessary?
data = data.drop(columns=columns_to_drop)

# Replace NaN values with 0 for parameters
columns_to_fill = [
    'Total number of previous pregnancies - IVF and DI', 
    'Total number of previous live births - IVF or DI', 
    'Causes of infertility - tubal disease',
    'Causes of infertility - ovulatory disorder',
    'Causes of infertility - male factor',
    'Causes of infertility - endometriosis',
    'Live birth occurrence'
]
data[columns_to_fill] = data[columns_to_fill].fillna(0)

# Set Causes of infertility - patient unexplained to 1 if other causes = 0
condition = (
    (data['Causes of infertility - tubal disease'] == 0) &
    (data['Causes of infertility - ovulatory disorder'] == 0) &
    (data['Causes of infertility - male factor'] == 0) &
    (data['Causes of infertility - endometriosis'] == 0)
)
data.loc[condition, 'Causes of infertility - patient unexplained'] = 1

# for col in ['Egg source', 'Sperm source', 'Patient ethnicity', 'Partner ethnicity']:
#     print(f"{col}: {data[col].unique()}")

# Egg source & Sperm source : 0: Patient/Partner, 1: Donor
egg_src_map = {'Patient': 0, 'Donor': 1}
sperm_src_map = {'Partner': 0, 'Donor': 1}

data['Egg source'] = data['Egg source'].map(egg_src_map).fillna(0).astype('int')
data['Sperm source'] = data['Sperm source'].map(sperm_src_map).fillna(0).astype('int')

# Patient/Partner ethnicity : Black: 0, White: 1, Asian: 2, Other: 3
ethnicity_map = {'Black': 0, 'White': 1, 'Asian': 2, 'Other': 3}
data['Patient ethnicity'] = data['Patient ethnicity'].map(ethnicity_map).fillna(3).astype('int')
data['Partner ethnicity'] = data['Partner ethnicity'].map(ethnicity_map).fillna(3).astype('int')

# Partner Type : Male: 0, Female: 1, NaN: 2
partner_type_map = {'Male': 0, 'Female': 1}
data['Partner Type'] = data['Partner Type'].map(partner_type_map).fillna(1).astype('int')


print("Data Transformation Finished")

#........................
## 4.DATA CLEANING
# Handle Missing Values
## KNN imputation?

# Check for missing ratios
#Show missing values percentage
missing_data = data.isnull().mean() * 100
print(missing_data[missing_data > 0])

#Encode missing categorical data as 'Unknown'
categorical_columns = ['Specific treatment type', 'Sperm source']
for col in categorical_columns:
    if col in data.columns:
        data[col] = data[col].fillna('Unknown')

#Use KNN Imputation (K-Nearest Neighbours) for numerical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns
imputer = KNNImputer(n_neighbors=2)
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

data = data.loc[:, data.nunique() > 1]  # Drop constant columns

print("Data Cleaning Finished")

# missing_data = data.isnull().mean() * 100
# print(missing_data[missing_data > 0])

#........................
# ## 5. CORREALATIN MATRIX
# print(data.dtypes)

# Correlation matrix
correlation_matrix = data.corr()

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


#........................
## 6. SAVE DATA
# Save the cleaned dataset
output_path = f'IVF_tool/Data/training_testing/processed-2017-2018_v1.csv'
data.to_csv(output_path, index=False)
print(f"Cleaned and reduced dataset saved to {output_path}")

print("Data preparation complete. Cleaned dataset saved.")


