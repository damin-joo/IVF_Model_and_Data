import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Load file
data = pd.read_excel('ar-2017-2018.xlsx', sheet_name='Anonymised register')

# Clean columns names by stripping extra spaces
data.columns = data.columns.str.strip()

# Define parameters
numerical_cols = [
    'Patient Age at Treatment', 'Total Number of Previous IVF Cycles', 
    'Total Number of Previous Pregnancies', 'Total Number of Previous Live Births', 
    'Egg Donor Age at Registration', 'Sperm Donor Age at Registration', 'Partner Age'
]

categorical_cols = [
    'Causes of Infertility', 'Egg Source', 'Sperm Source', 'Donated Embryo', 
    'Type of treatment - IVF or DI', 'Patient Ethnicity', 'Partner Ethnicity', 'Partner Type'
]

# Normalize numerical data
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Normalize the data
])

# Encode categorical data
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent category
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode the categories
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

# Split the data into features (X) and target (y)
X = data.drop('Type of treatment - IVF or DI', axis=1)  # Features: All columns except 'Type of Treatment'
y = data['Type of treatment - IVF or DI']  # Target: 'Type of Treatment'

# Update the target variable to binary (1 for 'IVF', 0 for others)
y = y.apply(lambda x: 1 if x == 'IVF' else 0)

# Apply the preprocessor to the entire dataset (X)
X_transformed = preprocessor.fit_transform(X)

# Convert the transformed data back to a DataFrame for better readability
# Get the column names for the transformed data
num_cols_transformed = numerical_cols
cat_cols_transformed = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_cols)

# Combine the column names
all_columns = list(num_cols_transformed) + list(cat_cols_transformed)

# Create a DataFrame with the transformed data
X_transformed_df = pd.DataFrame(X_transformed, columns=all_columns)

# Display the first few rows of the pre-processed data
print(X_transformed_df.head())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

