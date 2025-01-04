# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Install Libraries
#pip install pandas
#pip install numpy
#pip install matplotlib
#pip install seaborn
#pip install scikit-learn

#********************************************************
# Data pre-processing
#********************************************************
class IVFDataPreprocessor:
    def __init__(self, file_path, sheet_name, cols_to_keep, output_path):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.cols_to_keep = cols_to_keep
        self.output_path = output_path
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            print("Dataset loaded successfully!")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def reduce_data(self):
        # Select only the required columns
        self.data = self.data[self.cols_to_keep].copy()
        # Filter only IVF treatments
        self.data = self.data[self.data['Type of treatment - IVF or DI'] == 'IVF']
        self.data = self.data.drop(columns=['Type of treatment - IVF or DI'])
        print("Data Reduction Finished")

    def transform_data(self):
        # Normalize numerical columns
        scaler = MinMaxScaler()
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numerical_columns] = scaler.fit_transform(self.data[numerical_columns])

        # Replace specific categorical values
        self.data['Total number of previous live births - IVF or DI'].replace('>3', 4, inplace=True)
        self.data['Total number of previous IVF cycles'].replace('>5', 6, inplace=True)
        self.data['Total number of previous DI cycles'].replace('>5', 6, inplace=True)

        # Map categorical variables
        age_map = {'18-34': 0, '35-37': 1, '38-39': 2, '40-42': 3, '43-44': 4, '45-50': 5, '999': 6}
        donor_age_map = {'<= 20': 0, 'Between 21 and 25': 1, 'Between 26 and 30': 2, 'Between 31 and 35': 3, 'Between 36 and 40': 4, 'Between 41 and 45': 5, '>45': 6, '>35': 4}
        self.data['Patient age at treatment'] = self.data['Patient age at treatment'].map(age_map).fillna(6).astype('int')
        self.data['Partner age'] = self.data['Partner age'].map(age_map).fillna(6).astype('int')
        self.data['Egg donor age at registration'] = self.data['Egg donor age at registration'].map(age_map).fillna(6).astype('int')
        self.data['Sperm donor age at registration'] = self.data['Sperm donor age at registration'].map(age_map).fillna(6).astype('int')

        # Additional mappings
        egg_sperm_map = {'Patient': 0, 'Partner':0, 'Donor': 1}
        self.data['Egg source'] = self.data['Egg source'].map(egg_sperm_map).fillna(0).astype('int')
        self.data['Sperm source'] = self.data['Sperm source'].map(egg_sperm_map).fillna(0).astype('int')

        ethnicity_map = {'Black': 0, 'White': 1, 'Asian': 2, 'Other': 3}
        self.data['Patient ethnicity'] = self.data['Patient ethnicity'].map(ethnicity_map).fillna(3).astype('int')
        self.data['Partner ethnicity'] = self.data['Partner ethnicity'].map(ethnicity_map).fillna(3).astype('int')

        partner_type_map = {'Male': 0, 'Female': 1}
        self.data['Partner Type'] = self.data['Partner Type'].map(partner_type_map).fillna(1).astype('int')

        print("Data Transformation Finished")

    def clean_data(self):
        # Fill missing values with 0 for specific columns
        columns_to_fill = [
            'Total number of previous pregnancies - IVF and DI',
            'Total number of previous live births - IVF or DI',
            'Causes of infertility - tubal disease',
            'Causes of infertility - ovulatory disorder',
            'Causes of infertility - male factor',
            'Causes of infertility - endometriosis',
            'Live birth occurrence'
        ]
        self.data[columns_to_fill] = self.data[columns_to_fill].fillna(0)

        # Set 'patient unexplained' infertility to 1 if all other causes are 0
        condition = (
            (self.data['Causes of infertility - tubal disease'] == 0) &
            (self.data['Causes of infertility - ovulatory disorder'] == 0) &
            (self.data['Causes of infertility - male factor'] == 0) &
            (self.data['Causes of infertility - endometriosis'] == 0)
        )
        self.data.loc[condition, 'Causes of infertility - patient unexplained'] = 1

        # Use KNN imputation for numerical columns
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=2)
        self.data[numerical_columns] = imputer.fit_transform(self.data[numerical_columns])

        # Drop constant columns
        self.data = self.data.loc[:, self.data.nunique() > 1]
        print("Data Cleaning Finished")

    def save_data(self):
        self.data.to_csv(self.output_path, index=False)
        print(f"Cleaned and reduced dataset saved to {self.output_path}")

    def plot_correlation_matrix(self):
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()


#********************************************************
# Pipeline Execution
#********************************************************
file_path = 'IVF_tool/Data/original/ar-2017-2018.xlsx'
sheet_name = 'Anonymised register'
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

output_path = 'IVF_tool/Data/training_testing/processed-2017-2018.csv'

preprocessor = IVFDataPreprocessor(file_path, sheet_name, cols_to_keep, output_path)
preprocessor.load_data()
preprocessor.reduce_data()
preprocessor.transform_data()
preprocessor.clean_data()
preprocessor.save_data()
preprocessor.plot_correlation_matrix()

#********************************************************
# Validation and Modeling
#********************************************************
X = preprocessor.data.drop('Live birth occurrence', axis=1)
y = preprocessor.data['Live birth occurrence']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

print(f"Model Accuracy on Training Data: {rf.score(X_train, y_train):.2f}")
print(f"Model Accuracy on Testing Data: {rf.score(X_test, y_test):.2f}")

scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.2f}")

feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title("Feature Importance")
plt.show()

# Validate unique values and ranges
categorical_columns = ['Egg source', 'Sperm source', 'Patient ethnicity', 'Partner ethnicity']
for col in categorical_columns:
    print(f"Unique values in {col}: {preprocessor.data[col].unique()}")

numerical_columns = preprocessor.data.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    print(f"Range for {col}: {preprocessor.data[col].min()} to {preprocessor.data[col].max()}")

# Automated Tests
def test_reduce_data():
    preprocessor = IVFDataPreprocessor(file_path, sheet_name, cols_to_keep, output_path)
    preprocessor.load_data()
    preprocessor.reduce_data()
    assert 'Type of treatment - IVF or DI' not in preprocessor.data.columns
    assert preprocessor.data.shape[1] == len(cols_to_keep) - 1  # One column dropped

test_reduce_data()