import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#LOAD DATA
df= pd.read_excel('IVF_tool/Data/original/ar-2017-2018.xlsx', sheet_name='Anonymised register')

df.head(10)

#Summary Statistic
df.describe().T

#Show missing values percentage
df_na = (df.isnull().sum() / len(df)) * 100
df_null = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
#df_na = df_na.sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :df_null })
missing_data.head(25)

missing_data.info()

missing_data["Missing Ratio"]

data_with_null = missing_data[missing_data["Missing Ratio"]<=40]

data_with_null

data_not_null = df_na.drop(df_na[df_na > 0].index).sort_values(ascending=False)

filled_data = pd.DataFrame({'Missing Ratio' :data_not_null })
filled_data
filled_data.index

dataframe_used = filled_data + data_with_null
dataframe_used.index

cols = ['Causes of infertility - endometriosis',
       'Causes of infertility - male factor',
       'Causes of infertility - ovulatory disorder',
       'Causes of infertility - patient unexplained',
       'Causes of infertility - tubal disease',
       'Donated embryo', 'Egg source', 'Eggs thawed (0/1)',
       'Early outcome',
       'Elective single embryo transfer', 'Embryos stored for use by patient',
       'Embryos transferred', 'Fresh cycle', 'Fresh eggs collected',
       'Fresh eggs stored (0/1)', 'Frozen cycle', 'Live birth occurrence',
       'Number of live births', 'PGT-A treatment', 'PGT-M treatment', 'Partner age', 'Partner ethnicity',
       'Patient age at treatment', 'Patient ethnicity',
       'Specific treatment type', 'Sperm source', 'Stimulation used',
       'Total eggs mixed', 'Total embryos created', 'Total embryos thawed',
       'Total number of previous DI cycles',
       'Total number of previous IVF cycles',
       'Total number of previous live births - IVF or DI',
       'Total number of previous pregnancies - IVF and DI',
      ]

df_selected = df[[c for c in df.columns if c in cols]]
df_selected
df_selected.dtypes

#SELECT TARGET VARIABLE
#Split feature with target variable and add the feature of Live Birh Occurance
df_selected_target = df_selected[['Total number of previous live births - IVF or DI','Number of live births']].copy()
#Add live birth occurrence as it will add to our target value despite the missing ratio
df_selected_target= pd.concat([df_selected_target,df['Live birth occurrence']], axis=1)

df_selected_target['Total number of previous live births - IVF or DI'].unique()

# prompt: Convert all the nan in 'Total number of previous live births - IVF or DI' to 0

# Replace NaN values in 'Total number of previous live births - IVF or DI' with 0
df_selected_target['Total number of previous live births - IVF or DI'].fillna(0, inplace=True)

#TREAT Total number of previous live births - conceived through IVF or DI
#Transform >3 to certain number
df_selected_target['Total number of previous live births - IVF or DI'].replace('>3', 4, inplace=True)
#Transform data into integer
df_selected_target['Total number of previous live births - IVF or DI']= df_selected_target['Total number of previous live births - IVF or DI'].astype(str).astype(float)
#Transform dataframe to 0 and 1, with 1 is stated as successful live birth, while 0 is no recorded livebirth
df_selected_target['Total number of previous live births - IVF or DI'] = np.where(df_selected_target['Total number of previous live births - IVF or DI']==0, 0, 1)
#Check unique values of the transformed
df_selected_target['Total number of previous live births - IVF or DI'].unique()

#Transform number of live births as 0 and 1 as done previously
#No need to convert to int as it is already numeric
df_selected_target['Number of live births'] = np.where(df_selected_target['Number of live births']==0, 0, 1)
#Check unique values of the transformed
print(df_selected_target['Number of live births'].unique())

print(df_selected_target['Live birth occurrence'].unique())

df_selected_target

#COMBINE ALL COLUMNS, if there is any number 1 in a row, then it will be added as successful birth
# create a list of our conditions
conditions = [
    (df_selected_target['Total number of previous live births - IVF or DI'] == 1) |
    (df_selected_target['Number of live births'] == 1) |
    (df_selected_target['Live birth occurrence'] == 1)
    ]

    # create a list of the values we want to assign for each condition
values = [1]
    # create a new column and use np.select to assign values to it using our lists as arguments
df_selected_target['success or not'] = np.select(conditions, values, default=0)

#Take only the success or not column
df_selected_target.drop([ 'Number of live births', 'Live birth occurrence', 'Total number of previous live births - IVF or DI'], axis=1, inplace= True)

# Showing the distribution of TARGET
#Visualize target feature
plt.figure(figsize=(20,10))
sns.barplot(x=df_selected_target['success or not'].value_counts().index,
                 y=df_selected_target['success or not'].value_counts(normalize = False))


#Percentage success or not
print(df_selected_target['success or not'].value_counts(normalize = True) * 100)


#counting target variable
counting_target = df_selected_target['success or not'].value_counts(normalize = False).sum()
print(f'counting target = {counting_target}')

df_selected_target

#DIVIDE DATA FROM NUMERIC AND OBJECT
df_selected

cols_feature= ['Causes of infertility - endometriosis',
       'Causes of infertility - male factor',
       'Causes of infertility - ovulatory disorder',
       'Causes of infertility - patient unexplained',
       'Causes of infertility - tubal disease',
       'Donated embryo', 'Egg source', 'Eggs thawed (0/1)',
       'Early outcome',
       'Elective single embryo transfer', 'Embryos stored for use by patient',
       'Embryos transferred', 'Fresh cycle', 'Fresh eggs collected',
       'Fresh eggs stored (0/1)', 'Frozen cycle',
       'PGT-A treatment', 'PGT-M treatment', 'Partner age', 'Partner ethnicity',
       'Patient age at treatment', 'Patient ethnicity',
       'Specific treatment type', 'Sperm source', 'Stimulation used',
       'Total eggs mixed', 'Total embryos created', 'Total embryos thawed',
       'Total number of previous DI cycles',
       'Total number of previous IVF cycles',
       'Total number of previous pregnancies - IVF and DI',
       ]
df_selected_feature = df_selected[[c for c in df.columns if c in cols_feature]]

df_selected_feature.info()

# Define mapping for patient age ranges
Patmap = {'18-34': 0, '35-37': 1, '38-39': 2, '40-42': 3, '43-44': 4, '45-50': 5, '999': 6}

# Apply mapping and ensure integer type while handling unmapped values
df_selected_feature['Patient age at treatment'] = df_selected_feature['Patient age at treatment'].map(Patmap).fillna(6).astype('int')
#fillna(6) is added to fill the NaN values '999' with 6 before converting to int

df_selected_feature['Patient age at treatment']

df_selected_feature['Patient age at treatment'].unique()

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Patient age at treatment',
    data = df_selected_feature
)
plt.title('Distribution of Age')
# plt.show()

df_selected_feature['Partner age'].unique()

# Define mapping for partner age ranges
PARTmap = {'18-34': 0, '35-37': 1, '38-39': 2, '40-42': 3, '43-44': 4, '45-50': 5, '51-55': 6, '56-60':7, '999': 8}

# Apply mapping and ensure integer type while handling unmapped values
df_selected_feature['Partner age'] = df_selected_feature['Partner age'].map(PARTmap).fillna(8).astype('int')
#fillna(6) is added to fill the NaN values '999' with 8 before converting to int

df_selected_feature['Patient ethnicity'].unique()

 #Define mapping for patient ethnicities
TPEmap = {'Other': 0, 'Black': 1, 'White': 2, 'Asian': 3, 'Mixed': 4}

# Apply mapping and ensure integer type while handling unmapped values
df_selected_feature['Patient ethnicity'] = df_selected_feature['Patient ethnicity'].map(TPEmap).astype('int')

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Patient ethnicity',
    data = df_selected_feature
)
plt.title('Distribution of Patient ethnicity')
# plt.show()

df_selected_feature['Partner ethnicity'].unique()

#Define mapping for partner age ranges
TParEmap = {'Other': 0, 'Black': 1, 'White': 2, 'Asian': 3, 'Mixed': 4, 'Any other ethnicity': 0}

# Apply mapping and ensure integer type while handling unmapped values
df_selected_feature['Partner ethnicity'] = df_selected_feature['Partner ethnicity'].map(TParEmap)

df_selected_feature['Specific treatment type'].unique()

#Define mapping for specific treatment types
TSTTEmap = {'Unknown': 0, 'IVF': 1, 'IVF:Unknown': 1,  'ICSI:IVF': 2, 'ICSI:Unknown': 2, 'ICSI': 2, 'DI': 3}

# Apply mapping and ensure integer type while handling unmapped values
df_selected_feature['Specific treatment type'] = df_selected_feature['Specific treatment type'].map(TSTTEmap)

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Specific treatment type',
    data = df_selected_feature
)
plt.title('Distribution of Specific treatment type')
# plt.show()

#TREAT Total Number of Previous cycles IVF
TNPCmap = {0 : 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, '>5': 6}
df_selected_feature['Total number of previous IVF cycles'] = df_selected_feature['Total number of previous IVF cycles'].map(TNPCmap)

df_selected_feature['Total number of previous IVF cycles'].head ()

df_selected_feature['Total number of previous IVF cycles'].unique()

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Total number of previous IVF cycles',
    data = df_selected_feature
)
plt.title('Distribution of Total Number of Previous IVF cycles')
# plt.show()

# Fill NaN values with 6 and convert the column to integers
TNPDCmap = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5 ,'>5':6}
df_selected_feature['Total number of previous DI cycles'] = df_selected_feature['Total number of previous DI cycles'].map(TNPDCmap)

df_selected_feature['Total number of previous DI cycles'].head (57)

df_selected_feature['Total number of previous DI cycles'].unique()

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Total number of previous DI cycles',
    data = df_selected_feature
)
plt.title('Distribution of Total Number of Previous DI cycles')
# plt.show()

#Treat Total number of previous pregnancies, Both IVF and DI
# Fill NaN values with 6 and convert the column to integers
TNPPBIDmap = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5 ,'>5':6}
df_selected_feature['Total number of previous pregnancies - IVF and DI'] = df_selected_feature['Total number of previous pregnancies - IVF and DI'].fillna(0).map(TNPPBIDmap)

df_selected_feature['Total number of previous pregnancies - IVF and DI'].unique()

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Total number of previous pregnancies - IVF and DI',
    data = df_selected_feature
)
plt.title('Distribution of Total number of previous pregnancies - IVF and DI')
# plt.show()

#Treat Egg source
df_selected_feature['Egg source'].unique()

TESmap = {'Patient':0,'Donor':1}
df_selected_feature['Egg source'] = df_selected_feature['Egg source'].map(TESmap)

#Treat Sperm From
df_selected_feature['Sperm source'].unique()

TSSmap = {'Partner':0,'Donor':1}
df_selected_feature['Sperm source'] = df_selected_feature['Sperm source'].map(TSSmap)

#Treat Fresh Eggs Collected
#Do by converting to int and binning
df_selected_feature['Fresh eggs collected'].unique()

TFECmap = {0: 0, '1-5': 1, '6-10': 2, '11-15': 3,'16-20': 4, '21-25': 5,'26-30': 6, '31-35': 7, '36-40': 8, '>40':9}
df_selected_feature['Fresh eggs collected'] = df_selected_feature['Fresh eggs collected'].fillna(0).map(TFECmap)

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Fresh eggs collected',
    data = df_selected_feature
)
plt.title('Distribution of Fresh eggs collected')
# plt.show()

#Treat Total Eggs Mixed
df_selected_feature['Total eggs mixed'].unique()

TTEMmap = {0 : 0, '1-5': 1, '6-10': 2, '11-15': 3,'16-20': 4, '21-25': 5,'26-30': 6, '31-35': 7, '36-40': 8, '>40':9}
df_selected_feature['Total eggs mixed'] = df_selected_feature['Total eggs mixed'].map(TTEMmap)

#Treat Total Embryos Created
df_selected_feature['Total embryos created'].unique()

TTECmap = { 0 : 0, '1-5': 1, '6-10': 2, '11-15': 3,'16-20': 4, '21-25': 5,'26-30': 6, '>30':7}
df_selected_feature['Total embryos created'] = df_selected_feature['Total embryos created'].map(TTECmap)

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Total embryos created',
    data = df_selected_feature
)
plt.title('Distribution of Total embryos created')
# plt.show()

#Treat Embryos Stored For Use By Patient
df_selected_feature['Embryos stored for use by patient'].unique()

TESFUBPmap = { 0: 0, '1-5': 1, '6-10': 2, '11-15': 3,'16-20': 4, '>20':5}
df_selected_feature['Embryos stored for use by patient'] = df_selected_feature['Embryos stored for use by patient'].map(TESFUBPmap)

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Embryos stored for use by patient',
    data = df_selected_feature
)
plt.title('Distribution of Embryos stored for use by patient')
# plt.show()

#Treat Early Outcome
df_selected_feature['Early outcome'].unique()

#Treat Early Outcome
#Divide into None, Intrauterine Fetal Pulsation Seen,Biochemical Pregnancy Only, Misscarriage, & Ectopic, and decide accordingly
TEOmap = {'None': 0, 'Intrauterine Fetal Pulsation Seen': 1, 'Biochemical Pregnancy Only': 2, 'Miscarriage': 3, 'Ectopic/Hetrotopic': 4}
df_selected_feature['Early outcome'] = df_selected_feature['Early outcome'].fillna('None').map(TEOmap)

plt.figure(figsize = (8,8))
p = sns.histplot(
    x = 'Early outcome',
    data = df_selected_feature
)
plt.title('Distribution of Early outcome')
# plt.show()

#Treat Total Embryos Thawed
df_selected_feature['Total embryos thawed'].unique()

#Divide into None and the rest of the range
TTETmap = {0: 0, '1-5': 1, '6-10': 2, '>10': 3}
df_selected_feature['Total embryos thawed'] = df_selected_feature['Total embryos thawed'].map(TTETmap)

#COMBINE Cause of Infertility

# by Transform according to
# Tubal disease: 1, Ovulatory Disorder: 2 , Male Factor: 3, Patient Unexplained: 4, Endometriosis: 5
# And create new column while deleting the previous columns
col_cause =    ['Causes of infertility - tubal disease',
               'Causes of infertility - ovulatory disorder',
               'Causes of infertility - male factor',
               'Causes of infertility - patient unexplained',
               'Causes of infertility - endometriosis'
               ]

conditions_cause = [
    (df_selected_feature[col_cause[0]] == 1) ,
    (df_selected_feature[col_cause[1]] == 1) ,
    (df_selected_feature[col_cause[2]] == 1) ,
    (df_selected_feature[col_cause[3]] == 1) ,
    (df_selected_feature[col_cause[4]] == 1)
    ]

# create a list of the values we want to assign for each condition
values_cause = [1,2,3,4,5]

# create a new column and use np.select to assign values to it using our lists as arguments
df_selected_feature['Cause of Infertility'] = np.select(conditions_cause, values_cause, default=0)

# check unique number if data has met our condition
print(df_selected_feature['Cause of Infertility'].unique())

# delete previous column
df_selected_feature.drop(col_cause, axis=1, inplace= True)

#display updated Dataframe
df_selected_feature

#Save on new dataframe (to be further processed)
df_selected_new =df_selected_feature

df_selected_new.info()

#Advanced Feature Selection

#Create a feature correlation to select features
#Use a temporary dataframe so it will not interfere with our original dataframe to be used

df_feature_corr= df_selected_new.fillna(0)

df_feature_corr = df_feature_corr.astype(float)

fig, ax =plt.subplots(figsize=(20,25))
sns.heatmap(df_feature_corr.corr(), annot= True, linewidths=0.5, ax=ax, cbar=False, cmap=plt.cm.copper_r)
# plt.show()

fig, ax =plt.subplots(figsize=(5,5))
sns.heatmap(df_feature_corr[['Total number of previous pregnancies - IVF and DI','Total number of previous IVF cycles' ,'Total number of previous DI cycles']].corr(), annot= True, linewidths=0.5, ax=ax, cbar=False, cmap=plt.cm.copper_r)
# plt.show()

fig, ax =plt.subplots(figsize=(5,5))
sns.heatmap(df_feature_corr[['Stimulation used','Fresh cycle','Frozen cycle']].corr(), annot= True, linewidths=0.5, ax=ax, cbar=False, cmap=plt.cm.copper_r)
# plt.show()

df_feature_corr[['Stimulation used','Fresh cycle','Frozen cycle']].describe()

drop_features = ['Stimulation used',
                ]
df_feature_corr.drop(drop_features, axis=1, inplace= True)

#Check the dataframe of the removed feature
fig, ax =plt.subplots(figsize=(25,25))
sns.heatmap(df_feature_corr.corr(), annot= True, linewidths=0.5, ax=ax, cbar=False, cmap=plt.cm.copper_r)
# plt.show()

# Perform lasso reggresion in x_train and y to find feature importance
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=0.001)
lasso.fit(df_feature_corr, df_selected_target[df_selected_target.index.isin(df_feature_corr.index)])

drop_features_corrtarget = ['Early outcome']

df_selected_new.drop(drop_features_corrtarget, axis=1, inplace= True)

from sklearn.model_selection import train_test_split

y = df_selected_target #Target
X = df_selected_new #Features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train.columns

X_train.info()

for c in X_train.columns:
    X_train[c].fillna(X_train[c].mode()[0], inplace = True)

X_train.info()

y_train.info()

for c in X_test.columns:
    X_test[c].fillna(X_test[c].mode()[0], inplace = True)

X_test.info()
y_test.info()

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

#27
# cols_to_keep = [
#     'Patient age at treatment',
#     'Total number of previous IVF cycles',
#     'Total number of previous DI cycles',
#     'Total number of previous pregnancies - IVF and DI',
#     'Stimulation used',
#     'Donated embryo',
#     'Specific treatment type',
#     'PGT-M treatment',
#     'PGT-A treatment',
#     'Elective single embryo transfer',
#     'Egg source',
#     'Sperm source',
#     'Fresh cycle',
#     'Frozen cycle',
#     'Eggs thawed (0/1)',
#     'Fresh eggs collected',
#     'Fresh eggs stored (0/1)',
#     'Total eggs mixed',
#     'Total embryos created',
#     'Embryos transferred',
#     'Total embryos thawed',
#     'Embryos stored for use by patient',
#     'Patient ethnicity',
#     'Partner ethnicity',
#     'Partner age',
#     'Cause of Infertility',
#     'success or not'
# ]

# Drop
cols_to_keep = [
    'Patient age at treatment', 
    'Patient ethnicity', 
    'Partner age', 
    'Partner ethnicity', 
    'Cause of Infertility', 
    'Total number of previous IVF cycles', 
    'Total number of previous DI cycles', 
    'Total number of previous pregnancies - IVF and DI', 
    'Specific treatment type', 
    'Egg source',
    'Sperm source',
    'success or not'
    
    # 'Partner age',
    # 'Patient age at treatment',
    # 'Total number of previous IVF cycles', 
    # 'Cause of Infertility',
    # 'success or not'

#     'Partner age',
#     'Patient age at treatment',
#     'Total number of previous IVF cycles',
#     'Cause of Infertility',
#     'Patient ethnicity',
#     'Total number of previous pregnancies - IVF and DI',
#     'Partner ethnicity',
#     'success or not'
# ]
]

train = train[cols_to_keep]
test = test[cols_to_keep]

train.info()

# Save the filtered DataFrames to CSV
train.to_csv('IVF_tool/Data/training_testing/train_2017-2018.csv', index=False)
test.to_csv('IVF_tool/Data/training_testing/test_2017-2018.csv', index=False)

print("Filtered train and test datasets saved successfully.")
