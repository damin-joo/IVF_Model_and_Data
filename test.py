import pandas as pd

data = pd.read_excel('ar-2017-2018.xlsx', sheet_name='Anonymised register')
print(data.columns)