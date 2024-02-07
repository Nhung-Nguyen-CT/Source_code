import os
import numpy as np
import pandas as pd
import skillsnetwork

#URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/iris_data.csv'
#await skillsnetwork.download_dataset(URL)
data = pd.read_csv('iris_data.csv')
data.head(10)
data.shape[0]
data.columns.tolist()
data.dtypes
data['species'] = data.species.str.replace('Iris-', '')

stats_df = data.describe()
stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
out_fields = ['mean', '25%','50%','75%','range']
type(stats_df)