import pandas as pd
import numpy as np
import skillsnetwork

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats

#Load data: AMes Housing Data
URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv'
#await skillsnetwork.download_dataset(URL) : can't run directly on script, run from console
housing = pd.read_csv('Ames_Housing_Data1.tsv', sep = '\t')
housing.head(10)
housing.info()
housing['SalePrice'].describe()
#correlation finding
hous_num = housing.select_dtypes(include = ['float64', 'int64'])
hous_num_corr = hous_num.corr()['SalePrice'][:-1] #set up SalePrice at the end of table
top_features = hous_num_corr[abs(hous_num_corr)>0.5].sort_values(ascending = False)
print('There is {} strongly correlated values with SalePrice: \n{}'.format(len(top_features), top_features))
for i in range(0, len(hous_num.columns),5):
    sns.pairplot(data = hous_num,
                 x_vars = hous_num.columns[i:i+5],
                 y_vars = ['SalePrice'])
plt.show()

#Log Transformation
sp_untransformed = sns.displot(housing['SalePrice'])
print('Skewness: %f' % housing['SalePrice'].skew())
log_transformed = np.log(housing['SalePrice'])
sp_transformed = sns.displot(log_transformed)
print('Skewness: %f' % (log_transformed).skew())

fig1 = sns.distplot(housing['Lot Area'])
print('Skewness: %f' % housing['Lot Area'].skew())
log_transformed_lot = np.log(housing['Lot Area'])
fig2 = sns.distplot(housing['Lot Area'])
print('Skewness: %f' % log_transformed_lot.skew())

#Handling the Duplicates
duplicate = housing[housing.duplicated(['PID'])]