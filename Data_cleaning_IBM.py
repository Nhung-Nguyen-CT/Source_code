import pandas as pd
import numpy as np
import skillsnetwork

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import scipy
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

fig1 = sns.displot(housing['Lot Area'])
print('Skewness: %f' % housing['Lot Area'].skew())
log_transformed_lot = np.log(housing['Lot Area'])
fig2 = sns.displot(housing['Lot Area'])
print('Skewness: %f' % log_transformed_lot.skew())

#Handling the Duplicates
duplicate = housing[housing.duplicated(['PID'])]
#remove duplicated rows
dup_removed = housing.drop_duplicates()
#delete duplicated Order ID
dup_order = housing[housing.duplicated(['Order'])]
dup_order_removed = dup_order.drop_duplicates(subset = ['Order'])
#Handling with missing values
total_missing = housing.isnull().sum().sort_values(ascending = False)
total_missing_select = total_missing.head(20)
total_missing_select.plot(kind = 'bar', figsize = (8,6), fontsize = 10)
plt.xlabel('Columns', fontsize = 20)
plt.ylabel('Count', fontsize = 20)
plt.title('Total Missing values', fontsize = 20)
plt.show()
#delete rows with Lot Frontage missing values
housing.dropna(subset = ['Lot Frontage'])
#delete column: Lot Frontage:
housing.dropna(subset = ['Lot Frontage'])
#replace with median
median = housing['Lot Frontage'].median()
housing['Lot Frontage'].fillna(value = median, inplace = True)
housing['Lot Frontage'].isnull().sum()

#Feature Scaling
# normalization: Min-max scaling:
#values are shifted and rescaled, ranging from 0 to 1. by subtracting the min value and dividing by the max minus min
norm_data = MinMaxScaler().fit_transform(hous_num)
norm_data
# Standardization: to scale into standard distribution: shifted by mean, devide by standard deviation
scaled_data = StandardScaler().fit_transform(hous_num)
scaled_data
#standardization for 1 column:
hous_num_price = np.asarray(hous_num['SalePrice'])
scaled_price = StandardScaler().fit_transform(hous_num_price[:,np.newaxis])

#Handling with outliers
sns.boxplot(x=housing['Lot Area'])
sns.boxplot(x=housing['SalePrice'])
price_area = housing.plot.scatter(x='Gr Liv Area',
                      y='SalePrice')
plt.show()
#deleting outliers
housing.sort_values(by = 'Gr Liv Area', ascending = False)[:2]
outliers_dropped = housing.drop(housing.index[[1499,2181]])
new_plot = outliers_dropped.plot.scatter(x='Gr Liv Area',
                                         y='SalePrice')
plt.show()
#finding outlier by z-score
housing['L_stats'] = stats.zscore(housing['Low Qual Fin SF'])
outliers = housing[housing['L_stats'] > 3]
housing_removed_outliers = housing.drop(outliers.index)