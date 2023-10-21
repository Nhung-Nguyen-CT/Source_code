import numpy as np
import  os
import pandas as pd
import skillsnetwork
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

#URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/iris_data.csv'

#await skillsnetwork.download_dataset(URL)
data = pd.read_csv('iris_data.csv')
data.head()

#set up full colums full display
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


#remove name of species contain start with Iris
data['species'] = data.species.str.replace('Iris-', '')

# count every species
data.species.value_counts()

#find mean, range, median, quartile of numeric variables
stats_df = data.describe()
stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']
out_fields = ['mean','25%','50%','75%','range']
stats_df = stats_df.loc[out_fields]
stats_df.rename({'50%':'median'}, inplace = True)
stats_df

#caculate mean and median for numeric variables
data.groupby('species').mean()
data.groupby('species').median()
data.groupby('species').agg([np.mean, np.median])

# If certain fields need to be aggregated differently, we can do:
agg_dict = {field: ['mean','median'] for field in data.columns if field != 'species'}
agg_dict['petal_length'] = 'max'
pprint(agg_dict)
data.groupby('species').agg(agg_dict)

#make a scatter plot of sepal_length and sepal_width using Matplotlib
ax = plt.axes()
ax.scatter( x = data.sepal_length, y = data.sepal_width)
ax.set( xlabel = 'Sepal Length (cm)', ylabel = 'Sepal Width (cm)', title = 'Sepal Length vs Width')
plt.show()

# histogram using Matplotlib
ax = plt.axes()
ax.hist(data['petal_length'], bins = 25)
ax.set(xlabel = 'Petal length (cm)', ylabel = 'Frequency', title = 'Distribution of Petal Length')
plt.show()

#many histogram in the same axes using Seaborn
sns.set_context('notebook')
ax = data.plot.hist(bins = 25, alpha = 0.5)
ax.set_xlabel ('Size (cm)')
# seperate plots
axList = data.hist(bins = 25)

#boxplot of each petal and sepal by species
data.boxplot(by = 'species')

#stack data
plot_data = (data
             .set_index('species')
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns={0:'size', 'level_1':'measurement'})
            )

plot_data.head(20)

sns.set_style('white')
sns.set_context('notebook')
sns.set_palette('dark')

#plot boxplot of stacked data
f = plt.figure(figsize=(6,4))
sns.boxplot(x='measurement', y='size',
            hue='species', data=plot_data);

#pair plot use Seaborn, color by species
sns.set_context('talk')
sns.pairplot(data, hue='species');