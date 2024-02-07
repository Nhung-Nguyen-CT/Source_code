import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np

file_path = rf"./google_advance/marketing_sales_data_multiple_regression.csv"
df = pd.read_csv(file_path)
print(df.head(10))
df.info()
print(' Number of missing data rows: ',len(df[df.isnull().any(axis = 1)]))
df = df.rename(columns={'Social Media': 'Social_media'})

print(df.describe(include = 'all'))
print('Unique of TV: ',df['TV'].unique())
print('Unique of Influencer: ',df['Influencer'].unique())

#visualization
sns.pairplot(df)
plt.show()
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (40,50))
fig.suptitle('Bar chart of TV - Sales and Influencer - Sales')
axs[0].set_title('TV - Sales')
axs[1].set_title('Influencer - Sales')
df.sort_values('Sales')
sns.barplot( ax = axs[0] , data = df, x = 'TV', y = 'Sales', hue = 'TV', order = df.sort_values('Sales').TV)
sns.barplot( ax = axs[1] , data = df, x = 'Influencer', y = 'Sales', hue = 'Influencer', order = df.sort_values('Sales').Influencer)
plt.show()
print('clearly relationship with sales: TV, Radio - not too clearly: Social media - no relationship: Influencer')

#building model
ols_data = df[['TV','Radio','Social_media','Sales']]
ols_formula = ' Sales ~ C(TV) + Radio + Social_media'
OLS_model = ols(data = ols_data, formula = ols_formula)
model = OLS_model.fit()
result1 = model.summary()
print(result1)
#rebuild model: excluding 'Social_media'
ols_data = df[['TV','Radio','Sales']]
ols_formula = ' Sales ~ C(TV) + Radio'
OLS_model = ols(data = ols_data, formula = ols_formula)
model = OLS_model.fit()
result1 = model.summary()
print(result1)

#check model assumption
#linearity: use pairplot and bar chart
#independencc observation
# normality residuals
residuals = model.resid
fig = sns.histplot(residuals)
fig.set_xlabel('Residuals')
fig.set_title('Histogram of residuals')
plt.show()
sm.qqplot(residuals, line = 's')
plt.show()
print('Not well normality')
#Constant variance
X_predict = ols_data[['TV','Radio']]
fitted_y = model.predict(X_predict)
fig = sns.scatterplot(x= fitted_y, y = residuals)
fig.axhline(0)
fig.set_xlabel("Fitted values")
fig.set_ylabel("Residuals")
fig.set_title("Scatter plot resid by fitted values")
plt.show()
print('Homoscedasticity met')
#No multicollinearity
df.sort_values('Radio')
sns.barplot(data = df, x = 'TV', y = 'Radio', hue = 'TV', order = df.sort_values('Radio').TV)
plt.show()
print('Correlation between TV and Radio')

#Results and evaluation:
