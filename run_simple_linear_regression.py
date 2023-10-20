import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

file_path = rf"./google_advance/marketing_sales_data_multiple_regression.csv"
df = pd.read_csv(file_path)
print(df.head())

# descript statistic
df.describe(include = 'all')

#check and delete rows with missing values
print(df[df.isnull().any(axis = 1)],'    number of missing rows:',len(df[df.isnull().any(axis = 1)]))
df = df.dropna()
df = df.drop_duplicates()

#Exploration
sns.pairplot(df)
fig, axs = plt.subplots(1,2, figsize = (40,50))
fig.suptitle('Bar charts of TV - Sales and Influencers - Sales')
axs[0].set_title('TV - Sales')
axs[1].set_title('Influencer - Sales')
sns.barplot(ax = axs[0], data = df, x = df['TV'], y = df['Sales'], hue = df['TV'])
sns.barplot(ax = axs[1], data = df, x = df['Influencer'], y = df['Sales'], hue = df['Influencer'])
plt.show()

#Building model
x_df = sm.add_constant(df['Radio'].tolist())
y_df = df['Sales'].to_list()
result = sm.OLS(y_df,x_df).fit()
print(result.summary())
sns.regplot(x = df['Radio'], y = df['Sales'], data = df[['Radio','Sales']])
plt.show()

#Check assumptions
#Normality of resid by histogram
residuals = result.resid
fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()
#Normality of resid by qqplot
fig = sm.qqplot(residuals, line = 's')
plt.show()
#check homoscedasticity assumption:
fitted_y = result.predict(x_df)
fig = sns.scatterplot( y = residuals, x = fitted_y)
fig.axhline(0)
fig.set_xlabel("Fitted values")
fig.set_ylabel("Residuals")
fig.set_title("Scatter plot resid by fitted values")
plt.show()

print('end')
