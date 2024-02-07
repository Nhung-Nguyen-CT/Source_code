import pandas as pd
import matplotlib as plt
import statsmodels.api as sm
import seaborn as sns

file_path = rf"./google_advance/marketing_and_sales_data_evaluate_lr.csv"
df = pd.read_csv(file_path)
df.head()

# descript statistic
df.describe()

#find count of null in Sales
count_null_sales = len(df.loc[df['Sales'].isnull()])
percent_null_sales = count_null_sales/len(df)
#delete rows with missing Sales values
df.dropna(subset=['Sales'], inplace=True)
sns.histplot(df['Sales'])

#model building
sns.pairplot(df)
plt.show()
#build fit model
df.dropna(subset=['TV'], inplace=True)
x_df = sm.add_constant(df['TV'].tolist())
y_df = df['Sales'].to_list()
result = sm.OLS(y_df,x_df).fit()
result.summary()
#relationship X and Y
sns.scatterplot( x = df['TV'], y = df['Sales'])
plt.show()
#Independence of each observation
