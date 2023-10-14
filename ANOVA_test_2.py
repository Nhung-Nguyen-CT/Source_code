import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

file_path = r"./google_advance/ANOVA_marketing_sales_data.csv"
df = pd.read_csv(file_path)
print(df.head(10))
df.info()
print(df.describe( include = 'all'))
df['TV'].value_counts()
df['Influencer'].value_counts()

#Data exploration
#Box plot for mean TV - Sales
sns.boxplot(df, x = 'TV' , y = 'Sales', hue = 'TV')
plt.show()
sns.boxplot(df, x = 'Influencer' , y = 'Sales', hue = 'Influencer')
plt.show()

df.dropna(inplace = True)
print('Rows with missing values after dropping: ', len(df[df.isnull().any(axis = 1)]))

#Model building
model = ols(data = df, formula = "Sales ~ C(TV)").fit()
model.summary()
#check model assumption
#linear:
sns.barplot(data = df, x = 'TV', y = 'Sales', hue = 'TV')
plt.show()
#independence observation
#normality of resid, not normal
residuals = model.resid
fig = sns.histplot(residuals)
fig.set_title('Residuals Histogram')
fig.set_xlabel('Residual')
plt.show()
fig = sm.qqplot(residuals, line = 's')
plt.show()
#check Constant variance: constant
y_predict = model.predict(df['TV'])
fig = sns.scatterplot( x = y_predict , y = residuals)
fig.axhline(0)
fig.set_xlabel('Expected y')
fig.set_ylabel('Residuals')
fig.set_title('Scatter plot residuals by expected y')
plt.show()

#ANOVA test
print(sm.stats.anova_lm(model, typ = 2))
print(sm.stats.anova_lm(model, typ = 1))
print(sm.stats.anova_lm(model, typ = 3))

#post hoc test
tukey_hsd = pairwise_tukeyhsd(endog = df['Sales'], groups = df['TV'], alpha = 0.05)
print(tukey_hsd.summary())