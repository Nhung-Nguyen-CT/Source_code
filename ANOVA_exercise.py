import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

diamonds = sns.load_dataset("diamonds")
diamonds.head()
print(diamonds.info())
print(diamonds.describe(include = 'all'))
print('Missing value rows: ',len(diamonds[diamonds.isnull().any(axis = 1)]))
diamonds['color'].value_counts()
colorless = diamonds[diamonds["color"].isin(["E","F","H","D","I"])]
colorless = colorless[["color","price"]].reset_index(drop=True)
# Remove dropped categories of diamond color
colorless.color = colorless.color.cat.remove_categories(["G","J"])
colorless["color"].values

#take the logarithm of the price and insert it as the third columns
colorless.insert(2, "log_price", [math.log(price) for price in colorless["price"]])
colorless.dropna(inplace = True)
colorless.reset_index(inplace=True, drop=True)
colorless.head()
colorless.to_csv('diamonds.csv',index=False,header=list(colorless.columns))
diamonds = pd.read_csv("diamonds.csv")
diamonds.head()

sns.boxplot(x = "color", y = "log_price", data = diamonds, hue = 'color')
plt.show()

# Construct simple linear regression model, and fit the model
model = ols(formula = "log_price ~ C(color)", data = diamonds).fit()
print(model.summary())

# Run one-way ANOVA
print(sm.stats.anova_lm(model, typ = 2))
print(sm.stats.anova_lm(model, typ = 1))
print(sm.stats.anova_lm(model, typ = 3))

#two way ANOVA
diamonds = sns.load_dataset("diamonds")
diamonds2 = diamonds[["color","cut","price"]]
diamonds2 = diamonds2[diamonds2["color"].isin(["E","F","H","D","I"])]
diamonds2.color= diamonds2.color.cat.remove_categories(["G","J"])
diamonds2 = diamonds2[diamonds2['cut'].isin(['Ideal',"Premium","Very Good"])]
diamonds2.cut = diamonds2.cut.cat.remove_categories(["Good","Fair"])
diamonds2.dropna(inplace = True)
diamonds2.reset_index(inplace = True, drop=True)
diamonds2.insert(3,"log_price",[math.log(price) for price in diamonds2["price"]])
diamonds2.head()
#save to csv
diamonds2.to_csv('diamonds2.csv',index=False,header=list(diamonds2.columns))
diamonds2.head()
#construct multiple linear regression:
model2 = ols(formula = "log_price ~ C(color) + C(cut)", data = diamonds2).fit()
print(model2.summary())
#Run 2 way ANOVA
print(sm.stats.anova_lm(model2, typ = 2))
print(sm.stats.anova_lm(model2, typ = 1))
print(sm.stats.anova_lm(model2, typ = 3))
print()

#ANOVA post hoc test (Part II)
diamonds = pd.read_csv('diamonds.csv')
model = ols(formula = 'log_price ~ C(color)', data = diamonds).fit()
model.summary()
print(sm.stats.anova_lm(model, typ = 2))
#post hoc test
tukey_oneway = pairwise_tukeyhsd(endog = diamonds['log_price'], groups = diamonds['color'], alpha = 0.05)
tukey_oneway.summary()