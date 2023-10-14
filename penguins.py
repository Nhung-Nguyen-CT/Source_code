import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")
penguins.head()

penguins_sub = penguins[penguins["species"] != "Chinstrap"]
penguins_final = penguins_sub.dropna()
penguins_final.reset_index(inplace = True, drop = True)
sns.pairplot(penguins_final)
plt.show()

ols_data = penguins_final[["bill_length_mm","body_mass_g"]]
ols_formula = "body_mass_g ~ bill_length_mm"
OLS = ols(formula = ols_formula, data = ols_data)
model = OLS.fit()
model.summary()
sns.regplot(x = "bill_length_mm", y = "body_mass_g", data = ols_data)
plt.show()

X = ols_data["bill_length_mm"]
fitted_values = model.predict(X)
residuals = model.resid
fig = sns.histplot(residuals)
fig.set_xlabel("Residual value")
fig.set_title("Histogram of Residuals")
plt.show()

#check assumption for regression model:
# Check the normality assumption: by histogram of residuals or qq plot
fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()
import statsmodels.api as sm
fig = sm.qqplot(model.resid, line = 's')
plt.show()
#check homoscedasticity assumption:
fig = sns.scatterplot(x=fitted_values, y=residuals)
fig.axhline(0)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()