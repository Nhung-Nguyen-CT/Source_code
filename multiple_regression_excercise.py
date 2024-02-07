import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split


penguins = sns.load_dataset("penguins")
penguins.head()
penguins.info()
penguins = penguins[["body_mass_g", "bill_length_mm", "sex", "species"]]
penguins.dropna(inplace = True)
penguins.reset_index(inplace=True, drop=True)


#subset X and Y variables
penguins_X = penguins[["bill_length_mm", "sex", "species"]]
penguins_Y = penguins["body_mass_g"]
X_train, X_test, y_train, y_test = train_test_split(penguins_X, penguins_Y , random_state=42,test_size=0.3)
#build model
ols_data = pd.concat([X_train, y_train],axis = 1)
ols_formula = "body_mass_g ~ bill_length_mm + C(sex) + C(species)"
OLS_model = ols(formula = ols_formula, data = ols_data)
model = OLS_model.fit()
print(model.summary())
