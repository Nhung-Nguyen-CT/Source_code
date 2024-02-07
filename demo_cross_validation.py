import pandas as pd
import skillsnetwork
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import pickle


#URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/boston_housing_clean.pickle'
#await skillsnetwork.download_dataset(URL)
#a = pd.DataFrame()
#b = pd.DataFrame()
#c = pd.DataFrame()
#with open('file.pkl', 'wb') as file:
#    pickle.dump(a, file)
#    pickle.dump(b, file)
#    pickle.dump(c, file)

#with open('file.pkl', 'rb') as file:
#    a = pickle.load(file)
#    b = pickle.load(file)
#    c = pickle.load(file)
boston = pickle.load(open('boston_housing_clean.pickle', "rb" ))
boston.keys()
boston_data = boston['dataframe']
boston_description = boston['description']
boston_data.head()
boston_data.info()
boston_data.describe()

X = boston_data.drop('MEDV', axis = 1)
y = boston_data.MEDV

kf = KFold(shuffle=True, random_state=72018, n_splits=3)
for train_index, test_index in kf.split(X):
    print("Train index:", train_index[:10], len(train_index))
    print("Test index:",test_index[:10], len(test_index))
    print('')

scores = []
lr = LinearRegression()

#fit with non-scale
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = (X.iloc[train_index],
                                        X.iloc[test_index],
                                        y[train_index],
                                        y[test_index])
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score = r2_score(y_test.values, y_pred)
    scores.append(score)
print(scores)

#fit with scaled
cores = []
lr = LinearRegression()
s = StandardScaler()
for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = (X.iloc[train_index],
                                        X.iloc[test_index],
                                        y.iloc[train_index],
                                        y.iloc[test_index])
    X_train_s = s.fit_transform(X_train)
    lr.fit(X_train_s, y_train)
    X_test_s = s.transform(X_test)
    y_pred = lr.predict(X_test_s)
    score = r2_score(y_test, y_pred)
    scores.append(score)
print(scores)


#Pipeline and cross_val_predict
s = StandardScaler()
lr = LinearRegression()
estimator = Pipeline([('scaler',s),('regression', lr)])
predictions = cross_val_predict(estimator, X, y, cv = kf)
r2_score(y, predictions)
np.mean(scores)


#Hyperparameter tuning
#hyperparameter with scaler and lasso
alphas = np.geomspace(1e-9, 1e0, num = 10)
scores = []
coefs = []
scores = []
coefs = []
for alpha in alphas:
    las = Lasso(alpha=alpha, max_iter=100000)
    estimator = Pipeline([
        ("scaler", s),
        ("lasso_regression", las)])
    predictions = cross_val_predict(estimator, X, y, cv=kf)
    score = r2_score(y, predictions)
    scores.append(score)
Lasso(alpha=1e-6).fit(X, y).coef_
Lasso(alpha=1.0).fit(X, y).coef_
plt.figure(figsize=(10,6))
plt.semilogx(alphas, scores, '-o')
plt.xlabel('$\\alpha$')
plt.ylabel('$R^2$');
list(zip(alphas,scores))
#hyperparameter with scaler, polynomial and lasso
pf = PolynomialFeatures(degree=3)
scores1 = []
alphas1 = np.geomspace(0.06, 6.0, 20)
for alpha in alphas1:
    las = Lasso(alpha=alpha, max_iter=100000)
    estimator = Pipeline([
        ("scaler", s),
        ("make_higher_degree", pf),
        ("lasso_regression", las)])
    predictions = cross_val_predict(estimator, X, y, cv=kf)
    score = r2_score(y, predictions)
    scores1.append(score)
plt.semilogx(alphas1, scores1);
list(zip(alphas1,scores1))
#another tuning with Polynomial degree = 2
pf = PolynomialFeatures(degree=2)
scores2 = []
alphas2 = np.geomspace(1e-9, 1e0, 20)
for alpha in alphas2:
    las = Lasso(alpha=alpha, max_iter=100000)
    estimator = Pipeline([
        ("scaler", s),
        ("make_higher_degree", pf),
        ("lasso_regression", las)])
    predictions = cross_val_predict(estimator, X, y, cv=kf)
    score = r2_score(y, predictions)
    scores2.append(score)
plt.semilogx(alphas2, scores2);
list(zip(alphas2,scores2))
#Find the best alphas and degree of Polynomial
max(scores)
max(scores1)
max(scores2)
alphas2(np.argmax(scores2))

best_estimator = Pipeline([
                    ("scaler", s),
                    ("make_higher_degree", PolynomialFeatures(degree=2)),
                    ("lasso_regression", Lasso(alpha=0.012))])

best_estimator.fit(X, y)
best_estimator.score(X, y)
# => why modling without Kfold. When tuning with Kfold

#GRID SEARCH CV
estimator = Pipeline([("scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures()),
        ("ridge_regression", Ridge())])
params = {
    'polynomial_features__degree': [1, 2, 3],
    'ridge_regression__alpha': np.geomspace(4, 20, 30)
}
grid = GridSearchCV(estimator, params, cv=kf)
grid.fit(X, y)
grid.best_score_, grid.best_params_
y_predict = grid.predict(X)
# This includes both in-sample and out-of-sample
r2_score(y, y_predict)
grid.best_estimator_.named_steps['ridge_regression'].coef_
grid.cv_results_