import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso, LassoCV, ElasticNet
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)

np.set_printoptions(precision=3, suppress=True)


def plot_dis(y, yhat):
    y_plot = pd.DataFrame({'y': y, 'y_predict': yhat})
    sns.displot(y_plot, kind="kde")
    plt.legend()
    plt.title('Actual vs Fitted Values')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()


def plot_coef(X, model, name = None):
    plt.bar(X.columns[2:], abs(model.coef_[2:]))
    plt.xticks(rotation = 90)
    plt.ylabel("$coefficients$")
    plt.title(name)
    plt.show()
    print("R^2 on training  data ", r2_score(np.exp(y_train), np.exp(model.predict(X_train))))
    print("R^2 on testing data ", r2_score(np.exp(y_test), np.exp(model.predict(X_test))))


data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv')
data.head()
print(data.info())
print(data.describe(include = 'all'))

def numerical_cols_detect(data):
    print(data.nunique().sort_values(ascending = True))
    nunique_limit = 10
    numerical_cols = data.nunique().to_frame().rename(columns={0: 'nunique'}).sort_values(by ='nunique',ascending = True).query('nunique > {0}'.format(nunique_limit))
    return numerical_cols

X = data.drop('price', axis = 1)
y = data.price

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.1, random_state= 42)
X_test.shape
X_train.shape

def skew_detect(X_train, y_train): #have to improve, detect encoded features in float number format
    num_mask = X_train.dtypes == np.float64
    float_cols = X_train.columns[num_mask]
    skew_right_limit = 0.75
    skew_left_limit = - 0.75
    skew_vals = X_train[float_cols].skew()
    skew_right_X_cols = (skew_vals.to_frame().rename(columns = {0:'skew'}).sort_values(by = 'skew', ascending = False).query('skew> {0}'.format(skew_right_limit)))
    skew_left_X_cols = (skew_vals.to_frame().rename(columns={0: 'skew'}).sort_values(by='skew', ascending=False).query('skew < {0}'.format(skew_left_limit)))
    skew_cols_y = y_train.skew()
    return skew_right_X_cols, skew_left_X_cols, skew_cols_y

#detect skewed numerical features
X_numerical_cols = numerical_cols_detect(data).index.to_list()
X_numerical_cols.remove('price')
skew_right_X, skew_left_X, skew_cols_y = skew_detect(X_train[X_numerical_cols], y_train)
skew_right_cols = skew_right_X.index.to_list()
print('X_train right skew')
print(skew_right_X)
print('X_train left skew')
print(skew_left_X)
print('y_train skew')
print(skew_cols_y)



#Transform
#pt = PowerTransformer(method = "box-cox", standardize=True)
#use Box-cox transform if heavy skewed
X_train[skew_right_cols] = X_train[skew_right_cols].apply(np.log1p)
X_test[skew_right_cols] = X_test[skew_right_cols].apply(np.log1p)
y_train = y_train.apply(np.log1p)
y_test = y_test.apply(np.log1p)
#for col in skew_right_cols:
#    X_train[col] = pt.fit_transform(X_train[col].values.reshape((-1,1)))
#    X_test[col] = pt.transform(X_test[col].values.reshape((-1,1)))
#y_train = y_train.apply(np.log1p)
#y_test = y_test.apply(np.log1p)



#Linear Reagression
lm = LinearRegression()
lm.fit(X_train, y_train)
predicted = lm.predict(X_test)
print("R^2 on training  data ", r2_score(np.exp(y_train), np.exp(lm.predict(X_train))))
print("R^2 on testing data ", r2_score(np.exp(y_test), np.exp(lm.predict(X_test))))
plot_dis(np.exp(y_test), np.exp(predicted))
plot_coef(X,lm,name="Linear Regression")

#Ridge Regression
rr = Ridge(alpha = 0.01)
rr.fit(X_train, y_train)
rr.predict(X_test)
print('R^2 on training data', r2_score(np.exp(y_train), np.exp(rr.predict(X_train))))
print('R^2 on test data', r2_score(np.exp(y_test), np.exp(rr.predict(X_test))))

#Compare lr vs rr
plot_coef(X, lm, name = 'Linear Regression')
plot_coef(X, rr, name = 'Ridge Regression')

#Increase alpha in Ridge
rr1 = Ridge(alpha=1)
rr1.fit(X_train, y_train)
plot_coef(X,rr1, name = 'Ridge Regression')

#GridSearchCV based on pipeline
input = [('polynomial', PolynomialFeatures(include_bias=False, degree = 2)), ('ss', StandardScaler()), ('model', Ridge(alpha=1)) ]
pipe = Pipeline(input)
pipe.fit(X_train, y_train)
predicted=pipe.predict(X_test)
pipe.score(X_test, y_test)

param_grid = {
    'polynomial__degree': [1,2,3,4],
    'model__alpha':[0.0001,0.001,0.01,0.1,1,10]
} # polynomial__degree:  name of step in pipe: polynomial; param of step
search = GridSearchCV(pipe, param_grid= param_grid, n_jobs= 2)
search.fit(X_train, y_train)
search

pd.DataFrame(search.cv_results_).head()

print("best_score_: ",search.best_score_)
print("best_params_: ",search.best_params_)
best=search.best_estimator_
predict = best.predict(X_test)
r2_score(np.exp(y_test), np.exp(predict))
best.fit(X,y)

#Perform grid search on the following features and plot the results
laCV = LassoCV(alpha=[0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000])
laCV.fit(X_train,y_train)
predicted = laCV.predict(X_test)
print('R^2 on training data', r2_score(np.exp(y_train), np.exp(laCV.predict(X_train))))
print('R^2 on test data', r2_score(np.exp(y_test), np.exp(laCV.predict(X_test))))

#Pipe with Lasso
Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)),('ss',StandardScaler() ), ('model',Lasso(alpha=1, tol = 0.2))]
pipe = Pipeline(Input)
pipe.fit(X_train, y_train)
print('Pipe - Lasso R^2 on training data', r2_score(np.exp(y_train), np.exp(pipe.predict(X_train))))
print('Pipe - Lasso R^2 on test data', r2_score(np.exp(y_test), np.exp(pipe.predict(X_test))))

#GridSeachCV with Lasso
param_grid = {
    'polynomial__degree': [1,2,3,4,5],
    'model__alpha':[0.0001,0.001,0.01,0.1,1,10]
}
search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X_train, y_train)
best = search.best_estimator_
print('GridSearchCV- Lasso R^2 on training data', r2_score(np.exp(y_train), np.exp(best.predict(X_train))))
print('GridSearchCV - Lasso R^2 on test data', r2_score(np.exp(y_test), np.exp(best.predict(X_test))))