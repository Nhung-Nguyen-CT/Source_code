from tqdm import tqdm
import skillsnetwork
import numpy as np
import pandas as pd
from itertools import accumulate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, load_wine

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def plot_dis(y, yhat):
    plt.figure()
    #ax1 = sns.histplot(y, kde=True, binwidth=0, color="r", label="Actual Value")
    #sns.histplot(yhat, kde=True, binwidth=0, color="b", label="Fitted Values", ax=ax1)
    ax1 = sns.kdeplot(y,  label="Actual Value")
    sns.kdeplot(yhat,  label="Fitted Values", ax=ax1)
    plt.legend()

    plt.title('Actual vs Fitted Values')
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


def get_R2_features(model, test=True):
    # X: global
    features = list(X)
    features.remove("three")

    R_2_train = []
    R_2_test = []

    for feature in features:
        model.fit(X_train[[feature]], y_train)
        R_2_test.append(model.score(X_test[[feature]], y_test))
        R_2_train.append(model.score(X_train[[feature]], y_train))

    plt.bar(features, R_2_train, label="Train")
    plt.bar(features, R_2_test, label="Test")
    plt.xticks(rotation=90)
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()
    print("Training R^2 mean value {} Testing R^2 mean value {} ".format(str(np.mean(R_2_train)), str(np.mean(R_2_test))))
    print("Training R^2 max value {} Testing R^2 max value {} ".format(str(np.max(R_2_train)), str(np.max(R_2_test))))


#URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv'
#await skillsnetwork.download_dataset(URL)
data = pd.read_csv('encoded_car_data.csv')
data.info()
# sns.lmplot( x = 'curbweight', y = 'price', data = data, order = 2)
# sns.lmplot(x = 'carlength', y = 'price', data = data, order = 2)
# sns.lmplot(x = 'horsepower', y = 'price', data = data, order = 2)
X = data.drop('price', axis = 1)
y = data.price
num_cols_X = X.columns[X.dtypes != np.object_]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
print("Number of test samples:", X_test.shape[0])
print("Number of training samples:", X_train.shape[0])
#Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
predict = lm.predict(X_test)
print("R^2 on training  data ",lm.score(X_train, y_train))
print("R^2 on testing data ",lm.score(X_test,y_test))
plot_dis(y_test,predict)
{col:coef for col,coef in zip(X.columns, lm.coef_)}
# plt.bar(X.columns[2:],abs(lm.coef_[2:]))
# plt.xticks(rotation=90)
# plt.ylabel("$coefficients$")
# plt.show()
X_train1 = StandardScaler().fit_transform(X_train)
y_train1 = StandardScaler().fit_transform(np.asarray(y_train).reshape((-1,1)))
lm.fit(X_train1, y_train1)
X_test1 = StandardScaler().fit_transform(X_test)
y_test1 = StandardScaler().fit_transform(np.asarray(y_test).reshape((-1,1)))
predict1 = lm.predict(X_test1)
plot_dis(y_test1, predict1)

pipe = Pipeline([('ss',StandardScaler()),('lr',LinearRegression())])
get_R2_features(pipe)

#Polynnomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
X_train_poly.shape
X_test_poly.shape
lm = LinearRegression()
lm.fit(X_train_poly, y_train)
predict = lm.predict(X_test_poly)
print("R^2 on training data:", lm.score(X_train_poly, y_train))
print("R^2 on testing data:", lm.score(X_test_poly,y_test))

#Individual Features
pipe = Pipeline([('polynomial', PolynomialFeatures(include_bias = False)),('model', LinearRegression())])
pipe.fit(X_train,y_train)
pipe.fit(X_train,y_train)

#GridSearch and Pipeline
pipe = Pipeline([('polynomial', PolynomialFeatures(include_bias = False)),('model', LinearRegression())])
pipe.get_params().keys() #to get parameter list
param_grid = {
    "polynomial__degree": [1, 2, 3],
    "model__fit_intercept": [True, False]
}
search = GridSearchCV(pipe, param_grid, n_jobs=1)
search.fit(X_test, y_test)
best = search.best_estimator_
best.score(X_test,y_test)
predict = best.predict(X_test)
plot_dis(y_test, predict)

#calculate the  R2 using the object Pipeline with GridSearch for each individual features
features=list(X)
R_2_train=[]
R_2_test=[]

for feature in features:
    param_grid = {
    "polynomial__degree": [ 1, 2,3,4,5],
    "model__positive":[True, False]}
    Input=[ ('polynomial', PolynomialFeatures(include_bias=False,degree=2)), ('model',LinearRegression())]
    pipe=Pipeline(Input)
    print(feature)
    search = GridSearchCV(pipe, param_grid, n_jobs=2)
    search.fit(X_test[[feature]], y_test)
    best=search.best_estimator_

    R_2_test.append(best.score(X_test[[feature]],y_test))
    R_2_train.append(best.score(X_train[[feature]],y_train))


plt.bar(features,R_2_train,label="Train")
plt.bar(features,R_2_test,label="Test")
plt.xticks(rotation=90)
plt.ylabel("$R^2$")
plt.legend()
plt.show()
print("Training R^2 mean value {} Testing R^2 mean value {} ".format(str(np.mean(R_2_train)),str(np.mean(R_2_test))) )
print("Training R^2 max value {} Testing R^2 max value {} ".format(str(np.max(R_2_train)),str(np.max(R_2_test))) )
