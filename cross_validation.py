import pandas as pd
import skillsnetwork
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv'

#await skillsnetwork.download_dataset(URL)
data = pd.read_csv('encoded_car_data1.csv')
data.head()
data.dtypes.value_counts()
data.info()
#data preparation
X = data.drop(columns = ['price']).copy()
y = data['price'].copy()
#train test split
lr = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
lr.score(X_train, y_train)
lr.score(X_test, y_test)
r2_score(y_test, predicted)
mse = mean_squared_error(y_true = y_test, y_pred = predicted)
rmse = np.sqrt(mse)

#with pipe method
pipe = Pipeline([('ss', StandardScaler()), ('lr', LinearRegression())])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

#with Normalizer
pipe_1 = Pipeline([('nn', Normalizer()), ('lr', LinearRegression())])
pipe_1.fit(X_train, y_train)
predicted = pipe_1.predict(X_test)
pipe_1.score(X_test, y_test)
mse = mean_squared_error(y_true = y_test, y_pred = predicted)
rmse = np.sqrt(mse)

#One features
R_2 = []
features = list(X)
for feature in features:
    pipe.fit(X_train[[feature]],y_train)
    R_2.append(pipe.score(X_train[[feature]], y_train))
plt.bar(x = features, height = R_2)
plt.xticks(rotation = 90)
plt.ylabel('$R^2$')
plt.show()
best = features[np.argmax(R_2)]
best
pipe.fit(X_train[[best]], y_train)


# Without Standard Scaler
pipe2 = Pipeline([('lr', LinearRegression())])
R_2_2 = []
features2 = list(X)
for feature in features2:
    pipe.fit(X_train[[feature]],y_train)
    R_2_2.append(pipe.score(X_train[[feature]], y_train))
best2 = features2[np.argmax(R_2_2)]
pipe2.fit(X_train[[best2]], y_train)


#Cross validation score
N = len(X)
lr = LinearRegression()
scores = cross_val_score(estimator= lr, X = X, y = y, scoring = 'r2', cv = 3)


def display_scores(scores, print_=False):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(scores)

#with fold = 5 and scoring = neg_mean_squared_error
lr = LinearRegression()
scores2 = cross_val_score(estimator= lr, X = X, y = y, scoring = 'neg_mean_squared_error', cv = 5)

#KFOLD
#n_splits = 2
n_splits = 2
kf = KFold(n_splits= n_splits)
R_2 = np.zeros((n_splits, 1))
pipe = Pipeline([('ss', StandardScaler()), ('lr', LinearRegression())])
n = 0
for k, (train_index, test_index) in enumerate(kf.split(X, y)):
    print("TRAIN:", train_index)
    print("TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    n = +1
    R_2[k] = pipe.score(X_test, y_test)

#n_splits = 3
n_splits = 3
kf = KFold(n_splits=n_splits)
y = data['price'].copy()
X = data.drop(columns=['price'])
R_2 = np.zeros((n_splits, 1))
pipe = Pipeline([('ss', StandardScaler()), ('lr', LinearRegression())])
n = 0
for k, (train_index, test_index) in enumerate(kf.split(X, y)):
    print("TRAIN:", train_index)
    print("TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pipe.fit(X_train, y_train)
    n = +1
    R_2[k] = pipe.score(X_test, y_test)
R_2.mean()
