from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


#display full rows of data
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)

np.set_printoptions(precision=3, suppress=True)
np.random.seed(72018)

def to_2d(array):
    return array.reshape(array.shape[0],-1)

def plot_exponential_data():
    data = np.exp(np.random.normal(size = 1000))
    plt.hist(data)
    plt.show()
    return data

def plot_square_normal_data():
    data = np.square(np.random.normal(loc = 5, size = 1000))
    plt.hist(data)
    plt.show()
    return data

#download dataset
path ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
respone = requests.get(path)
if respone.status_code == 200:
    boston_data = pd.read_csv(BytesIO(respone.content))
boston_data.head(5)
boston_data = boston_data.drop('Unnamed: 0', axis = 1)
boston_data.dtypes
boston_data.describe()
boston_data.isnull().any()
#CHAS has only 0 and 1 value: so it's catagory

feature_cols = [x for x in boston_data.columns if (x != 'MEDV') and ( x != 'CHAS') ]
X = boston_data[feature_cols]
y = boston_data['MEDV']

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = 0.2, random_state=42)
num_mask = X_train.dtypes == np.float64
float_cols = X_train.columns[num_mask]
skew_limit = 0.75
skew_vals = X_train[float_cols].skew()
skew_cols = (skew_vals.to_frame().rename(columns = {0:'skew'}).sort_values(by = 'skew', ascending = False).query('abs(skew)> {0}'.format(skew_limit)))
skew_cols
skew_cols_y = y_train.skew()

#plot skew cols
field = "CRIM"
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
X_train[field].hist(ax=ax_before, bins = 50)
X_train[field].apply(np.log1p).hist(ax=ax_after, bins = 50)
ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
fig.suptitle('Field "{}"'.format(field))

#transform
pt = PowerTransformer(method = "box-cox", standardize=True)
for col in skew_cols.index.tolist():
    if col != 'ZN':
        X_train[col] = pt.fit_transform((X_train[col]).values.reshape((-1,1)))
        X_test[col] = pt.transform(X_test[col].values.reshape((-1,1)))
        X_val[col] = pt.transform(X_val[col].values.reshape((-1,1)))
    else:
        X_train[col] = X_train[col].apply(np.log1p)
        X_test[col] = X_test[col].apply(np.log1p)
        X_val[col] = X_val[col].apply(np.log1p)
y_train = y_train.apply(np.log1p)
y_test = y_test.apply(np.log1p)
y_val = y_val.apply(np.log1p)

pf = PolynomialFeatures(degree=2, include_bias=False)
X_train = pf.fit_transform(X_train)
X_val = pf.transform(X_val)
X_test = pf.transform(X_test)

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)
X_val = s.transform(X_val)

#Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_r2_score = r2_score(np.exp(y_val), np.exp(lr.predict(X_val)))
print('My way skew transform - standard scale - PolynomialFeature - Linear Regression: ',lr_r2_score)

#Lasso CV
alphas2 = np.array([1e-5, 5e-5, 0.0001, 0.0005])
lassoCV = LassoCV(alphas =  alphas2, max_iter = 50000, cv = 3).fit(X_train, y_train)
lassoCV_r2_score_val = r2_score(np.exp(y_val), np.exp(lassoCV.predict(X_val)))
print('My way skew transform - standard scale - PolynomialFeature - lassoCV ', lassoCV_r2_score_val)

#IBM ways
print('IBM way')
y_col = "MEDV"
X = boston_data.drop(y_col, axis=1)
y = boston_data[y_col]
pf = PolynomialFeatures(degree=2, include_bias=False)
X_pf = pf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_pf, y, test_size=0.2,
                                                    random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = 0.2, random_state=42)
s = StandardScaler()
X_train_s = s.fit_transform(X_train)
X_test_s = s.transform(X_test)
X_val_s = s.transform(X_val)

#IBM Linear Regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
lr_r2_score = r2_score(np.exp(y_val), np.exp(lr.predict(X_val_s)))
print('IBM way skew transform - standard scale - PolynomialFeature - Linear Regression: ',lr_r2_score)

#IBM lasso
las = Lasso()
las.fit(X_train_s, y_train)
y_pred = las.predict(X_val_s)
r2_score(y_val, y_pred)
print('IBM standard scale - polyfeatures - lasso: ',r2_score(y_val, y_pred))

alphas2 = np.array([1e-5, 5e-5, 0.0001, 0.0005])
lassoCV = LassoCV(alphas =  alphas2, max_iter = 50000, cv = 3).fit(X_train_s, y_train)
lassoCV_r2_score_val = r2_score(y_val, lassoCV.predict(X_val_s))
print('IBM standard scale - polyfeatures -lassoCV: ',lassoCV_r2_score_val)