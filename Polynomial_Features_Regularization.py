import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/X_Y_Sinusoid_Data.csv")
data.head()
X_real = np.linspace(0, 1.0, 100)
Y_real = np.sin(2* np.pi * X_real)

sns.set_style('white')
sns.set_context('talk')
sns.set_palette('dark')

ax = data.set_index('x')['y'].plot(ls='', marker='o', label='data')
ax.plot(X_real, Y_real, ls='--', marker='', label='real function')

ax.legend()
ax.set(xlabel='x data', ylabel='y data');
plt.show()

# Setup the polynomial features
degree = 20
pf = PolynomialFeatures(degree)
lr = LinearRegression()

# Extract the X- and Y- data from the dataframe
X_data = data[['x']]
Y_data = data['y']

# Create the features and fit the model
X_poly = pf.fit_transform(X_data)
lr = lr.fit(X_poly, Y_data)
Y_pred = lr.predict(X_poly)

# Plot the result
plt.plot(X_data, Y_data, marker='o', ls='', label='data', alpha=1)
plt.plot(X_real, Y_real, ls='--', label='real function')
plt.plot(X_data, Y_pred, marker='^', alpha=.5, label='predictions w/ polynomial features')
plt.legend()
ax = plt.gca()
ax.set(xlabel='x data', ylabel='y data');
plt.show()

#the ridge regression model
rr = Ridge(alpha = 0.001)
rr = rr.fit(X_poly, Y_data)
Y_pred_rr = rr.predict(X_poly)
#the lasso regression model
lassor = Lasso(alpha = 0.0001)
lassor = lassor.fit(X_poly, Y_data)
Y_pred_lr = lassor.predict(X_poly)

# The plot of the predicted values
plt.plot(X_data, Y_data, marker='o', ls='', label='data')
plt.plot(X_real, Y_real, ls='--', label='real function')
plt.plot(X_data, Y_pred, label='linear regression', marker='^', alpha=.5)
plt.plot(X_data, Y_pred_rr, label='ridge regression', marker='^', alpha=.5)
plt.plot(X_data, Y_pred_lr, label='lasso  regression', marker='^', alpha=.5)
plt.legend()
ax = plt.gca()
ax.set(xlabel='x data', ylabel='y data')
plt.show()

coefficients = pd.DataFrame()
coefficients['linear regression'] = lr.coef_.ravel()
coefficients['ridge regression'] = rr.coef_.ravel()
coefficients['lasso regression'] = lassor.coef_.ravel()
coefficients = coefficients.applymap(abs)
coefficients.describe()

#plot coefficients of 3 regression
#color set up
colors = sns.color_palette()
#set up dual y-axes
ax1 = plt.axes()
ax2 = ax1.twinx()
#Plot linear regression on the first axes and ridge, lasso on the second axes
ax1.plot(lr.coef_.ravel(), color = colors[0], marker = 'o', label = 'linear regression')
ax2.plot(rr.coef_.ravel(), color = colors[1], marker = 'o', label = 'ridge regression')
ax2.plot(lassor.coef_.ravel(), color = colors[2], marker='o', label = 'lasso regression')
#custom axes scales
ax1.set_ylim(-2e14, 2e14)
ax2.set_ylim(-25, 25)
#combine the legends
h1, l1 = ax1.get_legend_handles_labels()
h2, l2= ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1+l2, loc = 'best')
ax1.set(xlabel='coefficients',ylabel='linear regression')
ax2.set(ylabel='ridge and lasso regression')
ax1.set_xticks(range(len(lr.coef_)))
plt.show()

#new section
#by myself code
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/Ames_Housing_Sales.csv")
data.head(10)
feature_cols = [x for x in data.columns if x != 'SalePrice']
X = data[feature_cols]
y = data['SalePrice']
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = 0.2, random_state=42)
num_mask = X_train.dtypes == np.float64
float_cols = X_train.columns[num_mask]
skew_limit = 0.75
skew_vals = X_train[float_cols].skew()
skew_cols = (skew_vals.to_frame().rename(columns = {0:'skew'}).sort_values(by = 'skew', ascending = False).query('abs(skew)> {0}'.format(skew_limit)))
skew_cols
skew_cols_y = y_train.skew()

field = "BsmtFinSF1"
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
X_train[field].hist(ax=ax_before, bins = 50)
X_train[field].apply(np.log1p).hist(ax=ax_after, bins = 50)
ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
fig.suptitle('Field "{}"'.format(field))

#transform
for col in skew_cols.index.tolist():
    X_train[col] = X_train[col].apply(np.log1p)
    X_test[col]  = X_test[col].apply(np.log1p)
    X_val[col] = X_val[col].apply(np.log1p)
y_train = y_train.apply(np.log1p)
y_test = y_test.apply(np.log1p)
y_val = y_val.apply(np.log1p)

#onehotencoder
enc = OneHotEncoder(handle_unknown = 'ignore')
X_train = enc.fit_transform(X_train)
X_test = enc.transform(X_test)
X_val = enc.transform(X_val)

#Linear Regression model
lr = LinearRegression().fit(X_train, y_train)
y_val_predict = np.exp(lr.predict(X_val))
y_val_rescale = np.exp(y_val)
lr_r2_score_val = r2_score(y_val_rescale, y_val_predict)
print('lr_r2_score_val ',lr_r2_score_val)
#plot actual and predict value
f = plt.figure(figsize=(6,6))
ax = plt.axes()
ax.plot(y_val_rescale, y_val_predict,
         marker='o', ls='', ms=3.0)
lim = (0, y_val_rescale.max())
ax.set(xlabel='Actual Price',
       ylabel='Predicted Price',
       xlim=lim,
       ylim=lim,
       title='Linear Regression Results');
plt.show()


#RidgeCV
alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]
ridgeCV = RidgeCV(alphas = alphas, cv = 4).fit(X_train, y_train)
y_val_predict = np.exp(ridgeCV.predict(X_val))
y_val_rescale = np.exp(y_val)
ridgeCV_r2_score_val = r2_score(y_val_rescale, y_val_predict)
ridgeCV_r2_score_val
print('alpha, ridgeCV_r2_score_val ', ridgeCV.alpha_, ridgeCV_r2_score_val)

#LassoCV
alphas2 = np.array([1e-5, 5e-5, 0.0001, 0.0005])
lassoCV = LassoCV(alphas = alphas2, max_iter = 50000, cv = 3).fit(X_train, y_train)
y_val_predict = np.exp(lassoCV.predict(X_val))
y_val_rescale = np.exp(y_val)
lassoCV_r2_score_val = r2_score(y_val_rescale, y_val_predict)
print('alpha, lassoCV_r2_score_val ', lassoCV.alpha_, lassoCV_r2_score_val)

#ElasticNetCV
l1_ratios = np.linspace(0.1, 0.9, 9)
elasticNetCV = ElasticNetCV(alphas = alphas2, l1_ratio = l1_ratios, max_iter = 10000).fit(X_train, y_train)
y_val_predict = np.exp(elasticNetCV.predict(X_val))
y_val_rescale = np.exp(y_val)
elasticNetCV_r2_score_val = r2_score(y_val_rescale, y_val_predict)
print('elasticNetCV_r2_score_val, l1_ratio, alpha:  ', elasticNetCV_r2_score_val, elasticNetCV.l1_ratio_, elasticNetCV.alpha_)

#Compare r2_score and evaluate on test set
r2_score_vals = [ lr_r2_score_val, ridgeCV_r2_score_val, lassoCV_r2_score_val, elasticNetCV_r2_score_val]
labels = ['Linear', 'Ridge', 'Lasso', 'ElasticNet']
r2_score_vals = pd.Series(r2_score_vals, index=labels).to_frame()
r2_score_vals.rename(columns={0: 'R2_score'}, inplace=1)
r2_score_vals
#lasso is the best performance
y_test_predict = np.exp(lassoCV.predict(X_test))
y_test_rescale = np.exp(y_test)
lassoCV_r2_score_val = r2_score(y_test_rescale, y_test_predict)
#plot predict and actual output for ridge, lasso, elasticNet
f = plt.figure(figsize=(6,6))
ax = plt.axes()
labels = ['Ridge', 'Lasso', 'ElasticNet']
models = [ridgeCV, lassoCV, elasticNetCV]
for mod, lab in zip(models, labels):
    ax.plot(np.exp(y_test), np.exp(mod.predict(X_test)),
             marker='o', ls='', ms=3.0, label=lab)
leg = plt.legend(frameon=True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(1.0)
ax.set(xlabel='Actual Price',
       ylabel='Predicted Price',
       title='Linear Regression Results');
plt.show()

#with Stochastic Gradient Decent Regressor
model_parameters_dict = {
    'Linear': {'penalty': None},
    'Lasso' : {'penalty': 'l2', 'alpha': lassoCV.alpha_},
    'Ridge' : {'penalty': 'l1', 'alpha': ridgeCV.alpha_},
    'ElasticNetCV': {'penalty': 'elasticnet', 'alpha': elasticNetCV.alpha_, 'l1_ratio': elasticNetCV.l1_ratio_}
}
SGD_r2_score_vals = {}
for mdlabel, param in model_parameters_dict.items():
    SGD = SGDRegressor(random_state = 42, early_stopping= True, eta0 = 0.01411, learning_rate = 'adaptive', max_iter = 1000000, **param)
    SGD.fit(X_train, y_train)
    y_val_predict = np.exp(SGD.predict(X_val))
    SGD_r2_score_vals[mdlabel] = r2_score(y_val_rescale, y_val_predict)
SGD_r2_score_vals = pd.Series(SGD_r2_score_vals).to_frame()
SGD_r2_score_vals.rename(columns={0: 'R2_score'}, inplace=1)

#SGD with Minmax scaler
scaler = MinMaxScaler()
X_train_minmax_scaled = scaler.fit_transform(X_train.toarray())
X_val_minmax_scaled = scaler.transform(X_val.toarray())
X_test_minmax_scaled = scaler.transform(X_test.toarray())
SGD_minmaxscale_r2_score_vals = {}
for mdlabel, param in model_parameters_dict.items():
    SGD = SGDRegressor(random_state = 42, early_stopping= True, eta0 = 0.022, learning_rate = 'adaptive', max_iter = 1000000, **param)
    SGD.fit(X_train_minmax_scaled, y_train)
    y_val_predict = np.exp(SGD.predict(X_val_minmax_scaled))
    SGD_minmaxscale_r2_score_vals[mdlabel] = r2_score(y_val_rescale, y_val_predict)
SGD_minmaxscale_r2_score_vals = pd.Series(SGD_minmaxscale_r2_score_vals).to_frame()
SGD_minmaxscale_r2_score_vals.rename(columns={0: 'R2_score'}, inplace=1)
print(r2_score_vals, SGD_r2_score_vals, SGD_minmaxscale_r2_score_vals)

#compare lassoCv and SGD
y_val_predict = np.exp(lassoCV.predict(X_test))
y_test_predict = np.exp(lassoCV.predict(X_test))
y_test_rescale = np.exp(y_test)
lassoCV_r2_score_test = r2_score(y_test_rescale, y_test_predict)

SGD = SGDRegressor(penalty = 'elasticnet', alpha = elasticNetCV.alpha_, l1_ratio = elasticNetCV.l1_ratio_, early_stopping=True, eta0= 0.01411, learning_rate='adaptive', max_iter=1000000)
SGD.fit(X_train, y_train)
y_test_predict = np.exp(SGD.predict(X_test))
y_test_rescale = np.exp(y_test)
SGD_elasticNet_r2_score_test = r2_score(y_test_rescale, y_test_predict)
print('lassoCV, SGD r2_score', lassoCV_r2_score_test, SGD_elasticNet_r2_score_test )



#method by IBM course
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/Ames_Housing_Sales.csv")
data.head(10)
one_hot_encode_cols = data.dtypes[data.dtypes == 'object']
one_hot_encode_cols = one_hot_encode_cols.index.to_list()
for col in one_hot_encode_cols:
    data[col] = pd.Categorical(data[col])
data = pd.get_dummies(data, columns = one_hot_encode_cols)
train, test = train_test_split(data, test_size = 0.3, random_state = 42)
#checking skew
num_mask = train.dtypes == np.float64
float_cols = data.columns[num_mask]
skew_limit = 0.75
skew_vals_X = train[float_cols].skew()
skew_cols_X = (skew_vals.to_frame().rename(columns = {0:'skew'}).sort_values(by = 'skew', ascending = False).query('abs(skew)> {0}'.format(skew_limit)))



#plot columns with skew > skew_limit
field = "BsmtFinSF1"
fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
train[field].hist(ax=ax_before)
train[field].apply(np.log1p).hist(ax=ax_after)
ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
fig.suptitle('Field "{}"'.format(field))
#transfer for all skewed columns:
for field in skew_cols.index.to_list():
    if col == "SalePrice":
        continue
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(10, 5))
    #train[field].hist(ax=ax_before)
    #train[field].apply(np.log1p).hist(ax=ax_after)
    #ax_before.set(title='before np.log1p', ylabel='frequency', xlabel='value')
    #ax_after.set(title='after np.log1p', ylabel='frequency', xlabel='value')
    #fig.suptitle('Field "{}"'.format(field))

feature_cols = [x for x in train.columns if x != 'SalePrice']
X_train = train[feature_cols]
y_train = train['SalePrice']
X_test = test[feature_cols]
y_test = test['SalePrice']

