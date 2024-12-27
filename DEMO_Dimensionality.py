import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/Wholesale_Customers_Data.csv', sep=',')
data.shape
data.head()
data.info()
data = data.drop(['Channel', 'Region'], axis=1)
data.dtypes
for col in data.columns:
    data[col] = data[col].astype(np.float64)
data_orig = data.copy()
corr_mat =data.corr()
# Strip the diagonal for future examination
for x in range(corr_mat.shape[0]):
    corr_mat.iloc[x, x] = 0.0
corr_mat
corr_mat.abs().idxmax()
#check skew
log_columns = data.skew().sort_values(ascending=False)
log_columns = log_columns.loc[log_columns > 0.75]
print('skew columns: ',log_columns)
# The log transformations
for col in log_columns.index:
    data[col] = np.log1p(data[col])
mms = MinMaxScaler()
for col in data.columns:
    data[col] = mms.fit_transform(data[[col]]).squeeze()

sns.set_context('notebook')
sns.set_style('white')
sns.pairplot(data)

#use pipeline for preprocessing
log_transformer = FunctionTransformer(np.log1p)
#pipeline
estimators = [('log1p', log_transformer), ('minmaxscale', MinMaxScaler())]
pipeline = Pipeline(estimators)
#convert the orginal data
data_pipe = pipeline.fit_transform(data_orig)

#perform PCA with n_components range from 1 to 5 and plot
pca_list = list()
feature_weight_list =list()
#fit a range of PCAmodels
for n in range(1,6):
    PCAmod = PCA(n_components=n)
    PCAmod.fit(data)
    pca_list.append(pd.Series({'n': n, 'model': PCAmod, 'var': PCAmod.explained_variance_ratio_.sum()}))
    #calculate and store feature importances
    abs_feature_values= np.abs(PCAmod.components_).sum(axis = 0)
    feature_weight_list.append(pd.DataFrame({'n': n, 'features': data.columns, 'values': abs_feature_values/abs_feature_values.sum()}))
pca_df =pd.concat(pca_list, axis =1).T.set_index('n')
print(pca_df)
features_df = (pd.concat(feature_weight_list)
               .pivot(index='n', columns='features', values='values'))
features_df

#plot variance by n of components
sns.set_context('talk')
ax = pca_df['var'].plot(kind='bar')
ax.set(xlabel='Number of dimensions',
       ylabel='Percent explained variance',
       title='Explained Variance vs Dimensions')
#plot features importances
ax =features_df.plot(kind = 'bar', figsize = (13,8))
ax.legend(loc = 'upper right')
ax.set(xlabel='Number of dimensions',
       ylabel='Relative importance',
       title='Feature importance vs Dimensions')


#fit PCA with rbf kernel
def scorer(pcamodel, X,y = None):
    try:
        X_val = X.values
    except:
        X_val = X
    #calculate and inverse transform the data
    data_inv = pcamodel.fit_transform(X_val)
    data_inv = pcamodel.inverse_transform(data_inv)
    #error calculation
    mse =  mean_squared_error(data_inv.ravel(),X_val.ravel())
    return -1 * mse

param_grid ={'gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],'n_components':[2,3,4]}
#the  grid search
kernelPCA = GridSearchCV(KernelPCA(kernel='rbf',fit_inverse_transform=True),
                         param_grid=param_grid,
                         scoring=scorer,
                         n_jobs=-1)
kernelPCA  = kernelPCA.fit(data)
kernelPCA.best_estimator_


#NEW DATASET - HOW MODEL ACCURACY CHANGE IF WE INCLUDE PCA IN MODEL BUILDING PIPELINE
data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/Human_Activity_Recognition_Using_Smartphones_Data.csv', sep=',')
data.columns
X = data.drop('Activity', axis=1)
y = data.Activity
sss = StratifiedShuffleSplit(n_splits=5, random_state=42)
def get_avg_score(n):
    pipe = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n)),
        ('estimator', LogisticRegression(solver='liblinear'))
    ]
    pipe = Pipeline(pipe)
    scores = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        pipe.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, pipe.predict(X_test)))
    return np.mean(scores)
ns = [10, 20, 50, 100, 150, 200, 300, 400]
score_list = [get_avg_score(n) for n in ns]

sns.set_context('talk')

ax = plt.axes()
ax.plot(ns, score_list)
ax.set(xlabel='Number of Dimensions',
       ylabel='Average Accuracy',
       title='LogisticRegression Accuracy vs Number of dimensions on the Human Activity Dataset')
ax.grid(True)