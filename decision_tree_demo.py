import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Wine_Quality_Data.csv", sep=',')
data.head()
data.info()
data['color'] = data.color.replace('white',0).replace('red',1).astype(np.int_)
feature_cols = [x for x in data.columns if x not in 'color']

strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=42)
for train_idx, test_idx in strat_shuff_split.split(data[feature_cols], data.color):
    X_train = data.loc[train_idx, feature_cols]
    y_train = data.loc[train_idx, 'color']
    X_test = data.loc[test_idx, feature_cols]
    y_test = data.loc[test_idx, 'color']

y_train.value_counts(normalize=True).sort_index()
y_test.value_counts(normalize=True).sort_index()

#Fit to Decision tree
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt = DecisionTreeClassifier(random_state = 42)
dt = dt.fit(X_train, y_train)
print('node counts, max depth:  ',dt.tree_.node_count, dt.tree_.max_depth)

def measure_error(y_true, y_pred, label):
    return pd.Series({'accuracy': accuracy_score(y_true, y_pred),
                      'precision': precision_score(y_true, y_pred),
                      'recall': recall_score(y_true, y_pred),
                       'f1': f1_score(y_true, y_pred)},
                     name = label)

y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)
train_test_full_error = pd.concat([measure_error(y_train, y_train_pred,'train'),
                                   measure_error(y_test, y_test_pred, 'test')],
                                  axis = 1)
print(train_test_full_error)


#  train_test_full_error with scaled data
dt2 = DecisionTreeClassifier(random_state = 42)
dt2 = dt2.fit(X_train_scaled, y_train)
print('node count, max depth with Standard scaler: ',dt2.tree_.node_count, dt2.tree_.max_depth)
y_train_pred_scaled = dt2.predict(X_train_scaled)
y_test_pred_scaled = dt2.predict(X_test_scaled)
train_test_full_error_scaled = pd.concat([measure_error(y_train, y_train_pred_scaled,'train'),
                                   measure_error(y_test, y_test_pred_scaled, 'test')],
                                  axis = 1)
train_test_full_error_scaled

#use gridsearch to find the best decision tree params
param_grid = {'max_depth': range(1, dt.tree_.max_depth +1, 2),
              'max_features': range(1, len(dt.feature_importances_)+1)}
GCV = GridSearchCV(estimator= DecisionTreeClassifier(random_state=42),
                   param_grid=param_grid,
                   scoring='accuracy',
                   n_jobs = -1)
GCV = GCV.fit(X_train, y_train)
print('best Gridsearch for decision tree: ',GCV.best_estimator_.tree_.node_count, GCV.best_estimator_.tree_.max_depth)

y_train_pred_GCV = GCV.predict(X_train)
y_test_pred_GCV = GCV.predict(X_test)

train_test_GCV_error = pd.concat([measure_error(y_train, y_train_pred_GCV, 'train'),
                                 measure_error(y_test, y_test_pred_GCV, 'test')],
                                axis=1)
print(train_test_GCV_error)


#use residual_sugar as y
feature_cols = [x for x in data.columns if x != 'residual_sugar']
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'residual_sugar']
X_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'residual_sugar']
dr = DecisionTreeRegressor().fit(X_train, y_train)
param_grid = {'max_depth': range(1, dr.tree_.max_depth +1, 2),
              'max_features': range(1, len(dr.feature_importances_)+1,)}
GR_sugar = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error',
                        n_jobs= -1)
GR_sugar = GR_sugar.fit(X_train, y_train)
GR_sugar.best_estimator_.tree_.node_count, GR_sugar.best_estimator_.tree_.max_depth

y_train_pred_GR = GR_sugar.predict(X_train)
y_test_pred_GR = GR_sugar.predict(X_test)

train_test_GR_error = pd.Series({'train': mean_squared_error(y_train, y_train_pred_GR),
                                  'test': mean_squared_error(y_test, y_test_pred_GR)},
                                name = 'MSE').to_frame().T
print(train_test_GR_error)

#plot actual vs predicted residual sugar
sns.set_context('notebook')
sns.set_style('white')
fig = plt.figure( figsize=(6,6))
ax = plt.axes()
ph_test_predict = pd.DataFrame({'test': y_test.values,
                                'predict': y_test_pred_GR}).set_index('test').sort_index()
ph_test_predict.plot(marker = 'o', ls ='', ax = ax)
ax.set(xlabel = 'Test', ylabel = 'Predict', xlim = (0,35), ylim = (0,35))

#plot decision tree
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
filename = 'wine_tree.png'
graph.write_png(filename)
Image(filename=filename)

dot_data = StringIO()
export_graphviz(GCV.best_estimator_, out_file=dot_data, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
filename = 'wine_tree_prune.png'
graph.write_png(filename)
Image(filename=filename)

