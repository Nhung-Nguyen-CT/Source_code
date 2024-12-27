import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Human_Activity_Recognition_Using_Smartphones_Data.csv", sep=',')
print(data.info())
print(data.describe().T)
data = data.loc[:1000]

#select float columns and check min max
float_columns = (data.dtypes == float)
print((data.loc[:,float_columns].max() == 1.0).all())
print((data.loc[:,float_columns].min() == - 1.0).all())

#Encode Activity
le = LabelEncoder()
data['Activity'] = le.fit_transform(data['Activity'])
print(le.classes_)
print(data.Activity.nunique())

feature_columns = [x for x in data.columns if x != 'Activity']
X_train, X_test, y_train, y_test = train_test_split(data[feature_columns], data['Activity'], test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#fit gradient boosting and avaluate by accuracy score
error_list = list()

tree_list = [15,25,50,100,200,400]
for n_trees in tree_list:
    GBC = GradientBoostingClassifier(n_estimators=n_trees, random_state=42)
    GBC.fit(X_train.values, y_train.values)
    y_pred = GBC.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    error_list.append(pd.Series({'n_trees': n_trees, 'error': error}))

error_df = pd.concat(error_list, axis = 1).T.set_index('n_trees')
error_df

#plot error
sns.set_context('talk')
sns.set_style('white')
ax = error_df.plot(marker = 'o', figsize = (12,8), linewidth = 5)
ax.set(xlabel = 'Number of trees', ylabel = 'Error')
ax.set_xlim(0, max(error_df.index)*1.1)
plt.show()

#gridsearchcv for gradient boosting
param_grid = {'n_estimators': tree_list,
              'learning_rate': [0.1, 0.01, 0.001, 0.0001],
              'subsample': [1.0, 0.5],
              'max_features': [1, 2, 3, 4]}

GV_GBC = GridSearchCV(GradientBoostingClassifier(random_state=42),
                      param_grid=param_grid,
                      scoring='accuracy',
                      n_jobs=-1)
GV_GBC = GV_GBC.fit(X_train, y_train)
GV_GBC.best_estimator_

y_pred = GV_GBC.predict(X_test)
print(classification_report(y_pred, y_test))

#plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt ='g', cmap = 'Blues')
plt.show()


#fit to Logistic regression and compare
LR_L2 = LogisticRegression(penalty='l2', max_iter=500, solver = 'saga').fit(X_train, y_train)
y_pred = LR_L2.predict(X_test)
print(classification_report(y_pred, y_test))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt ='g', cmap = 'Blues')
plt.show()

#stacked model
estimators = [('LR_L2', LR_L2), ('GBC', GV_GBC)]
VC = VotingClassifier(estimators= estimators, voting='soft')
VC = VC.fit(X_train, y_train)

y_pred = VC.predict(X_test)
print(classification_report(y_pred, y_test))
#plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.show()
