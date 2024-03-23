import os
USERNAME = os.environ.get('USERNAME', 'admin')
PASSWORD = os.environ.get('USERNAME', 'admin')



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/churndata_processed.csv")
print(data.head())
print(data.dtypes)
print(data.info())
print(data.describe().T,2)

fig, ax = plt.subplots(figsize = (15,10))
sns.heatmap(data.corr())

#preparation
target = 'churn_value'
data[target].value_counts(normalize = 'true')

feature_cols = [x for x in data.columns if x != target]
strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=1500, random_state=42)
for train_idx, test_idx in strat_shuff_split.split(data[feature_cols], data.churn_value):
    X_train = data.loc[train_idx, feature_cols]
    y_train = data.loc[train_idx, 'churn_value']
    X_test = data.loc[test_idx, feature_cols]
    y_test = data.loc[test_idx, 'churn_value']

X_train.head(5)
y_train.head(5)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

#Random forest and Out-of-bag error
RF = RandomForestClassifier( oob_score= True, random_state=42, warm_start=True, n_jobs=-1)
oob_list = list()
for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
    RF.set_params(n_estimators = n_trees)
    RF.fit(X_train, y_train)
    oob_error = 1 - RF.oob_score_
    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

rf_oob_df = pd.concat(oob_list, axis = 1).T.set_index('n_trees')
rf_oob_df

#plot oob error
sns.set_context('talk')
sns.set_style('white')
ax = rf_oob_df.plot(legend = False, marker = 'o', figsize = (14,7), linewidth = 5)
ax.set(ylabel = 'out-of-bag error')

#oob error with ExtraTreeClassifier
EF = ExtraTreesClassifier(oob_score=True, random_state=42, warm_start=True, bootstrap=True, n_jobs=-1)
oob_list = list()
for n_trees in [15, 20, 30, 40, 50, 100, 150, 200, 300, 400]:
    EF.set_params(n_estimators = n_trees)
    EF.fit(X_train, y_train)
    oob_error = 1 - EF.oob_score_
    oob_list.append(pd.Series({'n_trees': n_trees, 'oob': oob_error}))

et_oob_df = pd.concat(oob_list, axis = 1).T.set_index('n_trees')
et_oob_df

oob_df = pd.concat([rf_oob_df.rename(columns = {'oob': 'RandomForest'}),
                    et_oob_df.rename(columns= {'oob': 'ExtraTrees'})])
print(oob_df)

sns.set_context('talk')
sns.set_style('white')
ax = oob_df.plot(marker='o', figsize=(14, 7), linewidth=5)
ax.set(ylabel='out-of-bag error')


#predicting
model = RF.set_params(n_estimators = 100)
y_pred = model.predict(X_test)

cr = classification_report(y_test, y_pred)
print(cr)

score_df = pd.DataFrame({'accuracy': accuracy_score(y_test, y_pred),
                         'precision': precision_score(y_test, y_pred),
                         'recall': recall_score(y_test, y_pred),
                         'f1': f1_score(y_test, y_pred),
                         'auc': roc_auc_score(y_test, y_pred)},
                         index=pd.Index([0]))
print(score_df)

#plot the results
#plot confusion matrix
sns.set_context('talk')
cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels= model.classes_)
disp.plot()
plt.show()

#plot ROC-AUC
sns.set_context('talk')

fig, axList = plt.subplots(ncols=2)
fig.set_size_inches(16, 8)

# Get the probabilities for each of the two categories
y_prob = model.predict_proba(X_test)

# Plot the ROC-AUC curve
ax = axList[0]
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
ax.plot(fpr, tpr, linewidth=5)
ax.plot([0, 1], [0, 1], ls='--', color='black', lw=.3)
ax.set(xlabel='False Positive Rate',
       ylabel='True Positive Rate',
       xlim=[-.01, 1.01], ylim=[-.01, 1.01],
       title='ROC curve')
ax.grid(True)

# Plot the precision-recall curve
ax = axList[1]

precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1])
ax.plot(recall, precision, linewidth=5)
ax.set(xlabel='Recall', ylabel='Precision',
       xlim=[-.01, 1.01], ylim=[-.01, 1.01],
       title='Precision-Recall curve')
ax.grid(True)

plt.tight_layout()
plt.show()
#plot feature_importance
feature_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(16, 6))
ax.pie(feature_imp, labels=None, autopct=lambda pct: '{:1.1f}%'.format(pct) if pct > 5.5 else '')
ax.set(ylabel='Relative Importance')
ax.set(xlabel='Feature')

# Adjust the layout to prevent label overlapping
plt.tight_layout()

# Move the legend outside the chart
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),labels=feature_imp.index)

plt.show()