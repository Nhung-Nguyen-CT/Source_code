import seaborn as sns, pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Human_Activity_Recognition_Using_Smartphones_Data.csv", sep=',')
data.info()
data.dtypes
data.dtypes.value_counts()
data.dtypes.tail()
data.iloc[:, :-1].min().value_counts()
data.iloc[:, :-1].max().value_counts()
data.Activity #target

#use LabelEncoder for
le = LabelEncoder()
data['Activity'] = le.fit_transform(data['Activity'])
data['Activity'].sample(5)

#calculate correllation matrix
feature_cols = data.columns[:-1]
corr_values = data[feature_cols].corr()

# Simplify by emptying all the data below the diagonal
tril_index = np.tril_indices_from(corr_values)

# Make the unused values NaNs
for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN

# Stack the data and convert to a data frame
corr_values = (corr_values
               .stack()
               .to_frame()
               .reset_index()
               .rename(columns={'level_0': 'feature1',
                                'level_1': 'feature2',
                                0: 'correlation'}))

# Get the absolute values for sorting
corr_values['abs_correlation'] = corr_values.correlation.abs()

#plot histogram of the abs value of correlation
sns.set_context('talk')
sns.set_style('white')
ax = corr_values.abs_correlation.hist(bins=50, figsize=(12, 8))
ax.set(xlabel='Absolute Correlation', ylabel='Frequency');
#sort abs correlation > 0.8
corr_values.sort_values('correlation', ascending = False).query('abs_correlation > 0.8')['correlation']


#split data by StratifiedShuffleSplit
strat_shuf_split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.3, random_state = 42)
for train_idx, test_idx in strat_shuf_split.split(data[feature_cols], data.Activity):
    X_train = data.loc[train_idx, feature_cols]
    y_train = data.loc[train_idx, 'Activity']
    X_test = data.loc[test_idx, feature_cols]
    y_test = data.loc[test_idx, 'Activity']

y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)

#fit LogisticRegression model:
lr = LogisticRegression(solver = 'liblinear').fit(X_train, y_train)
lr_l1 = LogisticRegressionCV(Cs = 10, cv = 4, penalty = 'l1', solver = 'liblinear').fit(X_train, y_train)
lr_l2 = LogisticRegressionCV(Cs = 10, cv = 4, penalty = 'l2', solver = 'liblinear').fit(X_train, y_train)

coefficients = list()
coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]],
                                 codes=[[0,0,0,0,0,0],[0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))

coefficients = pd.concat(coefficients, axis=1)
print(coefficients.sample(10))

#plot coefficients
fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10, 10)
for ax in enumerate(axList):
    loc = ax[0]
    ax = ax[1]

    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker='o', ls='', ms=2.0, ax=ax, legend=False)

    if ax is axList[0]:
        ax.legend(loc=4)

    ax.set(title='Coefficient Set ' + str(loc))
plt.tight_layout()

#Predict and store the class for each model
y_pred = list()
y_prob = list()

coeff_labels = ['lr','lr1','lr2']
coeff_models = [lr, lr_l1, lr_l2]

for lab, mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name = lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis = 1), name = lab))

y_pred = pd.concat(y_pred, axis = 1)
y_prob = pd.concat(y_prob, axis = 1)

y_pred.head()
y_prob.head()

#score
for lab in coeff_labels:
    precision, recall, fscore, _ = score(y_test, y_pred[lab], average = 'weighted')
    accuracy = accuracy_score(y_test, y_pred[lab])
    auc = roc_auc_score(label_binarize(y_test, classes = [0,1,2,3,4,5]), label_binarize(y_pred[lab], classes = [0,1,2,3,4,5]), average = 'weighted')
    cm[lab] = confusion_matrix(y_test, y_pred[lab])
    metrics.append(pd.Series({'precision': precision, 'recall': recall, 'fscore': fscore, 'accuracy': accuracy, 'auc': auc}, name = lab))
metrics = pd.concat(metrics, axis = 1)
metrics

#plot confusion matrix
fig, axList = plt.subplots(nrows=2, ncols=2)
axList = axList.flatten()
fig.set_size_inches(12, 10)
axList[-1].axis('off')
for ax, lab in zip(axList[:-1], coeff_labels):
    sns.heatmap(cm[lab], ax=ax, annot=True, fmt='d');
    ax.set(title=lab);
plt.tight_layout()
