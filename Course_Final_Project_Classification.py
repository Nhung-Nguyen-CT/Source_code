import math

import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
#from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import math

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('bank_credit_scoring.csv', delimiter= ',')
df.info()

#I. CLEAN DATA

print(df.duplicated().any())
print(df.isnull.any())


#Find and drop the high corellation variables:
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, mark = 0.5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[au_corr > mark]
numeric_columns = [x for x in df.columns if df[x].dtypes != 'O']
df1 = df[numeric_columns].copy()
print("Top Absolute Correlations")
print(get_top_abs_correlations(df = df1, mark = 0.6))

df = df.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN','Monthly_Inhand_Salary', 'Amount_invested_monthly', 'Monthly_Balance', 'Num_of_Loan', 'Interest_Rate', 'Num_Credit_Inquiries' ])

#object columns with low frequency of values (high n unique)
len_df = len(df)
high_nunique_object_columns = [x for x in df.columns if ((df[x].dtypes == 'O') & (df[x].nunique() > 0.1 * len_df))]
low_nunique_object_columns = [x for x in df.columns if ((df[x].dtypes == 'O') & (df[x].nunique() <= 0.1 * len_df))]
high_nunique_numeric_columns = [x for x in df.columns if ((df[x].dtypes != 'O') & (df[x].nunique() > 0.1 * len_df))]
low_nunique_numeric_columns = [x for x in df.columns if ((df[x].dtypes != 'O') & (df[x].nunique() <= 0.1 * len_df))]


print('high nunique and dtype = object', high_nunique_object_columns)
for x in high_nunique_object_columns:
    print('columns ',x,': nunique ',df[x].nunique())

print('low nunique and dtype = object', low_nunique_object_columns)
for x in low_nunique_object_columns:
    print('columns ', x, ': nunique ', df[x].nunique())

print('high nunique and dtype = numeric', high_nunique_numeric_columns)
for x in high_nunique_numeric_columns:
    print('columns ',x,': nunique ',df[x].nunique())

print('low nunique and dtype = numeric', low_nunique_numeric_columns)
for x in low_nunique_numeric_columns:
    print('columns ', x, ': nunique ', df[x].nunique())

numeric_cols_need_encode = ['Month']
object_cols_need_spare = ['Type_of_Loan']
object_cols_need_encode = [x for x in low_nunique_object_columns if ((x != 'Type_of_Loan') & (x != 'Credit_Score'))] # exclude Credit Score and Type of Loan
cols_need_encode = numeric_cols_need_encode + object_cols_need_encode
scale_cols = [x for x in df.columns if  (df[x].dtypes != 'O' ) & (x != 'Month') & ( x != 'Unnamed: 0')]


type_of_Loan_value_counts = (df['Type_of_Loan'].value_counts())
type_of_Loan_value_counts = type_of_Loan_value_counts.to_frame()
print(type_of_Loan_value_counts.head(100))

#drop rows with Type_of_Loan == 'No Data'

indexes = df.query('Type_of_Loan == "No Data" ').index.to_list()
df = df.drop(index = indexes)
df['Type_of_Loan'] = df['Type_of_Loan'].str.replace('and ','')
df['Type_of_Loan'] = df['Type_of_Loan'].str.split(', ')

df = pd.concat(
    [
        df.explode('Type_of_Loan')
        .pivot_table(index="Unnamed: 0", columns='Type_of_Loan', aggfunc='size', fill_value=0)
        .add_prefix('Type_of_Loan_'),
        df.set_index("Unnamed: 0"),
    ],
    axis=1,
)

#drop Type_of_Loan, Month, 'Num_Bank_Counts column and replace value >1 by 1 (encode)
df = df.drop(columns=['Type_of_Loan'])
df.iloc[:,:9] = np.where(df.iloc[:,:9] >= 1, 1, 0)
print(df.describe().T)

#plot data
sns.histplot(data = df, x = 'Credit_Score', kind = 'hist', hue = 'Credit_Score')
plt.show()
# => imbalance 'Credit_score' values

numeric_columns = [x for x in df.columns if df[x].dtypes != 'O']
df1 = df[numeric_columns + ['Credit_Score']].copy()


# Iterate through features and create KDE plots
for i in range ( math.ceil(len(df1.columns)/4)):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('KDE Plots for Numerical Features by Credit_Score')
    sns.kdeplot(data=df1, x=numeric_columns[i*4+0], hue='Credit_Score', ax=axes[0, 0])
    if i < 5:
        sns.kdeplot(data=df1, x=numeric_columns[i*4 + 1], hue='Credit_Score', ax=axes[0, 1])
        sns.kdeplot(data=df1, x=numeric_columns[i*4 + 2], hue='Credit_Score', ax=axes[1, 0])
        sns.kdeplot(data=df1, x=numeric_columns[i*4 + 3], hue='Credit_Score', ax=axes[1, 1])
    plt.show()

drop_cols = ['Month'] # drop because no relationship on y

cate_columns = [x for x in df.columns if df[x].dtypes == 'O']
df1 = df[cate_columns].copy()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('KDE Plots for Category Features by Credit_Score')
for i in range(2):
    axes[0, i].tick_params(axis='x', labelrotation=45)
    axes[1, i].tick_params(axis='x', labelrotation=45)
    sns.histplot(data=df1, x=cate_columns[i*2], hue='Credit_Score', ax=axes[0, i], hue_order=['Good', 'Standard', 'Poor'], kde= True)
    sns.histplot(data=df1, x=cate_columns[i*2+1], hue='Credit_Score', ax=axes[1, i], hue_order=['Good', 'Standard', 'Poor'], kde= True)
plt.show()

df = df.drop(columns = drop_cols)

#II.PREPROCESSING
#frac = 0.4
#df = df.sample(frac=frac).copy()

X = df.iloc[:,:-1].copy()
y = df.iloc[:,-1].copy()
#X['Month'] = X['Month'].astype('str')

strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in strat_shuff_split.split(X, y):
    X_train = X.iloc[train_idx,:].copy()
    y_train = y.iloc[train_idx].copy()
    X_test = X.iloc[test_idx, :].copy()
    y_test = y.iloc[test_idx].copy()

strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in strat_shuff_split.split(X_train, y_train):
    X_train = X_train.iloc[train_idx,:].copy()
    y_train = y_train.iloc[train_idx].copy()
    X_val = X.iloc[val_idx, :].copy()
    y_val = y.iloc[val_idx].copy()

#Scale
scale_skew_cols = pd.DataFrame(X_train[scale_cols].skew(), columns = ['skew']).sort_values(by = 'skew',ascending=False).query('abs(skew) >= 1').index
X_train[scale_skew_cols] = X_train[scale_skew_cols].apply(np.log1p)
X_test[scale_skew_cols] = X_test[scale_skew_cols].apply(np.log1p)
X_val[scale_skew_cols] = X_val[scale_skew_cols].apply(np.log1p)

scalar = StandardScaler()
X_train[scale_cols] = scalar.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scalar.transform(X_test[scale_cols])
X_val[scale_cols] = scalar.transform(X_val[scale_cols])

#encode
one_hot = ce.OneHotEncoder(use_cat_names = True)
X_train= one_hot.fit_transform(X_train)
X_test = one_hot.transform(X_test)
X_val = one_hot.transform(X_val)
print(X_train.head())
print(X_train.info())

la_encode = LabelEncoder()
y_train = la_encode.fit_transform(y_train)
y_test = la_encode.transform(y_test)
y_val = la_encode.transform(y_val)
print('20 first values of y_train: ',y_train[:20])

#add polynomial features of X
pf = PolynomialFeatures(degree=2)
X_train_poly = pf.fit_transform(X_train)
X_test_poly = pf.fit_transform(X_test)
X_val_poly = pf.fit_transform(X_val)


#smote_sampler = SMOTE(random_state = 42) # can only fit to X: array-like or parse matrix
#X_train, y_train = smote_sampler.fit_resample(X_train, y_train)

sns.displot(data = y_train)
plt.xlabel('Credit_Score value')
plt.ylabel('Count')
plt.show()  #balance between classes of Credit_Score


print ('Train set', X_train.shape,  y_train.shape)
print ('Test set', X_test.shape,  y_test.shape)
print ('Val set', X_val.shape,  y_val.shape)

X_train.info()
X_train.describe(include='all').T




#MODELLING

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='weighted')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

print('Model: XGBClassifier')

#PCA to decrease dimension of X poly
pca_ = PCA(n_components=500)
PCA_X_train_poly = pca_.fit_transform(X_train_poly)
PCA_X_val_poly = pca_.transform(X_val_poly)
PCA_X_test_poly = pca_.transform(X_test_poly)
print('Explained variation per principal component: {}'.format(np.cumsum(pca_.explained_variance_ratio_)))




param_grid = {'learning_rate': [0.2,0.4, 0.5],
             #'n_estimators' : [70, 100, 150, 220],
              'max_depth': [5, 10, 15, 20, 30]
              #'gamma' :[0.1, 1,2],
              #'reg_lamda' : [2, 10, 20],
              #'alpha': [0.5, 1, 2]
              }
early_stopping_rounds = 10
objective = 'multi:softmax'
eval_metric = 'mlogloss'
num_class = 3
eval_set = [(X_val, y_val)]
model = XGBClassifier(objective = objective, eval_metric = eval_metric, num_class = num_class, early_stopping_rounds = early_stopping_rounds )
search = GridSearchCV(estimator=model, param_grid=param_grid, scoring= 'neg_log_loss', cv = 3)
search.fit(X_train, y_train, eval_set = eval_set)
print(search.best_score_)
print(search.best_params_)
print('Evaluating  train', evaluate_metrics(yt=y_train, yp=search.predict(X_train)))
print('Evaluating  val', evaluate_metrics(yt=y_val, yp=search.predict(X_val)))
print('Evaluating  test', evaluate_metrics(yt=y_test, yp=search.predict(X_test)))


alpha = 0.8
gamma = 0.8
learning_rate = 0.13
max_depth = 30
n_estimators = 400
reg_lamda = 20
eval_set = [(X_train, y_train), (X_val, y_val)]
early_stopping_rounds = 40
model =XGBClassifier(objective=objective,alpha = alpha, gamma = gamma, learning_rate=learning_rate, max_depth = max_depth, n_estimators=n_estimators, reg_lamda = reg_lamda, early_stopping_rounds = early_stopping_rounds)
model.fit(X_train, y_train, eval_metric = eval_metric, eval_set = eval_set, verbose = False)
results = model.evals_result()
print(results)
#plot results for each epoch
fig, ax = plt.subplots()
ax.plot(results['validation_0']['mlogloss'], label = 'Train')
ax.plot(results['validation_1']['mlogloss'], label = 'Validation')
ax.legend()
plt.show()

print('Evaluating  train', evaluate_metrics(yt=y_train, yp=model.predict(X_train)))
print('Evaluating  val', evaluate_metrics(yt=y_val, yp=model.predict(X_val)))
print('Evaluating  test', evaluate_metrics(yt=y_test, yp=model.predict(X_test)))


#RandomForest
print('Model RandomForest: ')
param_grid = {'max_depth': [10,20,30]
              }
model = RandomForestClassifier(oob_score= True)
search = GridSearchCV(estimator=model, param_grid=param_grid, cv = 3)
search.fit(X_train, y_train)
print(search.best_score_)
print(search.best_params_)
print('Evaluating  train', evaluate_metrics(yt=y_train, yp=search.predict(X_train)))
print('Evaluating  val', evaluate_metrics(yt=y_val, yp=search.predict(X_val)))
print('Evaluating  test', evaluate_metrics(yt=y_test, yp=search.predict(X_test)))


#Logistic regression

#apply polynomial


#modeling
penalty = 'l2'
multi_class = 'multinomial'
solver = 'lbfgs'
max_iter = 1000
model = LogisticRegression(random_state = 42, penalty = penalty, multi_class = multi_class, solver = solver, max_iter = max_iter)
model.fit(X_train_poly, y_train)
print('Evaluating  train', evaluate_metrics(yt=y_train, yp=model.predict(X_train_poly)))
print('Evaluating  val', evaluate_metrics(yt=y_val, yp=model.predict(X_val_poly)))
print('Evaluating  test', evaluate_metrics(yt=y_test, yp=model.predict(X_test_poly)))
