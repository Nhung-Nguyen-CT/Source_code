import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, StandardScaler
import math
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
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)



# Load data from pickle file
with open('events_force.pkl', 'rb') as f:
    df = pickle.load(f)

df.info()
df = df.drop(columns = ['Record_Date','Open_Date','Open_Price','Num_Tick', 'Highest_Price','Lowest_Price','Close_Date','Close_Price','Maximum_Stoploss','Date'])
print(df.info)
#CLEAN AND TRANSFORM
# classify: profit into profit and loss
df['Have_profit'] = np.where(df['Profit'] >= 0, 'True', 'False')
df['Open_close_distace'] = (df['Close'] - df['Open'])
df['Low_open_distance'] = (df['Low'] - df['Open'])
df['High_open_distance'] = (df['High'] - df['Open'])
df['Price_distace_per_volume'] = df['Open_close_distace']/df['Volume']
#check lenghth of every values in 'FFT-Coeff' and sparse
df['FFT-Coeff'].apply(len).unique()
a = np.vstack(df['FFT-Coeff'])
for x in range(10):
    df[f'FFT-Coeff_{x}'] = a[:, x]
df = df.drop(columns = ['Profit', 'Open', 'Close', 'Low', 'High','Volume','FFT-Coeff','FFT-Freqs'])
df.info()

#CHECK CORRELATION BETWEEN NUMERICAL VARIABLE
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

df = df.drop(columns = ['Close-EMA-14', 'Close-EMA-28', 'Close-SMA-28','Open_close_distace','FFT-Coeff_6','FFT-Coeff_7','FFT-Coeff_8','FFT-Coeff_5','FFT-Coeff_3','FFT-Coeff_2'])
numeric_columns = [x for x in df.columns if df[x].dtypes != 'O']
df1 = df[numeric_columns].copy()
scale_skew_cols = pd.DataFrame(df1.skew(), columns = ['skew']).sort_values(by = 'skew',ascending=False).query('abs(skew) >= 1.5').index
pt = PowerTransformer(method='yeo-johnson')
df[scale_skew_cols ] = pt.fit_transform(df[scale_skew_cols])
print('after apply Yeo-Johnson Transformation: ',df[scale_skew_cols ].skew())

#PLOT DISTRIBUTION OF NUMERICAL VARIABLE BY 'HAVE_PROFIT'
df1 = df[numeric_columns + ['Have_profit']].copy()
# Iterate through features and create KDE plots
for i in range (math.ceil((len(df1.columns)-1)/4)):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('KDE Plots for Numerical Features by Have_profit')
    sns.kdeplot(data=df1, x=numeric_columns[i*4+0], hue='Have_profit', ax=axes[0, 0])
    sns.kdeplot(data=df1, x=numeric_columns[i*4 + 1], hue='Have_profit', ax=axes[0, 1])
    sns.kdeplot(data=df1, x=numeric_columns[i*4 + 2], hue='Have_profit', ax=axes[1, 0])
    if i < 4:
        sns.kdeplot(data=df1, x=numeric_columns[i*4 + 3], hue='Have_profit', ax=axes[1, 1])
    plt.show()

#PREPROCESSING
X = df[[i for i in df.columns if i != 'Have_profit']].copy()
y = df['Have_profit'].copy()

for i in range (50):
        print(' i = ',i)
        d = 200
        train_size = 1000
        test_size = 200
        X_train = X.iloc[i*d:i*d+train_size,:].copy()
        y_train = y[i*d:i*d+ train_size].copy()
        X_test1 = X.iloc[i*d + train_size + 1:i*d + train_size + test_size,:].copy()
        y_test1 = y[i*d + train_size + 1:i*d + train_size + test_size].copy()
        X_test2 = X.iloc[i*d + train_size + test_size + 1:i*d + train_size + test_size*2,:].copy()
        y_test2 = y[i*d + train_size + test_size + 1:i*d + train_size + test_size*2].copy()
        X_test3 = X.iloc[i*d + train_size + test_size*2 + 1:i*d + train_size + test_size*3,:].copy()
        y_test3 = y[i*d + train_size + test_size*2 + 1:i*d + train_size + test_size*3].copy()

        scalar = StandardScaler()
        X_train[numeric_columns] = scalar.fit_transform(X_train[numeric_columns])
        X_test1[numeric_columns] = scalar.transform(X_test1[numeric_columns])
        X_test2[numeric_columns] = scalar.transform(X_test2[numeric_columns])
        X_test3[numeric_columns] = scalar.transform(X_test3[numeric_columns])

        one_hot = ce.OneHotEncoder(use_cat_names = True)
        X_train= one_hot.fit_transform(X_train)
        X_test1= one_hot.transform(X_test1)
        X_test2= one_hot.transform(X_test2)
        X_test3= one_hot.transform(X_test3)
        print(X_test2.head())
        print(X_test2.info())

        la_encode = LabelEncoder()
        y_train = la_encode.fit_transform(y_train)
        y_test1 = la_encode.fit_transform(y_test1)
        y_test2 = la_encode.fit_transform(y_test2)
        y_test3 = la_encode.fit_transform(y_test3)

        #polynomial feature
        pf = PolynomialFeatures(degree=2)
        X_train_poly = pf.fit_transform(X_train)
        X_test1_poly = pf.fit_transform(X_test1)
        X_test2_poly = pf.fit_transform(X_test2)
        X_test3_poly = pf.fit_transform(X_test3)

        #BUILD MODEL
        def evaluate_metrics(yt, yp):
            results_pos = {}
            results_pos['accuracy'] = accuracy_score(yt, yp)
            precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='weighted')
            results_pos['recall'] = recall
            results_pos['precision'] = precision
            results_pos['f1score'] = f_beta
            return results_pos

        #RandomForest
        print('Model RandomForest: ')
        param_grid = {'max_depth': [5,10],
                      'n_estimators': [5,10]}
        model = RandomForestClassifier(oob_score= True)
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv = 3)
        search.fit(X_train, y_train)
        print(search.best_score_)
        print(search.best_params_)
        print('Evaluating  train', evaluate_metrics(yt=y_train, yp=search.predict(X_train)))
        print('Evaluating  train', evaluate_metrics(yt=y_test1, yp=search.predict(X_test1)))
        print('Evaluating  train', evaluate_metrics(yt=y_test2, yp=search.predict(X_test2)))
        print('Evaluating  train', evaluate_metrics(yt=y_test3, yp=search.predict(X_test3)))
        y_pred1_1 = search.predict(X_test1)
        #XGBoost
        # param_grid = {'n_estimators' : [5,10],
        #               'max_depth': [5,10]
        #               }
        # early_stopping_rounds = 10
        # objective = 'multi:softmax'
        # eval_metric = 'mlogloss'
        # num_class = 2
        # eval_set = [(X_train, y_train)]
        # model = XGBClassifier(objective = objective, eval_metric = eval_metric, num_class = num_class, early_stopping_rounds = early_stopping_rounds, verbosity = 0)
        # search = GridSearchCV(estimator=model, param_grid=param_grid, scoring= 'neg_log_loss', cv = 3)
        # search.fit(X_train, y_train, eval_set = eval_set)
        # print(search.best_score_)
        # print(search.best_params_)
        # print('Evaluating  train', evaluate_metrics(yt=y_train, yp=search.predict(X_train)))
        # print('Evaluating  train', evaluate_metrics(yt=y_test1, yp=search.predict(X_test1)))
        # print('Evaluating  train', evaluate_metrics(yt=y_test2, yp=search.predict(X_test2)))
        # print('Evaluating  train', evaluate_metrics(yt=y_test3, yp=search.predict(X_test3)))


        #we will fit polynomial X features to Logistic Regression
        print('Modeling by Logistic Regression: ')
        penalty = 'l2'
        solver = 'lbfgs'
        max_iter = 1000
        model = LogisticRegression(random_state = 42, penalty = penalty, solver = solver, max_iter = max_iter)
        model.fit(X_train_poly, y_train)
        print('Evaluating  train', evaluate_metrics(yt=y_train, yp=model.predict(X_train_poly)))
        print('Evaluating  train', evaluate_metrics(yt=y_test1, yp=model.predict(X_test1_poly)))
        print('Evaluating  train', evaluate_metrics(yt=y_test2, yp=model.predict(X_test2_poly)))
        print('Evaluating  train', evaluate_metrics(yt=y_test3, yp=model.predict(X_test3_poly)))
        y_pred1_2 = model.predict(X_test1_poly)

        #DECISION TREE
        print('Decision tree: ')
        param_grid = {'max_depth': [5, 10]}
        model = DecisionTreeClassifier(random_state=42)
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        search.fit(X_train, y_train)
        print('Evaluating  train', evaluate_metrics(yt=y_train, yp=search.predict(X_train)))
        print('Evaluating  train', evaluate_metrics(yt=y_test1, yp=search.predict(X_test1)))
        print('Evaluating  train', evaluate_metrics(yt=y_test2, yp=search.predict(X_test2)))
        print('Evaluating  train', evaluate_metrics(yt=y_test3, yp=search.predict(X_test3)))
        y_pred1_3 = search.predict(X_test1)

        #stacking
        print('stacking, average vote')
        y_pred = (y_pred1_3 + y_pred1_2 + y_pred1_1*2)/4
        y_pred = np.where(y_pred >0.5 , 1, 0)
        print('Evaluating  train', evaluate_metrics(yt=y_test1, yp = y_pred))
