import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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
df = pd.read_pickle('events_random_setup.pkl')

df.info()

print(df.info)
df = df.sort_values(by = 'Close_Date').copy()
df['Have_profit'] = np.where(df['Profit'] >= 0, 'True', 'False')
df['FFT-Coeff'].apply(len).unique()
a = np.vstack(df['FFT-Coeff'])
for x in range(4):
    df[f'FFT-Coeff_{x+1}'] = a[:, x*5]+a[:, x*5+1]+a[:, x*5+2]+a[:, x*5+3]+a[:, x*5+4]
df['FFT-Coeff_total'] = df['FFT-Coeff_1'] + df['FFT-Coeff_2'] + df['FFT-Coeff_3'] + df['FFT-Coeff_4']
df['Open_close_distace'] = (df['Close'] - df['Open'])
df['Price_distace_per_volume'] = df['Open_close_distace']/df['Volume']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np

# Initialize empty lists to store data
x_data = []
y_data = []
z_data = []


# Function to update the plot with new data
def update(frame, step = 200):
    # Calculate the start and end index for the current batch
    start_idx = frame * step
    end_idx = min((frame + 1) * step, len(df))

    # Get the data points for the current batch
    batch_data = df.iloc[start_idx:end_idx]
    batch_data['Have_profit_int']= batch_data['Have_profit'].astype('category').cat.codes

    # Update the scatter plot with the current batch of data
    ax.clear()
    ax.scatter(batch_data['RSI-Force-Entropy'], batch_data['RSI-Force'], batch_data['Profit'], c = batch_data['Have_profit_int'])
    ax.set_xlabel('RSI-Force-Entropy')
    ax.set_ylabel('RSI-Force')
    ax.set_zlabel('Profit')
    ax.set_title(f'Frame {frame}')

    return ax


# Create a figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
ani = FuncAnimation(fig, update, frames=20, interval=4000, repeat=True)

plt.show()
df2 = df.loc[(df['Type'] == 'short') & ((df['LON_Timerange'] == 1) | (df['NEW_Timerange'] == 1)) & (df['Target'] >= 2*df['Stoploss'])]
X = df2[['RSI-Force','RSI-Force-Entropy', 'Entropy', 'FFT-Coeff_1', 'FFT-Coeff_total', 'Price_distace_per_volume','Target', 'Stoploss']]
y = df2['Have_profit']
y_pred_final = []

for i in range (200):
        print(' i = ',i)
        d = 200
        train_size = 1000
        test_size = 200
        X_train = X.iloc[i*d:i*d+train_size,:].copy()
        y_train = y[i*d:i*d+ train_size].copy()
        X_test1 = X.iloc[i*d + train_size:i*d + train_size + test_size,:].copy()
        y_test1 = y[i*d + train_size:i*d + train_size + test_size].copy()
        X_test2 = X.iloc[i*d + train_size + test_size:i*d + train_size + test_size*2,:].copy()
        y_test2 = y[i*d + train_size + test_size:i*d + train_size + test_size*2].copy()
        X_test3 = X.iloc[i*d + train_size + test_size*2:i*d + train_size + test_size*3,:].copy()
        y_test3 = y[i*d + train_size + test_size*2:i*d + train_size + test_size*3].copy()

        numeric_columns = [x for x in X_train.columns if X_train[x].dtypes != 'O']
        X_train1 = X_train[numeric_columns].copy()
        scale_skew_cols = pd.DataFrame(X_train1.skew(), columns=['skew']).sort_values(by='skew', ascending=False).query('abs(skew) >= 1.5').index
        if len(scale_skew_cols) > 0:
            pt = PowerTransformer(method='yeo-johnson')
            X_train[scale_skew_cols] = pt.fit_transform(X_train[scale_skew_cols])
            X_test1[scale_skew_cols] = pt.transform(X_test1[scale_skew_cols])
            X_test2[scale_skew_cols] = pt.transform(X_test2[scale_skew_cols])
            X_test3[scale_skew_cols] = pt.transform(X_test3[scale_skew_cols])

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
        print('Len of y_pred', len(y_pred))
        print('Evaluating  train', evaluate_metrics(yt=y_test1, yp = y_pred))
        y_pred_final.append(y_pred)


y_pred_final=np.concatenate(y_pred_final)
print(len(y_pred_final))

df = pd.read_pickle('events_random_setup.pkl')
df = df.iloc[0:len(y_pred_final),:].copy()
df['Have_profit_predict'] = y_pred_final
df['Profit_after_classification'] = df['Profit']*df['Have_profit_predict']
# df['Have_profit'] = np.where(df['Profit_after_classification'] > 0, 'Profit',)
# df['Have_profit'].value_counts(normalize = True)
print(np.sum(df['Profit_after_classification'] > 0) / len(df))
print(np.sum(df['Profit_after_classification'] == 0) / len(df))
print(np.sum(df['Profit_after_classification'] < 0) / len(df))
