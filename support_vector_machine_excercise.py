import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
# Evaluation metrics related methods
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
rs = 123
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#load data

food_df = pd.read_csv("food_items_binary.csv")

food_df.head(10)
print(food_df.info())
feature_cols = list(food_df.iloc[:, :-1].columns)
feature_cols

X= food_df[feature_cols]
y= food_df.iloc[:,-1:]
X.describe()
y.value_counts(normalize=True)
y.value_counts().plot.bar(color = ['red','green'])
type(y)

X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.2, stratify=y, random_state=rs)
Minmax_scale = MinMaxScaler()
X_train = Minmax_scale.fit_transform(X_train)
X_test = Minmax_scale.transform(X_test)

model = SVC()
model.fit(X_train, y_train.values.ravel())
preds = model.predict(X_test)

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='binary')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

evaluate_metrics(y_test, preds)

#Train SVM with different regularization parameters and kernels
model = SVC( C = 10, kernel='rbf')
model.fit(X_train, y_train)
preds = model.predict(X_test)
evaluate_metrics(y_test, preds)

#tuning SVC
params_grid = {
    'C':[1,10,100],
    'kernel': ['poly','rbf', 'sigmoid']
}
model = SVC()
grid_search = GridSearchCV(estimator=model, param_grid=params_grid, scoring='accuracy', cv = 5, verbose=1)
grid_search.fit(X_train, y_train.values.ravel())
best_params = grid_search.best_params_

model = SVC( C = 100, kernel='rbf')
model.fit(X_train, y_train)
preds = model.predict(X_test)
evaluate_metrics(y_test, preds)

#plot SVM hyperplane and margin
simplified_food_df = food_df[['Calories','Dietary Fiber','class']]
X = simplified_food_df.iloc[:1000,:-1].values
y = simplified_food_df.iloc[:1000, -1:].values

under_sampler = RandomUnderSampler(random_state=rs)
X_under, y_under = under_sampler.fit_resample(X,y)
print(f"Dataset resampled shape, X: {X_under.shape}, y: {y_under.shape}")
print(f"Raw dataset shape, X: {X.shape}, y: {y.shape}")
print(y_under[:50])
plt.hist(x = y_under)

scaler = MinMaxScaler()
X_under = scaler.fit_transform(X_under)

linear_svm = SVC(C=1000, kernel='linear')
linear_svm.fit(X_under, y_under)

def plot_decision_boundry(X, y, model):
    plt.figure(figsize=(16, 12))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )

    # plot support vectors
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.show()

plot_decision_boundry(X_under, y_under, linear_svm)

svm_rbf_kernel = SVC( C = 100, kernel='rbf')
svm_rbf_kernel.fit(X_under, y_under)
plot_decision_boundry(X_under, y_under, svm_rbf_kernel)