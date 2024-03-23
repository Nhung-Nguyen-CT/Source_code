import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/Wine_Quality_Data.csv", sep=',')
data.head()
data.info()
data.nunique()
data.quality.value_counts(normalize = True)
data = data.dropna()

#encode y into binary and pairplot
y = (data['color'] == 'red').astype(int)
fields = list(data.columns[:-1])
correlations = data[fields].corrwith(y)
correlations.sort_values(inplace = True)
print(correlations)

#pair plot
#sns.set_context('talk')
#sns.set_style('white')
#sns.pairplot(data, hue = 'color')
#ax = correlations.plot(kind = 'bar')
#ax.set(ylim = [-1,1], ylabel = 'pearson correlation')
#plt.show()

#find correlations features by correl values
#correl_limit = 0.1
#correlations = (correlations.to_frame().rename(columns = {0:'correlation'}).sort_values(by = 'correlation', ascending = False).query('abs(correlation)> {0}'.format(correl_limit)))
#fields =  correlations.index


fields = correlations.map(abs).sort_values().iloc[-2:].index
print(fields)
X= data[fields]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['%s_scaled' % fld for fld in fields])
print(X.columns)

#fit Linear SVC
LSVC = LinearSVC()
LSVC.fit(X, y)

X_color = X.sample(300, random_state= 45)
y_color = y.loc[X_color.index]
y_color = y_color.map(lambda r:'red' if r == 1 else 'yellow')
ax = plt.axes()
ax.scatter(X_color.iloc[:,0], X_color.iloc[:,1], color = y_color, alpha = 1)
# -----------
x_axis, y_axis = np.arange(0, 1.005, .005), np.arange(0, 1.005, .005)
xx, yy = np.meshgrid(x_axis, y_axis)
xx_ravel = xx.ravel()
yy_ravel = yy.ravel()
X_grid = pd.DataFrame([xx_ravel, yy_ravel]).T
y_grid_predictions = LSVC.predict(X_grid)
y_grid_predictions = y_grid_predictions.reshape(xx.shape)
ax.contourf(xx, yy, y_grid_predictions, cmap=plt.cm.autumn_r, alpha=.3)
# -----------
ax.set(
    xlabel=fields[0],
    ylabel=fields[1],
    xlim=[0, 1],
    ylim=[0, 1],
    title='decision boundary for LinearSVC')


#Gaussian Kernel
def plot_decision_boundary(estimator, X, y):
    estimator.fit(X, y)
    X_color = X.sample(300)
    y_color = y.loc[X_color.index]
    y_color = y_color.map(lambda r: 'red' if r == 1 else 'yellow')
    x_axis, y_axis = np.arange(0, 1, .005), np.arange(0, 1, .005)
    xx, yy = np.meshgrid(x_axis, y_axis)
    xx_ravel = xx.ravel()
    yy_ravel = yy.ravel()
    X_grid = pd.DataFrame([xx_ravel, yy_ravel]).T
    y_grid_predictions = estimator.predict(X_grid)
    y_grid_predictions = y_grid_predictions.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.contourf(xx, yy, y_grid_predictions, cmap=plt.cm.autumn_r, alpha=.3)
    ax.scatter(X_color.iloc[:, 0], X_color.iloc[:, 1], color=y_color, alpha=1)
    ax.set(
        xlabel=fields[0],
        ylabel=fields[1],
        title=str(estimator))
gammas = [.5, 1, 2, 10]
for gamma in gammas:
    SVC_Gaussian = SVC(kernel='rbf', gamma = gamma)
    plot_decision_boundary(SVC_Gaussian, X, y)
Cs = [.1, 1, 5, 8, 10]
for C in Cs:
    SVC_Gaussian = SVC(kernel='rbf', gamma = 2, C = C)
    plot_decision_boundary(SVC_Gaussian, X, y)

#Comparing Kernel Executive times
import time
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

y = data.color == 'red'
X = data[data.columns[:-1]]
start = time.time() #save timestamp at this time
kwargs = {'kernel': 'rbf'}
svc = SVC(**kwargs)
nystroem = Nystroem(**kwargs)
sgd = SGDClassifier()
end = time.time() #save timestamp at this time
print('time to run',end - start)

start = time.time()
svc.fit(X, y)
end = time.time()
print('time to run svc',end - start)


#%%timeit in jupiter
start = time.time()
X_transformed = nystroem.fit_transform(X)
sgd.fit(X_transformed, y)
end = time.time() #save timestamp at this time
print('time to run sgd with nystroem',end - start)
print(X.shape)
print(y.shape)

start = time.time()
X_transformed = nystroem.fit_transform(X)
svc.fit(X_transformed, y)
end = time.time() #save timestamp at this time
print('time to run svc with nystroem',end - start)


X2 = pd.concat([X]*5)
y2 = pd.concat([y]*5)

#%%timeit in jupiter
start = time.time()
X2_transformed = nystroem.fit_transform(X2)
sgd.fit(X2_transformed, y2)
end = time.time() #save timestamp at this time
print('time to run sgd with nystroem',end - start)
print(X2.shape)
print(y2.shape)

#time for run sgd with nystroem is shorter than run with svc and suitable for run with big dataset

