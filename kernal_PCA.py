import numpy as np
import pandas as pd
from itertools import accumulate

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


sns.set_context('notebook')
sns.set_style('white')


def plot_proj(A, v, y, name=None):
    plt.scatter(A[:, 0], A[:, 1], label='data', c=y, cmap='viridis')
    # plt.plot(np.linspace(A[:,0].min(),A[:,0].max()),np.linspace(A[:,1].min(),A[:,1].max())*(v[1]/v[0]),color='black',linestyle='--',linewidth=1.5,label=name)
    plt.plot(np.linspace(-1, 1), np.linspace(-1, 1) * (v[1] / v[0]), color='black', linestyle='--', linewidth=1.5,
             label=name)
    # Run through all the data
    for i in range(len(A[:, 0]) - 1):
        # data point
        w = A[i, :]
        # projection
        cv = (np.dot(A[i, :], v)) / np.dot(v, np.transpose(v)) * v
        # line between data point and projection
        plt.plot([A[i, 0], cv[0]], [A[i, 1], cv[1]], 'r--', linewidth=1.5)
    plt.plot([A[-1, 0], cv[0]], [A[-1, 1], cv[1]], 'r--', linewidth=1.5, label='projections')
    plt.legend()
    plt.show()


#apply PCA for visualization
#transform dataset before apply PCA
# Create the toy dataset
X, y = make_circles(n_samples=1000, factor=0.01, noise=0.05, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
#visualize dataset before apply PCA
_, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,cmap='viridis')
train_ax.set_xlabel("$x_{0}$")
train_ax.set_ylabel("$x_{1}$")
train_ax.set_title("Training data")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,cmap='viridis')
test_ax.set_xlabel("$x_{0}$")
test_ax.set_ylabel("$x_{1}$")
test_ax.set_title("Test data")
plt.show()

pca = PCA(n_components=2)
score_pca = pca.fit(X_train).transform(X_test)
#plot score_pca by y_test
plt.scatter(score_pca[:, 0], score_pca[:, 1], c=y_test,label="data points", cmap='viridis')
plt.quiver([0,0],[0,0], pca.components_[0,:], pca.components_[1,:], label="eigenvectors")
plt.xlabel("$x_{0}$")
plt.ylabel("$x_{1}$")
plt.legend(loc='center right')
plt.show()
#plot the direction of projection onto the first principal component for each data point
plot_proj(X_train,pca.components_[0,:],y_train,"first principal component")
#plot out the actual points of projection onto the first component, dataset is not linear => can not seperate by a plane
plt.scatter(score_pca [:,0],np.zeros(score_pca[:,0].shape[0]),c=y_test,cmap='viridis')
plt.title("Projection of testing data\n using PCA")
plt.show()
#plot dataset onto the second component will be the same results cause dataset is symmetric

#fit the training set to Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression().fit(X_train, y_train)
print(str.format("Test set  mean accuracy score for for PCA: {}", lr.score(X_test, y_test)))

#transform dataset into a higher Dimension and then apply PCA
# transformed data = [x1,x2, x1^2 + x2^2] where x = [x1,x2]
PHI_train =  np.concatenate((X_train,  (X_train**2).sum(axis=1).reshape(-1,1)), axis =1)
PHI_test =  np.concatenate((X_test,  (X_test**2).sum(axis=1).reshape(-1,1)), axis =1)
#plot transformed data in 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(PHI_train[:,0], PHI_train[:,1],  PHI_train[:,2], c=y_train, cmap='viridis', linewidth=0.5);
ax.set_xlabel('$x_{1}$')
ax.set_ylabel('$x_{2}$')
ax.set_zlabel('$x_{1}^2+x_{2}^2$')
plt.show()
#apply PCA  with 3 components
pca = PCA(n_components=3)
score_poly = pca.fit(PHI_train).transform(PHI_test)
#plot out the datapoints projection onto the first PC
plt.scatter(score_poly [:,0],np.zeros(score_poly[:,1].shape[0]),c=y_test,cmap='viridis')
plt.title("Projection of testing data\n using PCA")
plt.show()
#plot out the datapoints onto top 2 PC:
plt.scatter(score_poly[:,0], score_poly[:,1],c=y_test, cmap='viridis', linewidth=0.5);
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.title("Projection onto PCs")
plt.show()
#then, build a Logistic Regression model on the training set, accuracy improved
lr= LogisticRegression().fit(PHI_train, y_train)
print(str.format("Test set  mean accuracy score for for Kernal PCA: {}", lr.score(PHI_test, y_test)))


#apply kernel PCA for dataset
kernel_pca =  KernelPCA(kernel='rbf', gamma=10, fit_inverse_transform= True,  alpha = 0.1)
kernel_pca.fit(X_train)
score_kernel_pca = kernel_pca.transform(X_test)
#plot PC and eigenvalue
plt.plot(kernel_pca.eigenvalues_)
plt.title("Principal component and their eigenvalues")
plt.xlabel("nth principal component")
plt.ylabel("eigenvalue magnitude")
plt.show()
#plot projection of datapoints onto the first 2 PC
plt.scatter(score_kernel_pca[:,0],score_kernel_pca[:,1] ,c=y_test,cmap='viridis')
plt.title("Projection onto PCs (kernel)")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.show()

#comparation of reconstruction ability of PCA and KernelPCA
#PCA perfectly reconstruct the original data but PCA can not
X_hat_kpca = kernel_pca.inverse_transform(kernel_pca.transform(X_test))
pca = PCA(n_components=2)
pca.fit(X_train)
X_hat_pca = pca.inverse_transform(pca.transform(X_test))
#plot original dataset, inverse of PCA and inverse of KernelPCA
plt.scatter(X_test[:,0],X_test[:,1] ,c=y_test,cmap='viridis')
plt.title("Original data")
plt.show()

plt.scatter(X_hat_kpca[:,0],X_hat_kpca[:,1] ,c=y_test,cmap='viridis')
plt.title("Inversely Transformed Data (Kernel PCA)")
plt.show()

plt.scatter(X_hat_pca[:,0],X_hat_pca[:,1] ,c=y_test,cmap='viridis')
plt.title("Inversely Transformed Data (PCA)")
plt.show()
#we can compute the mean square error of the inverse transformation
print("Mean squared error for Kernel PCA is:",((X_test-X_hat_kpca)**2).mean())
print("Mean squared error PCA is:" ,((X_test-X_hat_pca)**2).mean())


##USING KERNEL PCA TO PREDICT IF YOU'RE THE RICHEST PERSON IN THE WORLD
df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/billionaires.csv',index_col="Unnamed: 0")
df.head()
df.shape
print(df.info())
for col in df:
    print(str.format("{} has {} unique values.", col, len(df[col].unique())))
#check  number of unique values of every variables that is not to length of df
print(df['rank'].value_counts())
for column in ['country', 'industry']:
    df[column].hist(bins=len(df[column].unique()))
    plt.xticks(rotation='vertical')
    plt.show()
sns.pairplot(df[['age', 'rank']])
df[['age', 'rank']].corr()
# don't use variable 'name', 'network', 'source', industry
B_names,networths,sources,industrys=df['name'],df['networth'],df['source'],df['industry']
B_names,networths,sources,industrys
#assign y
y=df['rank']
y.head()
#assign X
df.drop(columns=['name','networth','source'],inplace=True)
df.head()
#onehot encode for input
one_hot = ColumnTransformer(transformers=[("one_hot", OneHotEncoder(), ['country','industry']) ],remainder="passthrough")
data=one_hot.fit_transform(df)
#convert to dataset with label names:
names=one_hot.get_feature_names_out()
column_names=[name[name.find("__")+2:] for name in names]
new_data=pd.DataFrame(data.toarray(),columns=column_names)
new_data.head()
#apply kernel PCA for new data
kernel_pca = KernelPCA(kernel="rbf" ,fit_inverse_transform=True, alpha=0.1)
kernel_score=kernel_pca.fit_transform(new_data)
#use PCA to improve visualization
ranking = 13
fig, ax = plt.subplots()
sc=ax.scatter(kernel_score[:,0],kernel_score[:,1] ,c=y,cmap='viridis')
fig.colorbar(sc, orientation='vertical')
ax.annotate(industrys[ranking], (kernel_score[ranking,0],kernel_score[ranking,1]))
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.title("Projection on the top 2 \nprincipal components (colored by ranking)")
plt.show()
#plot 3 component - 3 dimensional projection
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc=ax.scatter(kernel_score[:,0], kernel_score[:,1],  kernel_score[:,2], c=y, cmap='viridis', linewidth=0.5);
fig.colorbar(sc, orientation='horizontal')
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')
plt.show()

#apply PCA
pca = PCA()
score_pca = pca.fit_transform(new_data)
#plot 1D projection
fig, ax = plt.subplots()
sc=ax.scatter(score_pca[:,0],np.zeros(score_pca[:,1].shape ),c=y,cmap='viridis')
ax.set_title('1-dimensional projection space\n (1st PC)')
fig.colorbar(sc, orientation='vertical')
plt.show()
#plot 2D projection
fig, ax = plt.subplots()
sc=ax.scatter(score_pca[:,0],score_pca[:,1] ,c=y,cmap='viridis')
fig.colorbar(sc, orientation='vertical')
ax.set_title('2-dimensional projection space\n (Top 2 PCs)')
plt.xlabel("1st PC")
plt.ylabel("2nd PC")
plt.show()
#plot 3D projection
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc=ax.scatter(score_pca[:,0], score_pca[:,1],  score_pca[:,2], c=y, cmap='viridis', linewidth=0.5);
ax.set_title('3-dimensional projection space\n (Top 3 PCs)')
ax.set_xlabel('1st PC')
ax.set_ylabel('2nd PC')
ax.set_zlabel('3rd PC')
plt.show()


#use kernel PCA to improve Prediction
from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(kernel_score, y, test_size=0.4, random_state=0)
lr = Ridge(alpha=0).fit(X_train, y_train)
print(str.format("Test set R^2 score for Kernel PCA: {}", lr.score(X_test, y_test)))
#regression with kernel PCA
X_train, X_test, y_train, y_test = train_test_split(kernel_score, y, test_size=0.4, random_state=0)
lr = Ridge(alpha=0).fit(X_train, y_train)
print(str.format("Test set R^2 score for Kernel PCA: {}", lr.score(X_test, y_test)))
#regression with PCA
X_train, X_test, y_train, y_test = train_test_split(score_pca, y, test_size=0.40, random_state=0)
lr= Ridge(alpha=0).fit(X_train, y_train)
print(str.format("Test set R^2 score for PCA: {}", lr.score(X_test, y_test)))

