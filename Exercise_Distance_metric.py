import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import scipy
from scipy.spatial.distance import euclidean, cityblock, cosine
import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import paired_euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

# This function will allow us to find the average distance between two sets of data
def avg_distance(X1, X2, distance_func):
    from sklearn.metrics import jaccard_score
    res = 0
    for xi1 in X1:
        for xi2 in X2:
            if distance_func == jaccard_score: # the jaccard_score function only returns jaccard_similarity
                res += 1 - distance_func(xi1, xi2)
            else:
                res += distance_func(xi1, xi2)
    return res/(len(X1)*len(X2))

#pairwise distance
def avg_pairwise_distance(X1, X2, distance_func):
    return sum(map(distance_func, X1, X2)) / min(len(X1), len(X2))

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/iris.csv')
df.head()
species = df['species'].unique()
print(species)

#view the three species of irises' data in 3D
attrs = ['sepal_length', 'sepal_width', 'petal_length']
markers = ['o', 'v', '^']
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
for specie, marker in zip(species, markers):
    specie_data = df.loc[df['species'] == specie][attrs]
    xs, ys, zs = [specie_data[attr] for attr in attrs]
    ax.scatter(xs, ys, zs, marker = marker)
plt.show()

setosa_data = df.loc[df['species'] == 'setosa'][attrs].to_numpy()
versicolor_data = df.loc[df['species'] == 'versicolor'][attrs].to_numpy()
virginica_data = df.loc[df['species'] == 'virginica'][attrs].to_numpy()
setosa_data.shape

avg_distance(setosa_data, versicolor_data, euclidean)
avg_distance(setosa_data, virginica_data, euclidean)


X = np.array([[0, 0]], dtype=float)
Y = np.array([[3, 4]], dtype=float)
paired_euclidean_distances(X, Y).mean()
avg_pairwise_distance(X, Y, euclidean)

M,N = setosa_data.shape
print(f'{M} points and each column is {N} dimensions')

row_dist = paired_euclidean_distances(setosa_data, versicolor_data)
row_dist

cityblock([1, 1], [-2, 2])
avg_distance(setosa_data, setosa_data, cityblock)
avg_distance(setosa_data, versicolor_data, cityblock)
avg_distance(setosa_data, virginica_data, cityblock)

#manhattan distance
X = np.array([[1, 1]])
Y = np.array([[-2, 2]])

manhattan_distances(X,Y)

#Cosine distance
cosine([1, 1], [-1, -1])

#dataset auto-mpg.data
df = pd.read_csv(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/auto-mpg.data',
    header=None, delim_whitespace=True,
    names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'])
df.head()

df['car_name'] = df['car_name'].str.split(n=1).apply(lambda lst: lst[0]).replace('chevrolet', 'chevy')
df.rename(columns={'car_name': 'make'}, inplace=True)
df = df[['mpg', 'weight', 'make']]
df.head()

#Analyzing Distance metrics with DBScan

dfn = df[['mpg', 'weight']]
df[['mpg', 'weight']] = (dfn-dfn.min())/(dfn.max()-dfn.min())
df.head()

chevy = df.loc[df['make'] == 'chevy']
honda = df.loc[df['make'] == 'honda']
plt.scatter(chevy['mpg'], chevy['weight'], marker='o', label='chevy')
plt.scatter(honda['mpg'], honda['weight'], marker='^', label='honda')
plt.xlabel('mpg')
plt.ylabel('weight')
plt.legend()
plt.show()

#analyze using cosine distance
chevy_data = chevy[['mpg', 'weight']].to_numpy()
honda_data = honda[['mpg', 'weight']].to_numpy()
avg_distance(chevy_data, chevy_data, cosine)
avg_distance(honda_data, honda_data, cosine)
avg_distance(honda_data, chevy_data, cosine)


df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/synthetic_clustering.csv')
df.head()

plt.scatter(df['x'], df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#DBscan using Euclidean
dbscan = DBSCAN( eps = 0.1, metric= euclidean)
dbscan.fit(df)
colors = np.random.random(size = 3*(dbscan.labels_.max()+1)).reshape(-1,3)
plt.scatter(df['x'], df['y'], c = [colors[l] for l in dbscan.labels_])

#Dbscan with Manhattan distance
dbscan = DBSCAN(eps=0.1, metric=cityblock)
dbscan.fit(df)
colors = np.random.random(size=3*(dbscan.labels_.max()+1)).reshape(-1, 3)
plt.scatter(df['x'], df['y'], c=[colors[l] for l in dbscan.labels_])
plt.show()

#Dbscan using Cosine
dbscan = DBSCAN(eps=0.1, metric=cosine)
dbscan.fit(df)
colors = np.random.random(size=3*(dbscan.labels_.max()+1)).reshape(-1, 3)
plt.scatter(df['x'], df['y'], c=[colors[l] for l in dbscan.labels_])
plt.show()


#JACCARD DISTANCE
df = pd.read_csv(
    'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/breast-cancer.data',
    header=None,
    names=['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'])
df.head()
print(sorted(df['age'].unique()))
print(df.age.value_counts())

OH = OneHotEncoder()
X = OH.fit_transform(df.loc[:, df.columns != 'age']).toarray()
print(f"By using onehot encoding, we obtained a 2d array with shape {X.shape} that only has value 0 and 1 ")

X30to39 = X[df[df.age == '30-39'].index]
X60to69 = X[df[df.age == '60-69'].index]
X30to39.shape, X60to69.shape
avg_distance(X30to39, X30to39, jaccard_score)


#Exercise 1 - Jaccard distance
sentence1 = 'Hello everyone and welcome to distance metrics'
sentence2 = 'Hello world and welcome to distance metrics'
s1set = set(sentence1.split())
s2set = set(sentence2.split())
ans = len(s1set.intersection(s2set)) / len(s1set.union(s2set))