import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import seaborn as sns
import matplotlib.pyplot as plt
import string
import matplotlib

sns.set_context('notebook')
sns.set_style('white')

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/example1.csv')
df.head(6)

#plot data points
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
plt.scatter(df['0'], df['1'])
for t, p in zip(string.ascii_uppercase, df.iterrows()):
    plt.annotate(t, (p[1][0] + 0.2, p[1][1]))
plt.show()

#apply DBScan
cluster = DBSCAN(eps = 3, min_samples=4)
cluster.fit(df)
print(f'DBSCAN found {len(set(cluster.labels_) - set([-1]))} cluster and {(cluster.labels_ == -1).sum()} points of noise.')

#plot after DBScan
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
plt.scatter(df['0'], df['1'], c=[['blue', 'red'][l] for l in cluster.labels_])
plt.scatter(0, 0, c='blue', alpha=0.2, s=90000)
plt.scatter(6, 0, c='red', alpha=0.2, s=9000)
for t, p in zip(string.ascii_uppercase, df.iterrows()):
    plt.annotate(t, (p[1][0] + 0.2, p[1][1]))
plt.show()


#Proving Someone Has Bad Handwriting
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/012.csv')
df.head()

friend_digits = df.iloc[:, df.columns != 'y'].to_numpy()
it = (x.reshape(8, 8) for x in friend_digits)
c = 3
fig, ax = plt.subplots(1, c, sharex='col', sharey='row')
for j in range(c):
    ax[j].axis('off')
    ax[j].set_title(f'Sample of friend\'s number {j}')
    ax[j].imshow(next(it))
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

#load data load_digit contains handwritten numbers from hundreds individuals across the United States:
digits, y = load_digits(return_X_y=True)
pd.DataFrame(digits).head()

plt.rcParams['figure.figsize'] = (8,6)
it = (x.reshape(8, 8) for x in digits)
r, c = 3, 5
fig, ax = plt.subplots(r, c, sharex='col', sharey='row')
for i in range(r):
    for j in range(c):
        ax[i, j].axis('off')
        ax[i, j].imshow(next(it))
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

#Concatnate friend digits into digits
data = np.r_[digits, friend_digits]
y = np.r_[y, df['y']]

embedding = TSNE(n_components=2, init= 'pca', n_iter=500, n_iter_without_progress=150, perplexity=10, random_state=0)
e_data = embedding.fit_transform(data)

plt.rcParams['figure.figsize'] = (20,15)
n = friend_digits.shape[0]
plt.scatter(
    e_data[:-n, 0],
    e_data[:-n, 1],
    marker='o',
    alpha=0.75,
    label='mnist data',
    s=100)
plt.scatter(
    e_data[-n:, 0],
    e_data[-n:, 1],
    marker='x',
    color='black',
    label='friend\'s data',
    alpha=1,
    s=100)
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']
#apply DBSCAN
cluster = DBSCAN(eps = 5, min_samples=20)
cluster.fit(e_data)
print(f'DBSCAN found {len(set(cluster.labels_) - set([-1]))} clusters and {(cluster.labels_ == -1).sum()} points of noise.')

#visualazition
plt.rcParams['figure.figsize'] = (20,15)
unique_labels = set(cluster.labels_)
n_labels = len(unique_labels)
cmap = matplotlib.pyplot.get_cmap('brg', n_labels)
for l in unique_labels:
    plt.scatter(
        e_data[cluster.labels_ == l, 0],
        e_data[cluster.labels_ == l, 1],
        c=[cmap(l) if l >= 0 else 'Black'],
        marker='ov'[l%2],
        alpha=0.75,
        s=100,
        label=f'Cluster {l}' if l >= 0 else 'Noise')
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

#predict label for friend digit
print("The predicted labels of our friend's handwriting:")
print(cluster.labels_[-3:])

r, c = 1, 5
plt.rcParams['figure.figsize'] = (4*r,4*c)
for label in unique_labels:
    cluster_data = data[cluster.labels_ == label]
    nums = cluster_data[np.random.choice(len(cluster_data), r * c, replace=False)]
    it = (x.reshape(8, 8) for x in nums)
    fig, ax = plt.subplots(r, c)
    ax = ax.reshape(r, c)
    plt.subplots_adjust(wspace=0.1, hspace=-0.69)
    fig.suptitle(f'Original data from cluster {label}', fontsize=20, y=0.545)
    for i in range(r):
        for j in range(c):
            ax[i, j].axis('off')
            ax[i, j].imshow(next(it))
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

#prove our friend's handwriting is unreadable
for i, (l,t) in enumerate(zip(cluster.labels_[-3:], y [-3:])):
    print('-'*30)
    print(f'Your friend\'s {i}th sample was categorized as being in cluster #{l}')
    if l == -1:
        print('(IE: Noise)')
    else:
        v,c = np.unique(y[cluster.labels_ == l], return_counts= True)
        mfreq = v[np.argmax(c)]
        ratio = c.max() / c.sum()
        print(f'Cluster {l} is {ratio * 100:.2f}% the number {mfreq}')
    print(f'Your friend\'s {i}th sample is supposed to be the number {t}')


#another dataset with
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/data/DBSCAN_exercises.csv')
df.head()
plt.scatter(df['x'], df['y'])
plt.show()
#find cluster
eps = 2
min_samples=10
cluster = DBSCAN(eps=eps, min_samples=min_samples)
cluster.fit(df)
print(len(set(cluster.labels_) - {1}))
#Find the % of data marked as noise
c, p = np.unique(cluster.labels_, return_counts = True)
noise_percent = (p[0]*100/p.sum())
print(f'Percentage of noise is {noise_percent} %')

unique_labels = set(cluster.labels_)
n_labels = len(unique_labels)
cmap = matplotlib.pyplot.get_cmap('brg', n_labels)
for l in unique_labels:
    plt.scatter(
        df[cluster.labels_ == l]['x'],
        df[cluster.labels_ == l]['y'],
        c=[cmap(l) if l >= 0 else 'Black'],
        marker='ov'[l%2],
        alpha=0.75,
        s=100,
        label=f'Cluster {l}' if l >= 0 else 'Noise')
plt.legend(bbox_to_anchor=[1, 1])
plt.show()
plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']