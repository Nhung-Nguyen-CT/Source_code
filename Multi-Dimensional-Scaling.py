import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cityblock, cosine
import sklearn.metrics.pairwise
import seaborn as sns
import folium
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from matplotlib import offsetbox
from sklearn.manifold import MDS

def plot_points(df,color="red",title=""):
    X=df['lon']
    Y=df['lat']
    annotations=df.index
    plt.figure(figsize=(8,6))
    plt.scatter(X,Y,s=100,color=color)
    plt.title(title)
    plt.xlabel("lat")
    plt.ylabel("log")
    for i, label in enumerate(annotations):
        plt.annotate(label, (X[i], Y[i]))
    plt.axis('equal')
    plt.show()


def plot_embedding(X, title, ax):
    X = MinMaxScaler().fit_transform(X)
    for digit in digits.target_names:
        ax.scatter(
            *X[y == digit].T,
            marker=f"${digit}$",
            s=60,
            color=plt.cm.Dark2(digit),
            alpha=0.425,
            zorder=2,
        )
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(X.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title(title)
    ax.axis("off")

distance=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/distance.csv').set_index('name')
distance.head(8)
embedding = MDS(dissimilarity='precomputed', n_components=2, random_state=0, max_iter=300,eps= 1e-3)
X_transformed = embedding.fit_transform(distance)
df_t=pd.DataFrame(X_transformed , columns=["lon","lat"], index=distance.columns)
df_t.head(8)

embedding.embedding_
embedding.stress_
embedding.dissimilarity_matrix_

#create a df with orginal to compare lon and lat between orginal and transformed df
df = pd.DataFrame({
   'lon':[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],
   'lat':[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],
   'name':['Buenos Aires', 'Paris', 'Melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador']})
df=df.set_index('name')
df.head(10)
#compare by plotting
plot_points(df,title='original dataset')
plot_points(df_t,color='blue',title='Embedded Coordinates using Euclidean distance ')

#difference in distance method
from scipy.spatial.distance import squareform, pdist
distance=pd.DataFrame(squareform(pdist(df.iloc[:, 1:])), columns=df.index, index=df.index)
dist=['cosine','cityblock','seuclidean','sqeuclidean','cosine','hamming','jaccard','chebyshev','canberra','braycurtis']
plot_points(df,title='original dataset')
#recalculate distance with different method and compare dataframe and orginal position
for d in dist:
    distance=pd.DataFrame(squareform(pdist(df.iloc[:, 1:],metric=d)), columns=df.index, index=df.index)
    embedding =  MDS(dissimilarity='precomputed', random_state=0,n_components=2)
    X_transformed = embedding.fit_transform(distance)
    df_t=pd.DataFrame(X_transformed , columns=df.columns, index=df.index)
    plot_points(df_t,title='Embedded Coordinates using '+d ,color='blue')

#NON-METRIC MDS - preserve ranking between points
metric = False
embedding =  MDS(dissimilarity='precomputed',n_components=2,metric=metric,random_state=0)
X_transformed = embedding.fit_transform(distance)
df_t=pd.DataFrame(X_transformed , columns=df.columns, index=df.index)
df_t.head(8)
#plot non-metric MDS vs original
plot_points(df,title='original dataset')
plot_points(df_t,color='blue',title='Embedded Coordinates using Euclidean distance ')
#different distances to Non-metric MDS
dist=['cosine','cityblock','seuclidean','sqeuclidean','cosine','hamming','jaccard','chebyshev','canberra','braycurtis']
plot_points(df,title='original dataset')
metric=False
for d in dist:
    distance=pd.DataFrame(squareform(pdist(df.iloc[:, 1:],metric=d)), columns=df.index, index=df.index)
    embedding =  MDS(dissimilarity='precomputed', random_state=0,n_components=2,metric=False)
    X_transformed = embedding.fit_transform(distance)
    df_t=pd.DataFrame(X_transformed , columns=df.columns, index=df.index)
    plot_points(df_t,title='Embedded Coordinates using '+d ,color='blue')


#DIMENSION REDUCTION WITH MDS
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape

fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(6, 6))
for idx, ax in enumerate(axs.ravel()):
    ax.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
    ax.axis("off")
    fig.suptitle("A selection from the 64-dimensional digits dataset", fontsize=16)

embedding=MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2)

X_transformed=embedding.fit_transform(X)
fig, ax = plt.subplots()
plot_embedding(X_transformed, "Metric MDS ", ax)
plt.show()