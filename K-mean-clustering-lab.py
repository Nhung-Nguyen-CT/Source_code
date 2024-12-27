import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from  sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
import seaborn as  sns
from PIL import Image
import requests
from io import BytesIO


plt.rcParams['figure.figsize'] = [6,6]
sns.set_style("whitegrid")
sns.set_context("talk")

def display_cluster(X, km = [],num_clusters = 0):
    color = 'brgcmyk'
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1], c = color[0], alpha = alpha, s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_ == i, 0], X[km.labels_ == i,1], c = color[i],  alpha = alpha, s =  s)
            plt.scatter(km.cluster_centers_[i][0], km.cluster_centers_[i][1], c = color[i],marker = 'X', s = 100)


angle = np.linspace(0,2*np.pi,20, endpoint = False)
X = np.append([np.cos(angle)],[np.sin(angle)],0).transpose()
display_cluster(X)

#clustering with random state = 10, n cluster = 2
num_clusters = 2
km = KMeans(n_clusters=num_clusters, random_state= 10, n_init = 1)
km.fit(X)
display_cluster(X, km, num_clusters= num_clusters)

num_clusters = 2
km = KMeans(n_clusters=num_clusters, random_state= 20, n_init = 1)
km.fit(X)
display_cluster(X, km, num_clusters= num_clusters)


#determining optimum number of cluster
n_samples = 1000
n_bins = 4
centers = [(-3, -3), (0, 0), (3, 3), (6, 6)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
display_cluster(X)

#run K mean for 7 cluster
num_clusters = 7
km = KMeans(n_clusters=num_clusters)
km.fit(X)
display_cluster(X,km,num_clusters)
#run K mean for 4 cluster
num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(X)
display_cluster(X,km,num_clusters)

km.inertia_#sum of squared error between each point and its cluster center (variation)
inertia = []
list_num_clusters = list(range(1,11))
for num_clusters in list_num_clusters:
    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(list_num_clusters, inertia)
plt.scatter(list_num_clusters, inertia)
plt.xlabel('Number of Cluster')
plt.ylabel('Inertia')
plt.show()
#n = 4 is the elbow of the curve

#Clustering Colors from an Image
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%201/images/peppers.jpg'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img_np = np.array(img)
plt.imshow(img_np)
plt.axis('off')
plt.show()

img_np.shape # size of image
#The image above has 480 pixels in height and 640 pixels in width.  Each pixel has 3 values that represent how much red, green and blue it has. Below you can play with different combinations of RGB to create different colors.
# In total, you can create 256^3 = 16,777,216 unique colors.
# assign values for the RGB.  Each value should be between 0 and 255
R = 35
G = 95
B = 131
plt.imshow([[np.array([R,G,B]).astype('uint8')]])
plt.axis('off')
#reshape the image into a table that has a pixel per row and each column represents the red, green and blue channel.
img_flat = img_np.reshape(-1, 3)
img_flat[:5,:]
img_flat.shape

#run K means with 8 clusters
kmeans = KMeans(n_clusters=8, random_state=0).fit(img_flat)
#replace each row with its closest cluster center
img_flat2 = img_flat.copy()
for i in np.unique(kmeans.labels_):
    img_flat2[kmeans.labels_ == i, :] = kmeans.cluster_centers_[i]

img2 = img_flat2.reshape(img_np.shape)
plt.imshow(img2)
plt.axis('off')
#look like the dimension of image is reduced


#function that receives the image and number of clusters (k), and returns (1) the image quantized into k colors, and (2) the inertia
def image_cluster(img,k):
    img_flat=  img.reshape(img.shape[0] * img.shape[1],3)
    kmeans = KMeans( n_clusters=k, random_state=0).fit(img_flat)
    img_flat2 = img_flat.copy()
    for i in np.unique(kmeans.labels_):
        img_flat2[kmeans.labels_ == i, :] = kmeans.cluster_centers_[i]
    img2 = img_flat2.reshape(img.shape)
    return img2,kmeans.inertia_

k_vals = list(range(2,21,2))
img_list = []
inertia = []
for k in k_vals:
#    print(k)
    img2, ine = image_cluster(img_np,k)
    img_list.append(img2)
    inertia.append(ine)

# Plot to find optimal number of clusters
plt.plot(k_vals,inertia)
plt.scatter(k_vals,inertia)
plt.xlabel('k')
plt.ylabel('Inertia')

#plot a grid all the images for the different k values
plt.figure(figsize=[10,20])
for i in range(len(k_vals)):
    plt.subplot(5,2,i+1)
    plt.imshow(img_list[i])
    plt.title('k = '+ str(k_vals[i]))
    plt.axis('off');