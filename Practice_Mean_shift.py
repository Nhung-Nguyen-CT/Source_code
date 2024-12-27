import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


#Mean Shift Applied to the Titanic Dataset

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%202/titanic.csv")
df.head()
df=df.drop(columns=['Name','Ticket','Cabin','PassengerId','Embarked'])
df.loc[df['Sex']!='male','Sex']=0
df.loc[df['Sex']=='male','Sex']=1
df.head()
df.isna().sum()
df['Age'].fillna(df['Age'].mean(),inplace=True)
X=df.apply(lambda x: (x-x.mean())/(x.std()+0.0000001), axis=0)
X.head()

#apply mean shifting  to df
#set up auto detection bandwidth
bandwidth = estimate_bandwidth(X)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
#attach predict labels into X and df
X['cluster']=ms.labels_
df['cluster']=ms.labels_
#group by cluster to check with cluster have a larger chance of survival
df.groupby('cluster').mean().sort_values(by = ['Survived'], ascending=False)

#How Mean Shift Works (Optional)
#Kernel Density Estimation
def gaussian(d, h):
    return np.exp(-0.5*((d/h))**2) / (h*math.sqrt(2*math.pi))

plt.plot(x,kernel_1,label='h=1')
plt.plot(x,kernel_2,label='h=3')
plt.plot(s,0,'x',label="$x_{1}$=1")
plt.hist(s, 10, facecolor='blue', alpha=0.5,label="Histogram")
plt.xlabel('x')
plt.legend()
plt.show()

#we generate the KDE with bandwith  ℎ for set of point  𝑥𝑖, stored in the NumPy array S
def kernel_density(S, x, h=1):
    density = np.zeros((200))
    for s in S:
        # Determine the distance and kernel for each point
        dist = np.sqrt(((x - s) ** 2))
        kernel = gaussian(dist, h)
        # Find the sum
        density += kernel
    # Normalize the sum
    density = density / density.sum()
    return density


S=np.zeros((200))
S[0:100] = np.random.normal(-10, 1, 100)
S[100:200]=np.random.normal(10, 1, 100)
plt.plot(S,np.zeros((200)),'x')
plt.xlabel("$x_{i}$")
plt.show()

x = np.linspace(S.min()-3, S.max()+3, num=200)
density=kernel_density(S,x)

plt.plot(x,density,label=" KDE")
plt.plot(S,np.zeros((200,1)),'x',label="$x_{i}$")
plt.xlabel('x')
plt.legend()
plt.show()

#We can run the algorithm for three iterations, each point should converge to the cluster centers :
Xhat = np.copy(S.reshape(-1, 1))
S_ = S.reshape(-1, 1)

for k in range(3):
    plt.plot(x, density, label=" KDE")
    plt.plot(Xhat, np.zeros((200, 1)), 'x', label="$\hat{x}^{k}_i$,k=" + str(k))
    plt.xlabel('x')
    plt.legend()
    plt.show()

    for i, xhat in enumerate(Xhat):
        dist = np.sqrt(((xhat - S_) ** 2).sum(1))
        weight = gaussian(dist, 2.5)
        Xhat[i] = (weight.reshape(-1, 1) * S_).sum(0) / weight.sum()
