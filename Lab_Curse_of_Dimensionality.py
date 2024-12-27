import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def make_circle(point=0):
    fig = plt.gcf()
    ax = fig.add_subplot(111, aspect='equal')
    fig.gca().add_artist(plt.Circle((0, 0), 1, alpha=.5))
    ax.scatter(0, 0, s=10, color="black")
    ax.plot(np.linspace(0, 1, 100), np.zeros(100), color="black")
    ax.text(.4, .1, "r", size=48)
    ax.set_xlim(left=-1, right=1)
    ax.set_ylim(bottom=-1, top=1)
    plt.xlabel("Covariate A")
    plt.ylabel("Covariate B")
    plt.title("Unit Circle")

    if point:
        ax.text(.55, .9, "Far away", color="purple")
        ax.scatter(.85, .85, s=10, color="purple")
    else:
        plt.show()

make_circle()
make_circle(1)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

# Create figure
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect("equal")

# Draw cube
r = [-1, 1]
for s, e in combinations(np.array(list(product(r,r,r))), 2):
    if np.sum(np.abs(s-e)) == r[1]-r[0]:
        ax.plot3D(*zip(s,e))

# Draw sphere on same axis
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x=np.cos(u)*np.sin(v)
y=np.sin(u)*np.sin(v)
z=np.cos(v)
ax.plot_wireframe(x, y, z, color="black");


# Draw a sample of data in two dimensions
sample_data = np.random.sample((5, 2))
print("Sample data:\n", sample_data, '\n')


def norm(x):
    ''' Measure the distance of each point from the origin.

    Input: Sample points, one point per row
    Output: The distance from the origin to each point
    '''
    return np.sqrt((x ** 2).sum(1))  # np.sum() sums an array over a given axis


def in_the_ball(x):
    ''' Determine if the sample is in the circle.

    Input: Sample points, one point per row
    Output: A boolean array indicating whether the point is in the ball
    '''
    return norm(x) < 1  # If the distance measure above is <1, we're inside the ball


for x, y in zip(norm(sample_data), in_the_ball(sample_data)):
    print("Norm = ", x.round(2), "; is in circle? ", y)



def what_percent_of_the_ncube_is_in_the_nball(d_dim,
                                              sample_size=10**4):
    shape = sample_size,d_dim
    data = np.array([in_the_ball(np.random.sample(shape)).mean()
                     for iteration in range(100)])
    return data.mean()

dims = range(2,15)
data = np.array(list(map(what_percent_of_the_ncube_is_in_the_nball,dims)))


for dim, percent in zip(dims,data):
    print("Dimension = ", dim, "; percent in ball = ", percent)


plt.plot(dims, data, color='blue')
plt.xlabel("# dimensions")
plt.ylabel("% of area in sphere")
plt.title("What percentage of the cube is in the sphere?")
plt.show()