import numpy as np
import matplotlib.pylab as plt

from sklearn.decomposition import PCA
from sympy import Matrix, init_printing,Symbol
from numpy.linalg import qr,eig,inv,matrix_rank,inv, norm
from scipy.linalg import null_space

init_printing()


def plot_2da(dict_):
    for key, value in dict_.items():
        plt.scatter(value[:, 0], value[:, 1], label=key)
    plt.legend()
    plt.show()


def plot_2db(dict_):
    for key, value in dict_.items():
        if value.shape[0] > 2:
            plt.scatter(value[:, 0], value[:, 1], label=key)
        else:
            print(value)
            plt.quiver([0], [0], value[:, 0], value[:, 1], label=key)
    plt.legend()
    plt.show()

#BASICS OF MATRICES
A = np.array([[2,-3], [4,7]])
Matrix(A)
a1 = A[:, 0]
a2 = A[:,1]
AT = A.T
Matrix(AT)

#RANK OF A MATRIX
matrix_rank(A)

#PLOT THE COLUMNS OF MATRIX A AS VECTORS
fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],A[0,0], A[1,0],scale=30,label="$\mathbf{a}_{1}$")
ax.quiver([0, 0],[0, 0],A[0,1], A[1,1],scale=30,label="$\mathbf{a}_{2}$",color='red')
plt.title("columns of $\mathbf{A}$ ")
plt.legend()
plt.show()

F=np.array([[2,4],[4,8]])
matrix_rank(F)
#PLOT NOT RANK-DEFICIENT MATRIX F (RANK(F) = 1)
fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],F[0,1], F[1,1],scale=30,label="$\mathbf{f}_{2}$",color='red')
ax.quiver([0, 0],[0, 0],F[0,0], F[1,0],scale=30,label="$\mathbf{f}_{1}$")
plt.title("columns of $\mathbf{F}$ ")
plt.legend()
plt.show()

#CHECK PROPERITIES OF RANK DEFICIENCY
G=np.array([[2,4,6],[6,4,2],[16,16,16]])
matrix_rank(G)

F=np.array([[1,2],[1,-2],[-1,1]])
Matrix(F)
ax = plt.figure().add_subplot(projection='3d')
p=null_space(F.T)
xx, yy = np.meshgrid(np.arange(-3,3,0.1), np.arange(-3,3,0.1))
z=(p[0]*xx+p[1]*yy)/p[2]
ax.plot_surface(xx, yy, z, alpha=0.1)
ax.quiver([0,0], [0,0], [0,0], F[0,:], F[1,:], F[2,:])
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
plt.show()
matrix_rank(F)

#FROBENIUS NORM OF A MATRIX
norm(A)

#MATRIX ADDITION
B=np.array([[1,1],[1,-1]])
Matrix(B)
C=A+B
Matrix(C)

C=np.random.randn(2,2)
S=C+C.T
Matrix(S)
Matrix(S.T)

#DIAGONAL MATRIX is matrix that have elements off the main diagonal are all zero
Matrix(np.diag(np.array([1,2,3])))
#IDENTITY MATRIX: the main diagonal = 1 and all entries off the main diagonal = 0
Matrix(np.eye(3).astype(int))

#MATRIX AND VECTOR MULTIPLICATION
#DOT PRODUCT
a=np.array([1,1])
b=np.array([1,2])
Matrix(a)
a.ndim
a.T@b

one=np.ones(2)
a.T @ one

#THE OUTER PRODUCT
u= np.array([[1],[2],[3],[4]])
v= np.array([[0],[1],[2],[3],[6]])
Matrix(u@v.T)

#Exercise 2: create matrix with 4 columns
u= np.array([[1],[2],[3],[4]])
UT = u @ (np.array([[0,1,0,1]]))
print('The rank is: ' + str(matrix_rank(np.array(UT).astype(float))))
UT

#Matrix and vector multiplication
x=np.array([1,1])
x
A=np.array([[0,-1],[1,0]])
Matrix(A)
b=A@x
Matrix(b)

fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],A[0,0], A[1,0],scale=10,label="$\mathbf{a}_{1}$")
ax.quiver([0, 0],[0, 0],A[0,1], A[1,1],scale=10,label="$\mathbf{a}_{2}$")
ax.quiver([0,0],[0,0],b[0], b[1],scale=10,label="$\mathbf{b}$",color='r')
ax.quiver([0,0],[0,0],x[0], x[1],scale=10,label="$\mathbf{x}$",color='b')
ax.set_xlim([-10,10])
ax.set_ylim([-5,10])
fig.legend()
plt.show()

A=np.array([[-1,1],[1,2]])
Matrix(A)
b=A@x
Matrix(b)
fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],A[0,0], A[1,0],scale=10,label="$\mathbf{a}_{1}$")
ax.quiver([0, 0],[0, 0],A[0,1], A[1,1],scale=10,label="$\mathbf{a}_{2}$")
ax.quiver([0,0],[0,0],b[0], b[1],scale=10,label="$\mathbf{b}$",color='r')
ax.quiver([0,0],[0,0],x[0], x[1],scale=10,label="$\mathbf{x}$",color='b')
ax.set_xlim([-10,10])
ax.set_ylim([-5,10])
fig.legend()
plt.show()

#MULTIPLYING MATRICES
C=A@B
Matrix(C)
#inverted A, if a matrixx is full rank, it can be inverted A^-1
A_inv = inv(A)
# A @ A^-1 = I
I = np.round(A@A_inv, 8)
#multiply any square matrix with an Identity matrix, you get the original Matrix
Matrix((A@I))
x_ = b @ inv(A)
print(x)
print(x_)

#orthogonal matrix: matrix that Q.Q^T = Q^T.Q = I
Q=np.array([[1,1],[1,-1]])*2**(-1/2)
Q
I=Q@Q.T
Matrix(I)

fig, ax = plt.subplots(figsize = (12, 7))
ax.quiver([0, 0],[0, 0],Q[0,0], Q[1,0],scale=10,label="$\mathbf{q}_{1}$")
ax.quiver([0, 0],[0, 0],Q[0,1], Q[1,1],scale=10,label="$\mathbf{q}_{2}$",color='red')
plt.title("columns of $\mathbf{Q}$ ")
plt.legend()
plt.show()

samples=200

u=np.array([[1.0,1.0],[0.10,-0.10]])/(2)**(0.5)

X_=np.dot(4*np.random.randn(samples,2),u)+10
X_[0:5]
dict_={"design matrix samples":X_}
plot_2da(dict_)

N,D=X_.shape
print("number of smaples {}, dimensions is {}".format(N,D))
#find mean of matrix
mean=(np.ones((1,N))/N)@X_
#mean calculate by numpy
X_.mean(axis=0)

#Matrix multiplication using no_mean and X_
I = np.identity(N)
col1 = np.ones((1,N))
row1 = np.ones((N,1))/N
no_mean = (I - row1@col1)
X = no_mean@X_
print("mean of X",X.mean(axis=0))
dict_={"original data":X_,"zero mean data":X,"mean of original data":mean}
plot_2da(dict_)

#empirical covariance  matrix: diagonal: variance ; off- diagonal: co-variance
C = X.T @ X/N
Matrix(C)
#check if C is full rank matrix
matrix_rank(C)


#Eigen decomposit
# Eigenvectors and Eigenvalues
#if a matrix is full rank, can apply Eigen factorization and Eigen decomposit: Eigenvectors and Eigenvalues
eigen_values, eigen_vectors = eig(A)
Matrix(np.diag(eigen_values))
#retrieive the orginal matrix when having eigenvalues and eigenvectors
A = np.round(eigen_vectors@np.diag(eigen_values)@inv(eigen_vectors),8)
Matrix(A)

#FACTORIZATION FOR PCA
eigen_values, eigen_vectors = eig(C)
v = eigen_vectors[:, np.argmax(eigen_values)].reshape(-1,1)
v
#projection X by v
Z = X@v

#PCA by using sklearn package
pca = PCA(n_components=1)
Z_sklearn = pca.fit_transform(X_)
# This will print True if the vectors are identical (ignoring the sign) and False otherwise
if np.isclose(Z,Z_sklearn).min():
    print(np.isclose(Z,Z_sklearn).min())
else:
    print(np.isclose(Z,-Z_sklearn).min())

#use PCA in sklearn to transform X and reverse the transformed X to  X
pca =  PCA(n_components=1)
X_transformed = pca.fit_transform(X)
X_ = pca.inverse_transform(X_transformed)

#transform data back to its original
Xhat =Z@v.T

dict_ = {"Sklearn inverse_transform": X_, "Matrix inverse transform": Xhat, "First Principal Component": v.T}
plot_2db(dict_)