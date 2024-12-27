import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from itertools import chain
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import  math
import category_encoders as ce
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

sns.set_context('notebook')
sns.set_style('white')


# This function will allow us to easily plot data taking in x values, y values, and a title
def plot_univariate_mixture(means, stds, weights, N=10000, seed=10):
    """
    returns the simulated 1d dataset X, a figure, and the figure's ax

    """
    np.random.seed(seed)
    if not len(means) == len(stds) == len(weights):
        raise Exception("Length of mean, std, and weights don't match.")
    K = len(means)

    mixture_idx = np.random.choice(K, size=N, replace=True, p=weights)
    # generate N possible values of the mixture
    X = np.fromiter((ss.norm.rvs(loc=means[i], scale=stds[i]) for i in mixture_idx), dtype=np.float64)

    # generate values on the x axis of the plot
    xs = np.linspace(X.min(), X.max(), 300)
    ps = np.zeros_like(xs)

    for mu, s, w in zip(means, stds, weights):
        ps += ss.norm.pdf(xs, loc=mu, scale=s) * w

    fig, ax = plt.subplots()
    ax.plot(xs, ps, label='pdf of the Gaussian mixture')
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("P", fontsize=15)
    ax.set_title("Univariate Gaussian mixture", fontsize=15)
    # plt.show()

    return X.reshape(-1, 1), fig, ax


def plot_bivariate_mixture(means, covs, weights, N=10000, seed=10):
    """
    returns the simulated 2d dataset X and a scatter plot is shown

    """
    np.random.seed(seed)
    if not len(means) == len(covs) == len(weights):
        raise Exception("Length of mean, std, and weights don't match.")
    K = len(means)
    M = len(means[0])

    mixture_idx = np.random.choice(K, size=N, replace=True, p=weights)

    # generate N possible values of the mixture
    X = np.fromiter(chain.from_iterable(multivariate_normal.rvs(mean=means[i], cov=covs[i]) for i in mixture_idx),
                    dtype=float)
    X.shape = N, M

    xs1 = X[:, 0]
    xs2 = X[:, 1]

    plt.scatter(xs1, xs2, label="data")

    L = len(means)
    for l, pair in enumerate(means):
        plt.scatter(pair[0], pair[1], color='red')
        if l == L - 1:
            break
    plt.scatter(pair[0], pair[1], color='red', label="mean")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Scatter plot of the bivariate Gaussian mixture")
    plt.legend()
    plt.show()

    return X


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """
    Draw an ellipse with a given position and covariance

    """
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)

    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

X1, fig1, ax1 = plot_univariate_mixture(means=[2,5,8], stds=[0.2, 0.5, 0.8], weights=[0.3, 0.3, 0.4])

X1_sorted = np.sort(X1.reshape(-1)).reshape(-1,1)

# fit the GMM
GMM = GaussianMixture(n_components=3, random_state=10)
GMM.fit(X1_sorted)

# store the predicted probabilities in prob_X1
prob_X1 = GMM.predict_proba(X1_sorted)

# start plotting!
ax1.plot(X1_sorted, prob_X1[:,0], label='Predicted Prob of x belonging to cluster 1')
ax1.plot(X1_sorted, prob_X1[:,1], label='Predicted Prob of x belonging to cluster 2')
ax1.plot(X1_sorted, prob_X1[:,2], label='Predicted Prob of x belonging to cluster 3')
ax1.scatter(2, 0.6, color='black')
ax1.scatter(2, 1.0, color='black')
ax1.plot([2, 2], [0.6, 1.0],'--', color='black')
ax1.legend()
plt.show()
fig1


#Applying GMM on a 2d dataset
mean = [(1,5), (2,1), (6,2)]
cov1 = np.array([[1.9, 1.0], [1.0, 0.8]])
cov2 = np.array([[0.8, 0.4], [0.4, 1.2]])
cov3 = np.array([[1.9, 1.3], [1.3, 1.2]])
cov = [cov1, cov2, cov3]
weights = [0.3, 0.3, 0.4]

def is_symmetric(matrix):
    return np.array_equal(matrix, matrix.T)
def is_positive_semidefinite(matrix):
    eigenvalues, _ = np.linalg.eig(matrix)
    return np.all(eigenvalues >= 0)

# Check each covariance matrix
for i, cov_matrix in enumerate(cov):
     print(f"Covariance Matrix {i+1}:")
     print(cov_matrix)
     print("Symmetric:", is_symmetric(cov_matrix))
     print("Positive Semidefinite:", is_positive_semidefinite(cov_matrix))
     print()

X4 = plot_bivariate_mixture(means=mean, covs=cov, weights=weights, N=1000)
print("The dataset we generated has a shape of", X4.shape)

#fit GMM into the dataset just created
# since we generated a mixture dataset X4 with 3 Gaussians, it makes sense to set n_components = 3.
gm = GaussianMixture(n_components=3, random_state=0).fit(X4)
print("Means of the 3 Gaussians fitted by GMM are\n")
print(gm.means_)
print("Covariances of the 3 Gaussians fitted by GMM are")
gm.covariances_


#The default Covariance_type in sklearn.mixture.GaussianMixture is full.
plot_gmm(GaussianMixture(n_components=3, random_state=0), # the model,
          X4)
# try Covariance_type = 'tied'
plot_gmm(GaussianMixture(n_components=3, covariance_type='tied',random_state=0), # the model,
         X4)
# try Covariance_type = 'diag'
plot_gmm(GaussianMixture(n_components=3, covariance_type='diag',random_state=0), # the model,
         X4)

#excercise with customer data
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%201/customers.csv")
data.head()
SS = StandardScaler()
X = SS.fit(data).transform(data)
pca2= PCA(n_components=2)
reduced_2_PCA = pca2.fit_transform(X)
#fit reduced data into GMM
model = GaussianMixture(n_components=4, random_state=0) #how to know that 4 cluster?
model.fit(reduced_2_PCA)
PCA_2_pred = model.predict(reduced_2_PCA)
#plot cluster  of reduced_2_PCA
x = reduced_2_PCA[:,0]
y = reduced_2_PCA[:,1]
plt.scatter( x,y, c = PCA_2_pred)
plt.title("2d visualization of the clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

#PCA for 3 component and plotting
X = SS.fit(data).transform(data)
pca3= PCA(n_components=3)
reduced_3_PCA = pca3.fit_transform(X)
reduced_3_PCA = pd.DataFrame(reduced_3_PCA, columns=(['PCA 1', 'PCA 2', 'PCA 3']))
#fit reduced data into GMM
model = GaussianMixture(n_components=4, random_state=0) #how to know that 4 cluster?
model.fit(reduced_3_PCA)
PCA_3_pred = model.predict(reduced_3_PCA)
#plot cluster  of reduced_2_PCA
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(reduced_3_PCA['PCA 1'],reduced_3_PCA['PCA 2'],  reduced_3_PCA['PCA 3'],c = PCA_3_pred)
ax.set_title("3D projection of the clusters")


#apply for data credit scoring
df = pd.read_csv('bank_credit_scoring.csv', delimiter= ',')
df.info()
df.head(10)
df.describe(include='all')
#I. CLEAN DATA

print(df.duplicated().any())
print(df.isnull().any())


#Find and drop the high corellation variables:
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, mark = 0.5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[au_corr > mark]

numeric_columns = [x for x in df.columns if df[x].dtypes != 'O']
df1 = df[numeric_columns].copy()
print("Top Absolute Correlations")
print(get_top_abs_correlations(df = df1, mark = 0.6))

df = df.drop(columns=['ID','Customer_ID', 'Name', 'SSN','Monthly_Inhand_Salary', 'Amount_invested_monthly', 'Monthly_Balance', 'Num_of_Loan', 'Interest_Rate', 'Num_Credit_Inquiries' ])

#Check more details about dataset
len_df = len(df)
high_nunique_object_columns = [x for x in df.columns if ((df[x].dtypes == 'O') & (df[x].nunique() > 0.1 * len_df))]
low_nunique_object_columns = [x for x in df.columns if ((df[x].dtypes == 'O') & (df[x].nunique() <= 0.1 * len_df))]
high_nunique_numeric_columns = [x for x in df.columns if ((df[x].dtypes != 'O') & (df[x].nunique() > 0.1 * len_df))]
low_nunique_numeric_columns = [x for x in df.columns if ((df[x].dtypes != 'O') & (df[x].nunique() <= 0.1 * len_df))]


print('high nunique and dtype = object', high_nunique_object_columns)
for x in high_nunique_object_columns:
    print('columns ',x,': nunique ',df[x].nunique())

print('low nunique and dtype = object', low_nunique_object_columns)
for x in low_nunique_object_columns:
    print('columns ', x, ': nunique ', df[x].nunique())

print('high nunique and dtype = numeric', high_nunique_numeric_columns)
for x in high_nunique_numeric_columns:
    print('columns ',x,': nunique ',df[x].nunique())

print('low nunique and dtype = numeric', low_nunique_numeric_columns)
for x in low_nunique_numeric_columns:
    print('columns ', x, ': nunique ', df[x].nunique())

numeric_cols_need_encode = ['Month']
object_cols_need_spare = ['Type_of_Loan']
object_cols_need_encode = [x for x in low_nunique_object_columns if ((x != 'Type_of_Loan') & (x != 'Credit_Score'))] # exclude Credit Score and Type of Loan
cols_need_encode = numeric_cols_need_encode + object_cols_need_encode
scale_cols = [x for x in df.columns if  (df[x].dtypes != 'O' ) & (x != 'Month') & (x != 'Unnamed: 0')]


type_of_Loan_value_counts = (df['Type_of_Loan'].value_counts())
type_of_Loan_value_counts = type_of_Loan_value_counts.to_frame()
print(type_of_Loan_value_counts.head(100))

#drop rows with Type_of_Loan == 'No Data'

indexes = df.query('Type_of_Loan == "No Data" ').index.to_list()
df = df.drop(index = indexes)
df['Type_of_Loan'] = df['Type_of_Loan'].str.replace('and ','')
df['Type_of_Loan'] = df['Type_of_Loan'].str.split(', ')

df = pd.concat(
    [
        df.explode('Type_of_Loan')
        .pivot_table(index="Unnamed: 0", columns='Type_of_Loan', aggfunc='size', fill_value=0)
        .add_prefix('Type_of_Loan_'),
        df.set_index("Unnamed: 0"),
    ],
    axis=1,
)

#drop Type_of_Loan, Month, 'Num_Bank_Counts column and replace value >1 by 1 (encode)
df = df.drop(columns=['Type_of_Loan'])
df.iloc[:,:9] = np.where(df.iloc[:,:9] >= 1, 1, 0)
print(df.describe().T)

#plot data to more exploration
sns.histplot(data = df, x = 'Credit_Score', hue = 'Credit_Score')
plt.show()
# => imbalance 'Credit_score' values
#plot numerical features with Credit_Score
numeric_columns = [x for x in df.columns if df[x].dtypes != 'O']
df1 = df[numeric_columns + ['Credit_Score']].copy()
# Iterate through features and create KDE plots
for i in range ( math.ceil(len(df1.columns)/4)):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('KDE Plots for Numerical Features by Credit_Score')
    sns.kdeplot(data=df1, x=numeric_columns[i*4+0], hue='Credit_Score', ax=axes[0, 0])
    if i < 5:
        sns.kdeplot(data=df1, x=numeric_columns[i*4 + 1], hue='Credit_Score', ax=axes[0, 1])
        sns.kdeplot(data=df1, x=numeric_columns[i*4 + 2], hue='Credit_Score', ax=axes[1, 0])
        sns.kdeplot(data=df1, x=numeric_columns[i*4 + 3], hue='Credit_Score', ax=axes[1, 1])
    plt.show()

drop_cols = ['Month'] # drop because no relationship on y
#plot category features with Credit_Score
cate_columns = [x for x in df.columns if df[x].dtypes == 'O']
df1 = df[cate_columns].copy()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('KDE Plots for Category Features by Credit_Score')
for i in range(2):
    axes[0, i].tick_params(axis='x', labelrotation=45)
    axes[1, i].tick_params(axis='x', labelrotation=45)
    sns.histplot(data=df1, x=cate_columns[i*2], hue='Credit_Score', ax=axes[0, i], hue_order=['Good', 'Standard', 'Poor'], kde= True)
    sns.histplot(data=df1, x=cate_columns[i*2+1], hue='Credit_Score', ax=axes[1, i], hue_order=['Good', 'Standard', 'Poor'], kde= True)
plt.show()

#Non - linear relationships between Credit_Score and X features
df = df.drop(columns = drop_cols)

#II.PREPROCESSING
#frac = 0.4
#df = df.sample(frac=frac).copy()

X = df.iloc[:,:-1].copy()
y = df.iloc[:,-1].copy()
#X['Month'] = X['Month'].astype('str')

strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in strat_shuff_split.split(X, y):
    X_train = X.iloc[train_idx,:].copy()
    y_train = y.iloc[train_idx].copy()
    X_test = X.iloc[test_idx, :].copy()
    y_test = y.iloc[test_idx].copy()

strat_shuff_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in strat_shuff_split.split(X_train, y_train):
    X_train = X_train.iloc[train_idx,:].copy()
    y_train = y_train.iloc[train_idx].copy()
    X_val = X.iloc[val_idx, :].copy()
    y_val = y.iloc[val_idx].copy()

scale_skew_cols = pd.DataFrame(X_train[scale_cols].skew(), columns = ['skew']).sort_values(by = 'skew',ascending=False).query('abs(skew) >= 1').index
X_train[scale_skew_cols] = X_train[scale_skew_cols].apply(np.log1p)
X_test[scale_skew_cols] = X_test[scale_skew_cols].apply(np.log1p)
X_val[scale_skew_cols] = X_val[scale_skew_cols].apply(np.log1p)

scalar = StandardScaler()
X_train[scale_cols] = scalar.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scalar.transform(X_test[scale_cols])
X_val[scale_cols] = scalar.transform(X_val[scale_cols])

#encode
one_hot = ce.OneHotEncoder(use_cat_names = True)
X_train= one_hot.fit_transform(X_train)
X_test = one_hot.transform(X_test)
X_val = one_hot.transform(X_val)

la_encode = LabelEncoder()
y_train = la_encode.fit_transform(y_train)
y_test = la_encode.transform(y_test)
y_val = la_encode.transform(y_val)

print(X_train.head())
print(X_train.info())

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='weighted')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

model = GaussianMixture(n_components=3, random_state=0) #how to know that 4 cluster?
model.fit(X_train)
print('Evaluating  train', evaluate_metrics(yt=y_train, yp=model.predict(X_train)))
print('Evaluating  train', evaluate_metrics(yt=y_val, yp=model.predict(X_val)))
print('Evaluating  train', evaluate_metrics(yt=y_test, yp=model.predict(X_test)))




def sum_array(*args, ignore_negative = True):
    return sum(args)


def sum_array_gau(, *args, **kwargs):
    kwargs.pop('param_gau')
    sum_array(args, **kwargs)





sum_array(1, 2, 3, 4, 5, 6,ignore_negative=False, param_gau=1)

