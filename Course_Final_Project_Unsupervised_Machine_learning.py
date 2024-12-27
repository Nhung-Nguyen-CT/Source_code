import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
#from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import math
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('bank_credit_scoring.csv', delimiter= ',')
df.info()

#I. CLEAN DATA
#Find the high corellation variables:

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

df = df.drop(columns=['ID', 'Customer_ID', 'Name', 'SSN'])
len_df = len(df)

#object columns with low frequency of values (high n unique)
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
scale_cols = [x for x in df.columns if  (df[x].dtypes != 'O' ) & (x != 'Month') & ( x != 'Unnamed: 0')]


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

#plot data
sns.displot(data = df, x = 'Credit_Score', kind = 'hist', hue = 'Credit_Score')
plt.show()
# => imbalance 'Credit_score' values


frac = 0.5

numeric_columns = [x for x in df.columns if df[x].dtypes != 'O']
df1 = df[numeric_columns + ['Credit_Score']].copy()


# Iterate through features and create KDE plots
for i in range (math.ceil((len(df1.columns)-1)/4)):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('KDE Plots for Numerical Features')
    sns.kdeplot(data=df1, x=numeric_columns[i*4+0], ax=axes[0, 0])
    sns.kdeplot(data=df1, x=numeric_columns[i*4 + 1], ax=axes[0, 1])
    sns.kdeplot(data=df1, x=numeric_columns[i*4 + 2], ax=axes[1, 0])
    if i < 6:
        sns.kdeplot(data=df1, x=numeric_columns[i*4 + 3], ax=axes[1, 1])
    plt.show()

drop_cols = ['Month'] # drop because no relationship on y

cate_columns = [x for x in df.columns if df[x].dtypes == 'O']
df1 = df[cate_columns].copy()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('KDE Plots for Category Features')
for i in range(2):
    axes[0, i].tick_params(axis='x', labelrotation=45)
    axes[1, i].tick_params(axis='x', labelrotation=45)
    sns.histplot(data=df1, x=cate_columns[i*2], ax=axes[0, i], hue_order=['Good', 'Standard', 'Poor'], kde= True)
    sns.histplot(data=df1, x=cate_columns[i*2+1], ax=axes[1, i], hue_order=['Good', 'Standard', 'Poor'], kde= True)
plt.show()

df = df.drop(columns = drop_cols)
df = df.drop(columns = 'Credit_Score').copy()

#encode object variables
one_hot = ce.OneHotEncoder(use_cat_names = True)
df = one_hot.fit_transform(df)
#scale
scalar = StandardScaler()
scaled_df = scalar.fit_transform(df)

#PCA dimension reduction
pca = PCA(n_components=3)
pca.fit(scaled_df)
PCA_df = pd.DataFrame(pca.transform(scaled_df), columns=(["col1","col2", "col3"]))
PCA_df.describe().T

#using Elbow method to determine most effective number of clusters:
print('Elbow Method to determine the number of clusters to be formed:')
Elbow = KElbowVisualizer(KMeans(), k = 10)
Elbow.fit(PCA_df)
Elbow.show()
print('Optimize cluster: ',Elbow.elbow_value_)
k = Elbow.elbow_value_
#clustering model
#Agglomerative Clustering model
#AC= AgglomerativeClustering(n_clusters=4)
#predict_AC = AC.fit_predict(PCA_df)
#PCA_df['Clusters'] = predict_AC
#df['Clusters'] = predict_AC

#kmeans clustering
K_cluster = KMeans(n_clusters=k)
predict_K = K_cluster.fit_predict(PCA_df)

from mpl_toolkits.mplot3d import Axes3D

def display_cluster(X, km=[], num_clusters = 0):
    color = 'brgcmyk'
    alpha = 0.5
    s = 20
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if num_clusters == 0:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color[0], alpha=alpha, s=s)
    else:
            for i in range(num_clusters):
                clus = np.where(km.labels_ == i)
                X1 = X.iloc[clus]
                ax.scatter(X1.iloc[:,0], X1.iloc[:,1], X1.iloc[:, 2],
                           c=color[i], alpha=alpha, s=s)
                ax.scatter(km.cluster_centers_[i][0], km.cluster_centers_[i][1], km.cluster_centers_[i][2],
                           c=color[i], marker='X', s=100)
                ax.set_xlabel('Col 1')
                ax.set_ylabel('Col 2')
                ax.set_zlabel('Col 3')
    plt.show()

display_cluster(PCA_df, K_cluster, num_clusters= k)



