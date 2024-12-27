import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.preprocessing import MinMaxScaler


### BEGIN SOLUTION

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/churndata_processed.csv")
df.info()
round(df.describe(include = 'all'),2)
df_nunique = pd.DataFrame(df.nunique(), columns = ['nunique_value'])
bin_variables = list(df_nunique[df_nunique['nunique_value'] == 2].index)
print(bin_variables)
for col in bin_variables:
    print(df[col].value_counts())
categorical_variables = list(df_nunique[(df_nunique['nunique_value'] > 2) & (df_nunique['nunique_value'] <= 6)].index)
print(categorical_variables)
for col in categorical_variables:
    print(df[col].value_counts())
ordinal_variables = ['contract', 'satisfaction','months']
numeric_variables = list(set(df.columns) - set(bin_variables) - set(categorical_variables) - set(ordinal_variables))
df[numeric_variables].hist(figsize = (12,6))

#encode, scale before fit to K-nearest: required scaled data
Lab_bin, Lab_encode, Ord_encode, One_encode = LabelBinarizer(),LabelEncoder(), OrdinalEncoder(), OneHotEncoder()
df[ordinal_variables] = Ord_encode.fit_transform(df[ordinal_variables])
print(df[ordinal_variables])
df[ordinal_variables].astype('category').describe()
df[bin_variables] = Lab_bin.fit_transform(df[bin_variables])
print(df[bin_variables])
categorical_variables = list(set(categorical_variables) - set(ordinal_variables))
print(categorical_variables)
df1 = df[categorical_variables]
df1 = one.fit_transform(df1)
print(df1)
#categorical_variables is empty list => no need to concat df1 into df. If not, have to concate df1 into df
df.describe(include='all').T

mm = MinMaxScaler()
for column in [ordinal_variables + numeric_variables]:
    df[column] = mm.fit_transform(df[column])
round(df.describe().T, 3)

X,y = df.drop(columns = 'churn_value'), df['churn_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

#modeling KNN k= 3
knn = KNeighborsClassifier(n_neighbors=3)
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print( y_test[:5], y_pred[:5])
#metrics
print(classification_report(y_test, y_pred))
print('Accuracy score: ',round(accuracy_score(y_test, y_pred), 2))
print('F1 Score: ', round(f1_score(y_test, y_pred),2))

#plot confusion matrix
sns.set_palette(sns.color_palette())
_, ax = plt.subplots(figsize = (12,12))
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot = True, fmt = 'd', annot_kws = {'size': 40, 'weight': 'bold'})
labels = ['False', 'True']
ax.set_xticklabels(labels, fontsize = 25)
ax.set_yticklabels(labels[::-1], fontsize = 25)
ax.set_ylabel('Prediction', fontsize = 30)
ax.set_xlabel('Ground Truth', fontsize = 30)

#modeling KNN k= 5
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn = knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy score: ', round(accuracy_score(y_test, y_pred), 2))
print('F1 Score: ', round(f1_score(y_test, y_pred), 2))

# Plot confusion matrix
_, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', annot_kws={"size": 40, "weight": "bold"})
labels = ['False', 'True']
ax.set_xticklabels(labels, fontsize=25);
ax.set_yticklabels(labels[::-1], fontsize=25);
ax.set_ylabel('Prediction', fontsize=30);
ax.set_xlabel('Ground Truth', fontsize=30)

# tuning k
max_k = 40
f1_scores = list()
error_rates = list() # 1- accuracy

for k in range(1, max_k):
    knn = KNeighborsClassifier(n_neighbors= k, weights= 'distance')
    knn = knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    f1 = f1_score(y_pred, y_test)
    f1_scores.append((k, round(f1_score(y_test, y_pred), 4)))
    error = 1 - round(accuracy_score(y_test, y_pred), 4)
    error_rates.append((k, error))

f1_results = pd.DataFrame(f1_scores, columns=['K', 'F1 Score'])
error_results = pd.DataFrame(error_rates, columns=['K', 'Error Rate'])

# Plot F1 results
sns.set_context('talk')
sns.set_style('ticks')
plt.figure(dpi=300)
ax = f1_results.set_index('K').plot(figsize=(12, 12), linewidth=6)
ax.set(xlabel='K', ylabel='F1 Score')
ax.set_xticks(range(1, max_k, 2));
plt.title('KNN F1 Score')
plt.savefig('knn_f1.png')
#Plot error_score: KNN_elbow_curve
sns.set_context('talk')
sns.set_style('ticks')
plt.figure(dpi=300)
ax = error_results.set_index('K').plot(figsize=(12, 12), linewidth=6)
ax.set(xlabel='K', ylabel='Error score')
ax.set_xticks(range(1, max_k, 2));
plt.title('KNN Error score')
plt.savefig('knn_error_score.png')
