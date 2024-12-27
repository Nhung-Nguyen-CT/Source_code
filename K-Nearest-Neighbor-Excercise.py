import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

rs = 123
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"
tumor_df = pd.read_csv(dataset_url)
tumor_df.head()
tumor_df.info()
tumor_df.describe(include = 'all')
print(tumor_df.nunique())
for col in tumor_df.columns:
    print(tumor_df[col].value_counts(normalize= True))

X = tumor_df.iloc[:, :-1]
y = tumor_df.iloc[:, -1:]
y.value_counts(normalize=True)

#SPLIT DATA AND TRAINING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, stratify=y, random_state= rs)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, stratify=y_train, random_state=rs)
knn_model = KNeighborsClassifier(n_neighbors= 2)
knn_model.fit(X_train, y_train)
preds = knn_model.predict(X_val)

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt,yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

evaluate_metrics(y_val, preds)

#tune k
max_k = 50
f1_scores = []

for k in range(1,max_k+1):
    knn = KNeighborsClassifier(n_neighbors= k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_val)
    f1_scores.append((k,round(f1_score(y_val, preds),4)))
f1_results = pd.DataFrame(f1_scores, columns= ['K','f1_score'])
f1_results = f1_results.set_index('K')

#plot F1 results
ax = f1_results.plot(figsize = (12, 12))
ax.set(xlabel = 'Num of neighbor', ylabel = 'F1 score')
ax.set_xticks(range(1, max_k,2))
plt.ylim((0.85,1))
plt.title('KNN F1 score')

#best_k
k_best = f1_results['f1_score'].idxmax()
knn = KNeighborsClassifier(n_neighbors= k)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print(f1_score(y_test, preds))
