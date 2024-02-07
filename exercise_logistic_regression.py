import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"./google_advance/Invistico_Airline.csv"
df_original = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
df_original.head(10)
df_original.info()
df_original.describe (include = 'all')
df_original['satisfaction'].value_counts(dropna = False)
df_original.isnull().sum()
df_subset = df_original.dropna(axis = 0).reset_index(drop = True)
df_subset.head(5)
df_subset = df_subset.astype({'Inflight entertainment':float})
df_subset['satisfaction'] = OneHotEncoder(drop = 'first').fit_transform(df_subset[['satisfaction']]).toarray()
X_input = df_subset[['Inflight entertainment']]
y_input = df_subset['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size = 0.3, random_state = 42)

#Model building
clf = LogisticRegression().fit(X_train, y_train)
clf.coef_
clf.intercept_
sns.regplot(data = df_subset, x = 'Inflight entertainment', y = 'satisfaction', logistic = True, ci = None)
plt.show()

y_predict= clf.predict(X_test)
clf.predict_proba(X_test)
clf.predict(X_test)
print('Accuracy: ', '%.6f' % metrics.accuracy_score(y_test, y_predict))
print('Precision: ', '%.6f' % metrics.precision_score(y_test, y_predict))
print('Recall: ', '%.6f' % metrics.recall_score(y_test, y_predict))
#plot confusion matrix
cm = metrics.confusion_matrix(y_test, y_predict, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
disp.plot()
plt.show()
