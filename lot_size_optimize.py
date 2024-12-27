import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score, recall_score


with open('tom_method_data.pkl', 'rb') as file:
    long_events = pickle.load(file)
    short_events = pickle.load(file)
    data = pickle.load(file)

total_events = pd.concat([long_events, short_events])

long_events = long_events.sort_values(by = ['Close_Date']).copy()
short_events = short_events.sort_values(by = ['Close_Date']).copy()
total_events = total_events.sort_values(by = ['Close_Date']).copy()


long_events['acc_profit'] = np.cumsum(long_events['Profit'])
short_events['acc_profit'] = np.cumsum(short_events['Profit'])
total_events['acc_profit'] = np.cumsum(total_events['Profit'])

sns.lineplot(x= 'Close_Date', y= 'acc_profit', data= long_events)
plt.show()
sns.lineplot(x= 'Close_Date', y= 'acc_profit', data= short_events)
plt.show()
sns.lineplot(x= 'Date', y= 'Close', data= data)
plt.show()

