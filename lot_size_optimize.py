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
from extreme_points_detection import detect_extreme_points


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

def define_biggest_trend(in_arr: np.ndarray):
    np_arr = np.asarray(in_arr)
    max_index = np.argmax(np_arr)
    min_index = np.argmin(np_arr)
    if max_index > min_index:
        if 0.5 *(np.max(np_arr) - np.min(np_arr)) > (np.max(np_arr) - np.min(np_arr[max_index:])):
            trend = "up"
        else:
            trend = "down"
            max_index = np.argmax(np_arr)
            min_index = np.argmin(np_arr[max_index:]) + max_index
    else:
        trend = "down"
    return trend, max_index, min_index


def switch_status_trading(in_arr, trend, max_index, min_index, previous_trading_status, current_index):
    in_arr = np.asarray(in_arr).copy()
    next_status = previous_trading_status
    if trend == "down" :
        distance = np.abs(in_arr[min_index] - in_arr[max_index])
        current_top_index = np.argmax(in_arr[min_index:]) + min_index
        if (previous_trading_status == False) and ((in_arr[current_index] - in_arr[min_index]) >= (0.15 * distance)) and ((in_arr[current_top_index] - in_arr[current_index]) <= (0.15 * (in_arr[current_top_index] - in_arr[min_index]))):
            next_status = True
        else:
            if (previous_trading_status == True) and (((in_arr[current_index] - in_arr[min_index]) < ( 0.15 * distance )) or ((in_arr[current_top_index] - in_arr[current_index]) > ( 0.15 * (in_arr[current_top_index] - in_arr[min_index] )))):
                next_status = False

    if trend == "up":
        distance = np.abs(in_arr[min_index] - in_arr[max_index])
        bottom_after_max_index = np.argmin(in_arr[max_index:]) + max_index
        close = in_arr[min_index:max_index+1]
        extreme_points_index, top_ = detect_extreme_points(closes = close, num_ticks = len(close))
        extreme_points_index = extreme_points_index + min_index
        bottom_before_max_index = extreme_points_index[-2]
        if (previous_trading_status == False) and ((in_arr[current_index] - in_arr[bottom_after_max_index]) >= ( 0.15 * (in_arr[max_index] - in_arr[bottom_after_max_index]) )):
                next_status = True
        else:
            if (previous_trading_status == True) and (((in_arr[max_index] - in_arr[current_index]) >= (0.15 * (in_arr[max_index] - in_arr[min_index])) or ((in_arr[max_index] - in_arr[current_index]) >= (0.15 * (in_arr[max_index] - in_arr[bottom_before_max_index]))))):
                next_status = False

    return next_status

status = []
for i in range(1200):
    if i < 30:
        status.append(True)
    else:
        arr = short_events['acc_profit'].iloc[:i]
        trend, max_index, min_index = define_biggest_trend(arr)
        next_status = switch_status_trading(in_arr= arr,trend = trend, max_index= max_index,min_index= min_index,  previous_trading_status= status[i-1], current_index= i-1)
        status.append(next_status)

arr = short_events.iloc[:1200].copy()
arr['trading_status'] = status
sns.scatterplot(
    x='Close_Date',
    y='acc_profit',
    data=arr.iloc[:1200],
    hue='trading_status',
    palette={True: 'green', False: 'red'},
    legend=False  # Hide legend if not needed###
    )

plt.xlabel('Close Date')
plt.ylabel('Accumulated Profit')
plt.title('Accumulated Profit with Trading Status Indicators')
plt.show()


