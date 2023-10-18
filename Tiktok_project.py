import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file_path = r"./google_advance/project_end_course/tiktok_dataset.csv"
data = pd.read_csv(file_path)
pd.set_option('display.max_columns', None)
data.head(10)
data.info()
data.describe(include = 'all')
data.isnull().sum()
data.dropna(inplace = True)
data.isnull().sum()
data.duplicated().sum()

#describe category variables:
data['claim_status'].value_counts()
data['verified_status'].value_counts()
data['author_ban_status'].value_counts()

#plot to figure out outlier
def box_plot_numeric(x_input):
        plt.figure(figsize=(6,2))
        plt.title(f'Boxplot detect outliers for {x_input} :')
        sns.boxplot(x = data[x_input])
        plt.show()

box_plot_numeric('video_duration_sec')
box_plot_numeric('video_view_count')
box_plot_numeric('video_like_count')
box_plot_numeric('video_share_count')
box_plot_numeric('video_download_count')
box_plot_numeric('video_comment_count')

#handle outlier
percentile25 = data["video_comment_count"].quantile(0.25)
percentile75 = data["video_comment_count"].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
data.loc[data["video_comment_count"] > upper_limit, "video_comment_count"] = upper_limit
