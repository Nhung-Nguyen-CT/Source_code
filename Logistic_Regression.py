import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

rs = 123

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items.csv"
food_df = pd.read_csv(dataset_url)
food_df.info()
food_df['class'].value_counts()

feature_cols = list(food_df.columns)
feature_cols.remove('class')
feature_cols

food_df['class'].value_counts().plot.bar(color=['yellow', 'red', 'green'])


#Feature Engineering
X_raw = food_df[feature_cols]
y_raw = food_df['class']

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)
print(f"The range of feature inputs are within {X.min()} to {X.max()}")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())
np.unique(y, return_counts = True)

#train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = rs)
print(f"Training dataset shape, X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing dataset shape, X_test: {X_test.shape}, y_test: {y_test.shape}")
penalty = 'l2'
multi_class = 'multinomial'
solver = 'lbfgs'
max_iter = 1000
l2_model = LogisticRegression(random_state = rs, penalty = penalty, multi_class = multi_class, solver = solver, max_iter = max_iter)
l2_model.fit(X_train, y_train)
l2_preds = l2_model.predict(X_test)

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt,yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp)
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

evaluate_metrics(y_test, l2_preds)

#l1 model
penalty = 'l1'
multi_class = 'multinomial'
solver = 'saga'
max_iter = 1000
l1_model = LogisticRegression(random_state = rs, penalty = penalty, multi_class = multi_class, solver = solver, max_iter = max_iter)
l1_model.fit(X_train, y_train)
l1_preds = l1_model.predict(X_test)
l1_model.predict_proba(X_test[:1,:])[0]
evaluate_metrics(y_test, l1_preds)

#confusion matrix
cf = confusion_matrix(y_test, l1_preds, normalize='true')
sns.set_context('talk')
disp = ConfusionMatrixDisplay(confusion_matrix= cf, display_labels= l1_model.classes_)
disp.plot()
plt.show()

#plot features
# Extract and sort feature coefficients
def get_feature_coefs(regression_model, label_index, columns):
    coef_dict = {}
    for coef, feat in zip(regression_model.coef_[label_index, :], columns):
        if abs(coef) >= 0.01:
            coef_dict[feat] = coef
    # Sort coefficients
    coef_dict = {k: v for k, v in sorted(coef_dict.items(), key=lambda item: item[1])}
    return coef_dict

# Generate bar colors based on if value is negative or positive
def get_bar_colors(values):
    color_vals = []
    for val in values:
        if val <= 0:
            color_vals.append('r')
        else:
            color_vals.append('g')
    return color_vals

# Visualize coefficients
def visualize_coefs(coef_dict):
    features = list(coef_dict.keys())
    values = list(coef_dict.values())
    y_pos = np.arange(len(features))
    color_vals = get_bar_colors(values)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    ax.barh(y_pos, values, align='center', color=color_vals)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    # labels read top-to-bottom
    ax.invert_yaxis()
    ax.set_xlabel('Feature Coefficients')
    ax.set_title('')
    plt.show()

# Get the coefficents for Class 1, Less Often
coef_dict = get_feature_coefs(l1_model, 1, feature_cols)
visualize_coefs(coef_dict)

coef_dict = get_feature_coefs(l1_model, 2, feature_cols)
visualize_coefs(coef_dict)

#train elasticnet LogisticRegression
penalty = 'elasticnet'
l1_ratio = 0.1
multi_class = 'multinomial'
solver = 'saga'
max_iter = 1000
elastic_model = LogisticRegression(random_state = rs, penalty = penalty, l1_ratio = l1_ratio, multi_class = multi_class, solver = solver, max_iter = max_iter)
elastic_model.fit(X_train, y_train)
elastic_preds = elastic_model.predict(X_test)
evaluate_metrics(y_test, elastic_preds)
#plot confusion matrix
cf = confusion_matrix(y_test, elastic_preds, normalize='true')
sns.set_context('talk')
disp = ConfusionMatrixDisplay(confusion_matrix= cf, display_labels= elastic_model.classes_)
disp.plot()
plt.show()
#interprete coefficients
coef_dict = get_feature_coefs(elastic_model, 1, feature_cols)
visualize_coefs(coef_dict)