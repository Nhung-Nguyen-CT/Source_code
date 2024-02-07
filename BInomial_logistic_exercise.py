import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import  sklearn.metrics as metrics

file_path= r"./google_advance/activity.csv"
activity = pd.read_csv(file_path)
activity.head(10)
activity.info()
activity.rename(columns = {'Acc (vertical)':'Acc(vertical)'}, inplace = True)
activity.describe(include = 'all')

#Construct binomial logistic regression model
x_input = activity[['Acc(vertical)']]
y_input = activity[['LyingDown']]
#split dataset into training and holdout datasets
X_train, X_test, y_train, y_test = train_test_split(x_input, y_input, test_size= 0.3, random_state = 42)
clf = LogisticRegression().fit(X_train, y_train)
clf.coef_
clf.intercept_
#plot logistic regression
sns.regplot(x="Acc(vertical)", y="LyingDown", data=activity, logistic=True)
plt.show()

#Confusion matrix
#predicted labels
y_pred = clf.predict(X_test)
#print out the predicted probabilities
clf.predict_proba(X_test)[::,-1]