import requests
import pandas as pd
from io  import BytesIO

#path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv"
#path = 'https://raw.githubusercontent.com/NTTrung9204/assignment_basic/master/Credit%20Score/train.csv'
#respone = requests.get(path)
#respone.content
#respone.text.encode()
#df = pd.read_csv(BytesIO(respone.content))
#df.head()
#df.to_csv('bank_credit_scoring.csv')




#Power transform
# imports
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# as always, a public dataset
path ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
respone = requests.get(path)
if respone.status_code == 200:
    df = pd.read_csv(BytesIO(respone.content))


# for this, we're going to use the
# open minus close to predict tomorrows close price

# lets take every 10th line, since the dataset is massive



# import our powertransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

# define it
pt = PowerTransformer(method = "box-cox", standardize=True)

## ******* Remember to split before you apply your transformer ************
X = df['RAD'].values
y = df['CRIM'].values

## stock data is time series
## we turn off shuffle here
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,\
                                                 random_state=32,
                                                 shuffle=False)

## now we apply our transformer and get the lamdbas
# from only our train set
yeo_X_train = pt.fit_transform(X_train.reshape((-1,1)))

# we can see the difference
pd.DataFrame(X_train).hist(bins=60)
pd.DataFrame(yeo_X_train).hist(bins=60)
