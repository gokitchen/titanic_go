# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this 
# (by clicking run or pressing Shift+Enter) will list the files in the input directory
# This kernel is on the docker container, therefore os module access the linux os in the docker.

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#read datas
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submit = pd.read_csv("../input/gender_submission.csv")
y = train["Survived"]

#check top of datas 
train.head()

train["Age"].plot()

train.describe()

train.isnull().sum()

test.isnull().sum()

train.info()

train[["Fare","Age"]].hist(figsize=(12, 4))

#Normalization minmaxscaler
from sklearn.preprocessing import MinMaxScaler 

MMScaler = MinMaxScaler()

train["Age"].median()

train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

train["Age"].plot()

#Normalization

train[["Age","Fare"]] = MMScaler.fit_transform(train[["Age","Fare"]])

test[["Age", "Fare"]] = MMScaler.fit_transform(test[["Age", "Fare"]])

train["Fare"].plot()

​
#drop name and cabin
train = train.drop(["Name", "Cabin"], axis=1)
test = test.drop(["Name", "Cabin"], axis=1)

#drop ticket
train = train.drop("Ticket", axis=1)
test = test.drop("Ticket", axis=1)

train_set = pd.get_dummies(train.iloc[:, 2:])
test_set = pd.get_dummies(test.iloc[:, 1:])

test_set.isnull().sum()

#learning module
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

#estimator
#model = GradientBoostingClassifier()
model = GradientBoostingClassifier(criterion= 'mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1,subsample=1.0, tol=0.0001, validation_fraction=0.1,
              verbose=0, warm_start=False)

#parameter setting
params = {
    "min_samples_leaf" : [2, 3, 5, 10],
}

GCV = GridSearchCV(model, param_grid=params, cv=7, scoring='accuracy', n_jobs=-1)

GCV.fit(train_set, y)

GCV.best_estimator_
GCV.best_score_

pred = GCV.predict(test_set)
submit["Survived"] = pred

#convert to csv (index=None)
submit.to_csv("submission2.csv", index=None)