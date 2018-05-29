#importing libarary
import numpy as np
import pandas as pd
import matplotlib as plt

#IMPORTING DATASET
dataset=pd.read_csv('Social_Network_Ads.csv')
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#DIVIDING DATASET INTO TRAIN AND TEST SET
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#fitting logisitic regression to Train set
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

classifier.score(x_train,y_train)

#prediciting value of test set
y_pred=classifier.predict(x_test)
classifier.score(x_test,y_test)

#computing efficency of classification model using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)