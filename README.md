# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file and print the number of contents to be displayed.
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:GURUPRASAD.B
RegisterNumber:212221230032
*/
```
~~~
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
~~~

## Output:
## Data.head():
![ml1](https://user-images.githubusercontent.com/95342910/175802632-54f7b63b-532a-4d33-af66-de434764dd9b.png)

## Data.info():
![ml2](https://user-images.githubusercontent.com/95342910/175802636-2450ff9d-7b17-4769-ba21-c94bfc274652.png)

## Data.isnull().sum():
![ml3](https://user-images.githubusercontent.com/95342910/175802644-38eeab10-c630-4967-a14c-a6e0bc623358.png)

## Y_Pred:
![ml4](https://user-images.githubusercontent.com/95342910/175802661-1d708533-114e-47a9-878f-5eb52d39a08d.png)

## Accuracy:
![ml5](https://user-images.githubusercontent.com/95342910/175802668-59a979fb-f0b7-4419-8cd4-a5d0b17e7526.png)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
