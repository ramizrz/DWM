import numpy as num
import pandas as pd
import matplotlib as mtlb
import matplotlib.pyplot as plt

#splitting the detect into the training and test sets
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,:1].values
#used selection in place of cross_validation since the latter is description
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size =1/3,random_state =0)
#fitting sim[le linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)
#predicting the set results
y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train, y_train,color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience Training sets')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results
plt.scatter(X_test, y_test,color='red')
plt.plot(X_test,regressor.predict(X_test), color='blue')
plt.title('Salary Vs Experience Training sets')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()