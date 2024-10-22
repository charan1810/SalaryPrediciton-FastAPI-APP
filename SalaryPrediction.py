import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv(r"C:\Users\CHARAN\Datascience-programs\MachineLeaarningPrograms\LinearRegression\SalaryPrediction\Salary_dataset.csv")


dataset.drop('Unnamed: 0',axis=1,inplace=True)

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

x_test,x_train,y_test,y_train=train_test_split(x,y,train_size=0.80,test_size=0.20,random_state=0)


regressor=LinearRegression()
regressor.fit(x_train,y_train)


y_pred=regressor.predict(x_test)


comparison=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)

# visualization #

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary')



