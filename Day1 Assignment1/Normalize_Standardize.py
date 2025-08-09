import pandas as pd
from sklearn.preprocessing import StandardScaler
a=pd.read_csv("C:/Users/Abhishek/Desktop/Elevate labs/Assignment1/Categorical to numerical.csv")
numericals=['Age','Fare','SibSp','Parch']
scaler=StandardScaler()
a[numericals]=scaler.fit_transform(a[numericals])
print(a[numericals].head())
a.to_csv("Standard.csv")