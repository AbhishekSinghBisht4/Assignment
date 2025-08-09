    
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

#load data ( Housing )
neww=pd.read_csv("Housing.csv")

print(neww) 

new = pd.get_dummies(neww, drop_first=True)

X = new.drop('bedrooms', axis=1)  # Features
y = new['bedrooms']   

scaler =StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test=train_test_split(X_scaled, y,test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Mean",mse)

plt.figure(figsize=(8, 5))
sns.regplot(x='area', y='bedrooms', data=neww, line_kws={"color": "red"})
plt.title("Regression Line: Area vs Bedrooms")
plt.xlabel("Area (sqft)")
plt.ylabel("Number of Bedrooms")
plt.show()