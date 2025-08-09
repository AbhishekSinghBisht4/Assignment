#using and doing futher steps with cleaned data.
import pandas as pd
data=pd.read_csv("Titanic_cleaned_dataset.csv")
print(data.head())
if 'Sex' in data.columns:
    data["Sex"]=data['Sex'].map({"male":0, 'female':1})
print(data[['Sex']].head())
data.to_csv("Categorical to numerical.csv")