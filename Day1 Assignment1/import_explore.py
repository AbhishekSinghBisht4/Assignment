import pandas as pd 
read=pd.read_csv("Titanic-Dataset.csv")

"""
print(read.head())  #prints the first 5 row in the cvs file 

print(read.info()) #information about the file

print(read.isnull().sum()) #count nulls

"""
print(read.isnull().sum())

if 'Age' in read.columns:
    read['Age'].fillna(read['Age'].median(), inplace=True)

if 'Embarked' in read.columns:
    read['Embarked'].fillna(read['Embarked'].mode()[0], inplace=True)

print(read.isnull().sum())

read.to_csv("Titanic_cleaned_dataset.csv") #created new file after filling null values