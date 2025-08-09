import pandas as pd
import matplotlib.pyplot as plt
new=pd.read_csv(r"C:\Users\Abhishek\Desktop\Elevate labs\Day1 Assignment1\Categorical to numerical.csv") #file name not working
# print(new.describe(include='all'))
numeric_columns=new.select_dtypes(include=['int64','float64']).columns
for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    new[col].hist(bins=30, color='skyblue', edgecolor='green')
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    new[col].hist(bins=30, color='skyblue', edgecolor="green")
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()
    # ploting data into histogram and boxplots 