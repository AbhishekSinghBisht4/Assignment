import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#seaborn for pairplot and correlation

df = pd.read_csv(r"C:\Users\Abhishek\Desktop\Elevate labs\Day1 Assignment1\Categorical to numerical.csv")

numeric_df = df.select_dtypes(include=['int64', 'float64'])

corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


#pairplot 
sns.pairplot(df, hue='Survived')
plt.show()

#Trends and visuals from plots

#