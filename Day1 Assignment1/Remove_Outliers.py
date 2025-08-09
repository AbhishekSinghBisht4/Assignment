import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Assignment1/Standard.csv')

numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col], vert=False)
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.grid(True)
    plt.show()

def remove_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return dataframe[(dataframe[column] >= lower) & (dataframe[column] <= upper)]

original_shape = df.shape[0]

for col in numerical_cols:
    df = remove_outliers_iqr(df, col)

print(f"\Final shape: {df.shape}")
print(f"Rows removed: {original_shape - df.shape[0]}")

df.to_csv('titanic_no_outliers.csv')
