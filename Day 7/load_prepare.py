from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

data = pd.read_csv("breast-cancer.scv")

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


