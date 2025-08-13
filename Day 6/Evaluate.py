import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib as plt

a=pd.read_csv("Iris.csv")
#print(a.head())

X = a.drop("Species", axis=1) 
y = a["Species"]         

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

# View normalized data
#print(X_normalized.head())
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_normalized, y)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#print("KNN Accuracy:", accuracy)
for k in range(1, 11):  # Try k = 1 to 10
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
   #print(f"k = {k} -> Accuracy = {acc:.3f}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap='Blues')