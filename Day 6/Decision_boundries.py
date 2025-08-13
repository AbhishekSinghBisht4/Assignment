import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np 

a = pd.read_csv("Iris.csv")
X = a.drop("Species", axis=1) 
y = a["Species"]         

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#print("KNN Accuracy:", accuracy)

feature_x = 'PetalLengthCm'
feature_y = 'PetalWidthCm'

X_vis = a[[feature_x, feature_y]]
y_vis = y

scaler_vis = StandardScaler()
X_vis_scaled = scaler_vis.fit_transform(X_vis)

knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_vis_scaled, y_vis)

h = .02
x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z_numeric = pd.factorize(Z)[0]
Z_numeric = Z_numeric.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z_numeric, alpha=0.3)
plt.scatter(X_vis_scaled[:, 0], X_vis_scaled[:, 1], c=pd.factorize(y_vis)[0], 
            edgecolors='k', cmap='viridis')
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("KNN Decision Boundary (k=5)")
plt.show()
