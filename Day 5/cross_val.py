import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score



a=pd.read_csv("heart.csv")
#print(a)

X = a.drop('target', axis=1)
y = a['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(5,5))
plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
# plt.show()

#analyze Overfitting by 
# Controlling Tree Depth

from sklearn.metrics import accuracy_score

depths = [None, 2, 4, 6, 8, 10]
#print("Analyzing overfitting with different max_depth values:")

for d in depths:
    dt_temp = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_temp.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, dt_temp.predict(X_train))
    test_acc = accuracy_score(y_test, dt_temp.predict(X_test))

    #print(f"max_depth={d}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

#print("\nRandom Forest Accuracy:")
#print(f"Train Acc = {rf_train_acc:.3f}, Test Acc = {rf_test_acc:.3f}")

# Decision Tree
dt_importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
##print("\nDecision Tree Feature Importances:")
#print(dt_importances)

# Random Forest
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
#print("\nRandom Forest Feature Importances:")
#print(rf_importances)


dt_cv = DecisionTreeClassifier(max_depth=4, random_state=42)
rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)

dt_scores = cross_val_score(dt_cv, X, y, cv=5)
rf_scores = cross_val_score(rf_cv, X, y, cv=5)

print("\nCross-Validation Results:")
print(f"Decision Tree: {dt_scores.mean():.3f} ± {dt_scores.std():.3f}")
print(f"Random Forest: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
