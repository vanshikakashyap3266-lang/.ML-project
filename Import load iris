# Decision Tree Full Implementation with Visualization

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = model.score(X_test, y_test)
print("Decision Tree Accuracy:", accuracy)


plt.figure(figsize=(12,8))
plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True
)
plt.title("Decision Tree Visualization")
plt.show()
