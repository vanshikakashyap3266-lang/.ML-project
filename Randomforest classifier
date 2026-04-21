from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=3,
    max_depth=3,
    random_state=0
)


rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = rf_model.score(X_test, y_test)
print("Random Forest Accuracy:", accuracy)

for i, tree in enumerate(rf_model.estimators_):
    plt.figure(figsize=(10,6))
    plot_tree(
        tree,
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True
    )
    plt.title(f"Tree {i+1} in Random Forest")
    plt.show()
