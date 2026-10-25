
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# PART A: DECISION TREE ON THE IRIS DATASET
# =========================================================

iris = load_iris()

# Use two features so the decision boundary can be plotted
X = iris.data[:, [2, 3]]  # Petal length and petal width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 1. Unrestricted decision tree
unpruned = DecisionTreeClassifier(random_state=42)
unpruned.fit(X_train, y_train)

# 2. Depth-controlled decision tree
depth_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
depth_tree.fit(X_train, y_train)

# 3. Pruned tree using cost-complexity pruning
pruned = DecisionTreeClassifier(
    random_state=42,
    ccp_alpha=0.02
)
pruned.fit(X_train, y_train)

# Evaluate all models
models = {
    "Unpruned tree": unpruned,
    "Depth-controlled tree": depth_tree,
    "Pruned tree": pruned
}

print("=" * 65)
print("DECISION TREE: IRIS CLASSIFICATION")
print("=" * 65)

for name, model in models.items():
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n{name}")
    print(f"Tree depth:       {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy:  {test_acc:.4f}")

print("\nClassification report for the depth-controlled tree:")
print(classification_report(
    y_test,
    depth_tree.predict(X_test),
    target_names=iris.target_names,
    zero_division=0
))

# =========================================================
# PART B: VISUALIZE THE DECISION TREE
# =========================================================

plt.figure(figsize=(16, 8))

plot_tree(
    depth_tree,
    feature_names=[
        iris.feature_names[2],
        iris.feature_names[3]
    ],
    class_names=list(iris.target_names),
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Depth-Controlled Decision Tree (max_depth=3)")
plt.tight_layout()
plt.show()

# =========================================================
# PART C: VISUALIZE DECISION BOUNDARIES
# =========================================================

def plot_boundary(model, X, y, title):
    h = 0.02

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(
        xx, yy, Z,
        alpha=0.3,
        cmap="viridis"
    )

    plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap="viridis",
        edgecolors="black",
        s=35
    )

    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_boundary(unpruned, X, y, "Unpruned Tree Decision Boundary")
plot_boundary(depth_tree, X, y, "Depth-Controlled Tree Decision Boundary")
plot_boundary(pruned, X, y, "Pruned Tree Decision Boundary")

# =========================================================
# PART D: DEMONSTRATE OVERFITTING USING SYNTHETIC DATA
# =========================================================

X_syn, y_syn = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    flip_y=0.20,
    class_sep=0.6,
    random_state=42
)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_syn,
    y_syn,
    test_size=0.30,
    random_state=42,
    stratify=y_syn
)

# Compare unrestricted, depth-limited, and pruned models
overfit_models = {
    "Unrestricted": DecisionTreeClassifier(random_state=42),
    "Depth = 3": DecisionTreeClassifier(
        max_depth=3, random_state=42
    ),
    "Pruned": DecisionTreeClassifier(
        random_state=42, ccp_alpha=0.01
    )
}

print("\n" + "=" * 65)
print("OVERFITTING DEMONSTRATION: SYNTHETIC DATA")
print("=" * 65)

names = []
train_scores = []
test_scores = []

for name, model in overfit_models.items():
    model.fit(X_train_s, y_train_s)

    train_acc = accuracy_score(
        y_train_s, model.predict(X_train_s)
    )
    test_acc = accuracy_score(
        y_test_s, model.predict(X_test_s)
    )

    names.append(name)
    train_scores.append(train_acc)
    test_scores.append(test_acc)

    print(f"\n{name}")
    print(f"Tree depth:        {model.get_depth()}")
    print(f"Number of leaves:  {model.get_n_leaves()}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy:  {test_acc:.4f}")

# Plot training vs testing accuracy
positions = np.arange(len(names))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    positions - width / 2,
    train_scores,
    width,
    label="Training accuracy",
    color="steelblue"
)

plt.bar(
    positions + width / 2,
    test_scores,
    width,
    label="Testing accuracy",
    color="orange"
)

plt.xticks(positions, names)
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.title("Decision Tree Overfitting and Complexity Control")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nExperiment completed successfully.")
