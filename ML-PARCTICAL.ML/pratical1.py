
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix, -
    classification_report
)

import warnings
warnings.filterwarnings("ignore")

# 1. Load and explore the dataset
iris = load_iris()
X = iris.data[:, :2]  # First two features
y = iris.target

print("Dataset Shape:", X.shape)
print("Classes:", iris.target_names)
print("Feature Names:", iris.feature_names[:2])

# 2. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the dataset (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 4. Train Logistic Regression model
model = LogisticRegression(max_iter=300, random_state=42)
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# Part A: Train-test evaluation
print("\n== Part A: Train/Test Split (70/30) ==")

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, average='weighted'):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# Part B: 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_model = LogisticRegression(max_iter=300, random_state=42)

cv_acc = cross_val_score(
    cv_model, X_scaled, y, cv=kf, scoring="accuracy"
)

cv_f1 = cross_val_score(
    cv_model, X_scaled, y, cv=kf, scoring="f1_weighted"
)

print("\n== Part B: 5-Fold Cross-Validation ==")

for i, (acc, f1) in enumerate(zip(cv_acc, cv_f1), start=1):
    print(f"Fold {i}: Accuracy = {acc:.4f}, F1-score = {f1:.4f}")

print(f"\nMean Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
print(f"Mean F1-score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")


# 6. Plot the decision boundary
def plot_boundary(model, X, y, title, fname):
    h = 0.03

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 5))

    plt.contourf(
        xx, yy, Z,
        alpha=0.35,
        cmap="RdYlBu"
    )

    scatter = plt.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap="RdYlBu",
        edgecolor="k",
        s=50
    )

    plt.colorbar(scatter, ticks=[0, 1, 2])
    plt.xlabel("Sepal length (scaled)")
    plt.ylabel("Sepal width (scaled)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.show()

    print(f"Plot saved: {fname}")


# Train a model using the full scaled dataset for visualization
full_model = LogisticRegression(max_iter=300, random_state=42)
full_model.fit(X_scaled, y)

plot_boundary(
    full_model,
    X_scaled,
    y,
    "Decision Boundary - Iris (Logistic Regression)",
    "exp1_boundary.png"
)
