
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# =========================================================
# PART A: KNN CLASSIFICATION ON IRIS DATASET
# =========================================================

iris = load_iris()

# Use two features for easy visualization
X = iris.data[:, [2, 3]]  # Petal length and petal width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Scaling is fitted only on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("=" * 60)
print("KNN CLASSIFICATION - IRIS DATASET")
print("=" * 60)

print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))
print("Selected k:", knn.n_neighbors)
print(f"Training accuracy: {knn.score(X_train_scaled, y_train):.4f}")
print(f"Testing accuracy:  {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    zero_division=0
))

# =========================================================
# PART B: VISUALIZE KNN DECISION BOUNDARY
# =========================================================

def plot_knn_boundary(model, X_scaled, y, title):
    h = 0.025

    x_min = X_scaled[:, 0].min() - 0.5
    x_max = X_scaled[:, 0].max() + 0.5
    y_min = X_scaled[:, 1].min() - 0.5
    y_max = X_scaled[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(
        xx, yy, Z,
        alpha=0.3, cmap="viridis"
    )

    plt.scatter(
        X_scaled[:, 0], X_scaled[:, 1],
        c=y, cmap="viridis",
        edgecolors="black", s=35
    )

    plt.xlabel("Standardized petal length")
    plt.ylabel("Standardized petal width")
    plt.title(title)
    plt.tight_layout()
    plt.show()


plot_knn_boundary(
    knn, X_train_scaled, y_train,
    "KNN Decision Boundary (k=5)"
)

# =========================================================
# PART C: TUNE K AND COMPARE ACCURACY
# =========================================================

print("\n" + "=" * 60)
print("TUNING THE VALUE OF K")
print("=" * 60)

# Pipeline prevents data leakage during cross-validation
cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

k_values = range(1, 22, 2)
cv_scores = []
train_scores = []
test_scores = []

for k in k_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    # Cross-validation uses training data only
    scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="accuracy"
    )
    cv_scores.append(scores.mean())

    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

    print(
        f"k={k:2d} | CV accuracy={scores.mean():.4f} "
        f"| Train={train_scores[-1]:.4f} "
        f"| Test={test_scores[-1]:.4f}"
    )

best_index = int(np.argmax(cv_scores))
best_k = list(k_values)[best_index]

print("\nBest k from cross-validation:", best_k)
print(f"Best mean CV accuracy: {cv_scores[best_index]:.4f}")
print(f"Test accuracy at selected k: {test_scores[best_index]:.4f}")

# Plot CV accuracy against k
plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values), cv_scores,
    marker="o", color="blue",
    label="Mean CV accuracy"
)
plt.plot(
    list(k_values), train_scores,
    marker="s", linestyle="--",
    label="Training accuracy"
)
plt.xlabel("Number of neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Hyperparameter Tuning")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Plot training and test accuracy
plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values), train_scores,
    marker="o", label="Training accuracy"
)
plt.plot(
    list(k_values), test_scores,
    marker="s", label="Testing accuracy"
)
plt.xlabel("Number of neighbors (k)")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy for Different k")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# =========================================================
# PART D: CURSE OF DIMENSIONALITY
# =========================================================

print("\n" + "=" * 60)
print("CURSE OF DIMENSIONALITY EXPERIMENT")
print("=" * 60)

dimensions = [2, 5, 10, 20, 50, 100]
dim_cv_accuracy = []
distance_ratios = []

for dim in dimensions:
    X_dim, y_dim = make_classification(
        n_samples=1200,
        n_features=dim,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        flip_y=0.05,
        random_state=42
    )

    Xtr, Xte, ytr, yte = train_test_split(
        X_dim, y_dim,
        test_size=0.30,
        random_state=42,
        stratify=y_dim
    )

    dim_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    dim_scores = cross_val_score(
        dim_model, Xtr, ytr,
        cv=5, scoring="accuracy"
    )
    dim_cv_accuracy.append(dim_scores.mean())

    # Examine nearest vs farthest distances on a fixed
    # sample of standardized training points
    scaler_dim = StandardScaler()
    Xtr_scaled = scaler_dim.fit_transform(Xtr)

    rng = np.random.default_rng(42)
    sample = Xtr_scaled[
        rng.choice(len(Xtr_scaled), size=100, replace=False)
    ]

    # Pairwise Euclidean distances, excluding self-distances
    distances = np.sqrt(
        np.sum(
            (sample[:, None, :] - sample[None, :, :]) ** 2,
            axis=2
        )
    )

    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)

    # Ignore the diagonal when calculating farthest distance
    distances_far = distances.copy()
    np.fill_diagonal(distances_far, -np.inf)
    farthest = distances_far.max(axis=1)

    ratio = np.mean(nearest / farthest)
    distance_ratios.append(ratio)

    print(
        f"Features={dim:3d} | "
        f"Mean CV accuracy={dim_scores.mean():.4f} "
        f"(+/- {dim_scores.std():.4f}) | "
        f"Nearest/Farthest distance ratio={ratio:.4f}"
    )

# Plot accuracy against number of dimensions
plt.figure(figsize=(8, 5))
plt.plot(
    dimensions, dim_cv_accuracy,
    marker="o", color="darkgreen"
)
plt.xlabel("Number of features")
plt.ylabel("Mean 5-fold CV accuracy")
plt.title("KNN Performance vs Dimensionality")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Plot distance concentration indicator
plt.figure(figsize=(8, 5))
plt.plot(
    dimensions, distance_ratios,
    marker="s", color="darkred"
)
plt.xlabel("Number of features")
plt.ylabel("Mean nearest / farthest distance ratio")
plt.title("Distance Concentration as Dimensionality Increases")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("\nExperiment completed successfully.")
