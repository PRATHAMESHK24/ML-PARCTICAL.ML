
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# PART A: LOAD AND EXPLORE THE DATASET
# =========================================================

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

print("=" * 60)
print("DIMENSIONALITY REDUCTION: PCA AND t-SNE")
print("=" * 60)

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Classes:", class_names)

# Split first to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Fit scaler only on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# PART B: BASELINE CLASSIFICATION USING ALL FOUR FEATURES
# =========================================================

baseline_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

baseline_model.fit(X_train_scaled, y_train)
y_pred_base = baseline_model.predict(X_test_scaled)

baseline_accuracy = accuracy_score(y_test, y_pred_base)

print("\n--- Baseline: Original Features ---")
print("Number of features:", X_train_scaled.shape[1])
print(f"Test accuracy: {baseline_accuracy:.4f}")

# =========================================================
# PART C: PCA DIMENSIONALITY REDUCTION
# =========================================================

# Fit PCA on training data only
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\n--- PCA Results ---")
print("Original dimensions:", X_train_scaled.shape[1])
print("Reduced dimensions:", X_train_pca.shape[1])

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total variance explained:",
    f"{pca.explained_variance_ratio_.sum():.4f}"
)

# Train classifier on PCA-reduced data
pca_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

pca_model.fit(X_train_pca, y_train)
y_pred_pca = pca_model.predict(X_test_pca)

pca_accuracy = accuracy_score(y_test, y_pred_pca)

print(f"PCA test accuracy: {pca_accuracy:.4f}")

print("\nPCA Classification Report:")
print(classification_report(
    y_test,
    y_pred_pca,
    target_names=class_names,
    zero_division=0
))

# =========================================================
# PART D: PCA 2D VISUALIZATION
# =========================================================

# Transform all samples for visualization using the
# scaler and PCA fitted on training data only.
X_all_scaled = scaler.transform(X)
X_all_pca = pca.transform(X_all_scaled)

plt.figure(figsize=(8, 6))

for class_id, class_name in enumerate(class_names):
    mask = y == class_id
    plt.scatter(
        X_all_pca[mask, 0],
        X_all_pca[mask, 1],
        label=class_name,
        s=55,
        edgecolors="black",
        alpha=0.8
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Iris Dataset in 2D")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# Plot explained variance
plt.figure(figsize=(7, 4))
plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_,
    color=["steelblue", "orange"]
)
plt.ylabel("Explained variance ratio")
plt.title("Variance Explained by PCA Components")
plt.tight_layout()
plt.show()

# =========================================================
# PART E: t-SNE 2D VISUALIZATION
# =========================================================

# t-SNE is fitted on all standardized samples for an
# exploratory visualization only.
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1000,
    random_state=42
)

X_tsne = tsne.fit_transform(X_all_scaled)

plt.figure(figsize=(8, 6))

for class_id, class_name in enumerate(class_names):
    mask = y == class_id
    plt.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        label=class_name,
        s=55,
        edgecolors="black",
        alpha=0.8
    )

plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.title("t-SNE: Iris Dataset in 2D")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# =========================================================
# PART F: ACCURACY COMPARISON
# =========================================================

print("\n--- Accuracy Comparison ---")
print(f"Original features (4): {baseline_accuracy:.4f}")
print(f"PCA components (2):    {pca_accuracy:.4f}")

print("\nAccuracy change (PCA - original):")
print(f"{pca_accuracy - baseline_accuracy:+.4f}")

print("\nExperiment completed successfully.")
