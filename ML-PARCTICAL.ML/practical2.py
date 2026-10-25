
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# PART A: SIMPLE LINEAR REGRESSION
# ============================================================

print("=" * 55)
print("SIMPLE LINEAR REGRESSION")
print("=" * 55)

# 1. Generate sample data
rng = np.random.default_rng(42)

X_simple = np.linspace(1, 20, 100).reshape(-1, 1)
y_simple = (
    4 + 3 * X_simple[:, 0]
    + rng.normal(0, 4, 100)
)

# 2. Split into training and testing sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple,
    y_simple,
    test_size=0.2,
    random_state=42
)

# 3. NumPy implementation using least squares
A_train = np.column_stack([
    np.ones(len(X_train_s)),
    X_train_s[:, 0]
])

coefficients = np.linalg.lstsq(
    A_train, y_train_s, rcond=None
)[0]

b0_np, b1_np = coefficients

y_pred_np = b0_np + b1_np * X_test_s[:, 0]

# 4. Scikit-learn implementation
simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)

y_pred_sk = simple_model.predict(X_test_s)

# 5. Evaluation
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return rmse, r2


print("\nNumPy coefficients:")
print(f"Intercept: {b0_np:.4f}")
print(f"Slope:     {b1_np:.4f}")

evaluate_model("NumPy Simple Regression", y_test_s, y_pred_np)

print("\nScikit-learn coefficients:")
print(f"Intercept: {simple_model.intercept_:.4f}")
print(f"Slope:     {simple_model.coef_[0]:.4f}")

evaluate_model(
    "Scikit-learn Simple Regression",
    y_test_s,
    y_pred_sk
)

# 6. Plot regression line
x_line = np.linspace(
    X_simple.min(), X_simple.max(), 200
).reshape(-1, 1)

y_line = simple_model.predict(x_line)

plt.figure(figsize=(8, 5))
plt.scatter(
    X_simple[:, 0], y_simple,
    color="steelblue", label="Actual data"
)
plt.plot(
    x_line[:, 0], y_line,
    color="red", linewidth=2, label="Regression line"
)
plt.xlabel("Independent variable (X)")
plt.ylabel("Dependent variable (Y)")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Residual plot
residuals_s = y_test_s - y_pred_sk

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_sk, residuals_s, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Simple Linear Regression")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# PART B: MULTIPLE LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 55)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 55)

# 1. Generate three independent variables
rng = np.random.default_rng(123)

X_multiple = rng.uniform(1, 10, size=(150, 3))

# True relationship: y = 5 + 2*x1 - 1.5*x2 + 0.7*x3 + noise
y_multiple = (
    5
    + 2 * X_multiple[:, 0]
    - 1.5 * X_multiple[:, 1]
    + 0.7 * X_multiple[:, 2]
    + rng.normal(0, 2, 150)
)

# 2. Split the dataset
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multiple,
    y_multiple,
    test_size=0.2,
    random_state=42
)

# 3. NumPy implementation
# Add a column of ones to estimate the intercept
A_train_m = np.column_stack([
    np.ones(len(X_train_m)),
    X_train_m
])

beta = np.linalg.lstsq(
    A_train_m, y_train_m, rcond=None
)[0]

y_pred_np_m = (
    beta[0] + X_test_m @ beta[1:]
)

print("\nNumPy coefficients:")
print(f"Intercept: {beta[0]:.4f}")
print("Feature coefficients:", beta[1:])

evaluate_model(
    "NumPy Multiple Regression",
    y_test_m,
    y_pred_np_m
)

# 4. Scikit-learn implementation
multiple_model = LinearRegression()
multiple_model.fit(X_train_m, y_train_m)

y_pred_sk_m = multiple_model.predict(X_test_m)

print("\nScikit-learn coefficients:")
print(f"Intercept: {multiple_model.intercept_:.4f}")
print("Feature coefficients:", multiple_model.coef_)

evaluate_model(
    "Scikit-learn Multiple Regression",
    y_test_m,
    y_pred_sk_m
)

# 5. Plot actual versus predicted values
plt.figure(figsize=(8, 5))
plt.scatter(
    y_test_m, y_pred_sk_m,
    color="teal", edgecolor="black", alpha=0.8
)

lower = min(y_test_m.min(), y_pred_sk_m.min())
upper = max(y_test_m.max(), y_pred_sk_m.max())

plt.plot(
    [lower, upper], [lower, upper],
    color="red", linestyle="--", label="Perfect prediction"
)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Multiple Regression: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Residual plot
residuals_m = y_test_m - y_pred_sk_m

plt.figure(figsize=(8, 5))
plt.scatter(
    y_pred_sk_m, residuals_m,
    color="darkorange", edgecolor="black"
)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Plot - Multiple Linear Regression")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 7. Compare NumPy and Scikit-learn predictions
print("\nImplementation comparison:")
print(
    "Maximum simple prediction difference:",
    f"{np.max(np.abs(y_pred_np - y_pred_sk)):.10f}"
)
print(
    "Maximum multiple prediction difference:",
    f"{np.max(np.abs(y_pred_np_m - y_pred_sk_m)):.10f}"
)

print("\nExperiment completed successfully.")
