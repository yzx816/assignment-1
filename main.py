import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error  # Calculate error

# Read training data
train_data = pd.read_excel("E:\Training Data.xlsx")  # Update path
X_train = train_data.iloc[:, 0].values.reshape(-1, 1)
y_train = train_data.iloc[:, 1].values

# Read test data
test_data = pd.read_excel("E:\TestData.xlsx")  # Update path
X_test = test_data.iloc[:, 0].values.reshape(-1, 1)
y_test = test_data.iloc[:, 1].values

# === 1. Least Squares Method ===
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_ls_train = lr.predict(X_train)
y_pred_ls_test = lr.predict(X_test)

# === 2. Gradient Descent Method ===
def loss_function(theta, X, y):
    return np.mean((X @ theta - y) ** 2)

X_train_bias = np.c_[np.ones_like(X_train), X_train]  # Add bias term
X_test_bias = np.c_[np.ones_like(X_test), X_test]

theta_gd = np.zeros(2)
res = minimize(loss_function, theta_gd, args=(X_train_bias, y_train), method="BFGS")
theta_gd = res.x

y_pred_gd_train = X_train_bias @ theta_gd
y_pred_gd_test = X_test_bias @ theta_gd

# === 3. Newton's Method ===
theta_newton = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
y_pred_newton_train = X_train_bias @ theta_newton
y_pred_newton_test = X_test_bias @ theta_newton

# === Calculate Errors ===
mse_ls_train = mean_squared_error(y_train, y_pred_ls_train)
mse_ls_test = mean_squared_error(y_test, y_pred_ls_test)

mse_gd_train = mean_squared_error(y_train, y_pred_gd_train)
mse_gd_test = mean_squared_error(y_test, y_pred_gd_test)

mse_newton_train = mean_squared_error(y_train, y_pred_newton_train)
mse_newton_test = mean_squared_error(y_test, y_pred_newton_test)

# Print Errors
print("=== Training Error (MSE) ===")
print(f"Least Squares: {mse_ls_train:.4f}")
print(f"Gradient Descent: {mse_gd_train:.4f}")
print(f"Newton's Method: {mse_newton_train:.4f}")

print("\n=== Test Error (MSE) ===")
print(f"Least Squares: {mse_ls_test:.4f}")
print(f"Gradient Descent: {mse_gd_test:.4f}")
print(f"Newton's Method: {mse_newton_test:.4f}")

# === Plot Training Data Fit ===
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, label="Training Data", color='gray', alpha=0.5)
plt.plot(X_train, y_pred_ls_train, label="Least Squares", color='red', linestyle="--")
plt.plot(X_train, y_pred_gd_train, label="Gradient Descent", color='blue', linestyle="-.")
plt.plot(X_train, y_pred_newton_train, label="Newton's Method", color='green', linestyle=":")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Data - Linear Fit")
plt.legend()
plt.show()

# === Plot Test Data Fit ===
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, label="Test Data", color='gray', alpha=0.5)
plt.plot(X_test, y_pred_ls_test, label="Least Squares", color='red', linestyle="--")
plt.plot(X_test, y_pred_gd_test, label="Gradient Descent", color='blue', linestyle="-.")
plt.plot(X_test, y_pred_newton_test, label="Newton's Method", color='green', linestyle=":")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Test Data - Linear Fit")
plt.legend()
plt.show()
