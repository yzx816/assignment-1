import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
# Read training data
train_data = pd.read_excel("E:\Training Data.xlsx")  # Update path
X_train = train_data.iloc[:, 0].values.reshape(-1, 1)
y_train = train_data.iloc[:, 1].values

# Read test data
test_data = pd.read_excel("E:\TestData.xlsx")  # Update path
X_test = test_data.iloc[:, 0].values.reshape(-1, 1)
y_test = test_data.iloc[:, 1].values

# 选择多项式的阶数范围
degrees = np.arange(2, 10)  # 选择 2 到 9 阶的多项式
best_degree = 2
best_mse = float("inf")
best_model = None

# 交叉验证寻找最优阶数
for d in degrees:
    model = make_pipeline(PolynomialFeatures(d), StandardScaler(), RidgeCV(alphas=np.logspace(-6, 6, 13)))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_degree = d
        best_model = model

# 使用最优阶数的模型进行训练
print(f"Best Polynomial Degree: {best_degree}")
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# 计算误差
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# 绘制训练数据拟合曲线
plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color='gray', alpha=0.5, label="Training Data")
plt.plot(np.sort(X_train, axis=0), y_train_pred[np.argsort(X_train, axis=0)], color='red',
         label=f"Polynomial Fit (Degree {best_degree})")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Optimized Polynomial Regression (Degree {best_degree}) - Training Data")
plt.legend()
plt.show()

# 绘制测试数据拟合曲线
plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='gray', alpha=0.5, label="Test Data")
plt.plot(np.sort(X_test, axis=0), y_test_pred[np.argsort(X_test, axis=0)], color='blue',
         label=f"Polynomial Fit (Degree {best_degree})")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Optimized Polynomial Regression (Degree {best_degree}) - Test Data")
plt.legend()
plt.show()
