import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
import seaborn as sns

# Load the California Housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Explore the dataset
print(df.head())
print(df.info())

# Visualize the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['MedHouseVal'], kde=True, bins=30, color='teal', stat='density')
plt.title('Distribution of Median House Value', fontsize=16)
plt.xlabel('Median House Value', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Features', fontsize=16)
plt.tight_layout()
plt.show()

# Feature preprocessing
target = 'MedHouseVal'
categorical_vars = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
continuous_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical Variables:", categorical_vars)
print("Continuous Variables:", continuous_vars)

# Scaling continuous features
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[continuous_vars] = scaler.fit_transform(df[continuous_vars])

# Define features and target
X = df_scaled.drop(columns=[target])
y = df_scaled[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Linear Regression
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"Linear Regression Testing MSE: {mse_test}")
print(f"Linear Regression R² Score: {r2_test}")

# Visualize Linear Regression Coefficients
coefficients = linear_regressor.coef_
feature_names = X_train.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Linear Regression Coefficients')
plt.grid(True)
plt.tight_layout()
plt.show()

# Lasso Regression
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
best_alpha = None
best_mse = float('inf')
best_r2 = -float('inf')

for alpha in alpha_values:
    lasso_regressor = Lasso(alpha=alpha, random_state=42)
    lasso_regressor.fit(X_train, y_train)
    y_test_pred = lasso_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_r2 = r2
    print(f"Lasso Alpha: {alpha}, MSE: {mse}, R²: {r2}")

print(f"Best Lasso Alpha: {best_alpha}")
print(f"Best Lasso MSE: {best_mse}")
print(f"Best Lasso R²: {best_r2}")

# Ridge Regression
best_alpha = None
best_mse = float('inf')
best_r2 = -float('inf')

for alpha in alpha_values:
    ridge_regressor = Ridge(alpha=alpha, random_state=42)
    ridge_regressor.fit(X_train, y_train)
    y_test_pred = ridge_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_r2 = r2
    print(f"Ridge Alpha: {alpha}, MSE: {mse}, R²: {r2}")

print(f"Best Ridge Alpha: {best_alpha}")
print(f"Best Ridge MSE: {best_mse}")
print(f"Best Ridge R²: {best_r2}")

# ElasticNet Regression
best_alpha = None
best_mse = float('inf')
best_r2 = -float('inf')

for alpha in alpha_values:
    elasticnet_regressor = ElasticNet(alpha=alpha, random_state=42)
    elasticnet_regressor.fit(X_train, y_train)
    y_test_pred = elasticnet_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_r2 = r2
    print(f"ElasticNet Alpha: {alpha}, MSE: {mse}, R²: {r2}")

print(f"Best ElasticNet Alpha: {best_alpha}")
print(f"Best ElasticNet MSE: {best_mse}")
print(f"Best ElasticNet R²: {best_r2}")
