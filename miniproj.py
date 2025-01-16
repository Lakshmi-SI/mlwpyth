!wget -qq https://cdn.iisc.talentsprint.com/CDS/MiniProjects/Bike_Sharing_Dataset.zip
!unzip Bike_Sharing_Dataset.zip

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

df = pd.read_csv('hour.csv')
print(df.head())

print(df.dtypes)


plt.figure(figsize=(10, 6))
sns.countplot(x='hr', data=df, palette='viridis')
plt.title('Distribution of Bike Sharing by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Bike Shares')
plt.xticks(range(0, 24)) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


df1 = pd.read_csv('day.csv')
# Visualize the distribution of the 'count' variable (number of bike shares)
plt.figure(figsize=(10, 6))
# Plot the histogram of the 'count' column
sns.histplot(df1['cnt'], kde=True, bins=30, color='teal', stat='density')
# Customize plot appearance
plt.title('Distribution of Bike Shares (Count)', fontsize=16)
plt.xlabel('Number of Bike Shares', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df1['casual'], kde=True, bins=30, color='blue', stat='density')
plt.title('Distribution of Bike Shares (Casual)', fontsize=16)
plt.xlabel('Number of Bike Shares', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df1['registered'], kde=True, bins=30, color='red', stat='density')
plt.title('Distribution of Bike Shares (Registered)', fontsize=16)
plt.xlabel('Number of Bike Shares', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


sns.catplot(
    data=df1,
    x='weekday',
    y='cnt',
    hue='workingday',  # color by working day
    col='holiday',  # create separate plots for holidays (0/1)
    kind='strip',  # use boxplot for distribution
    palette='Set2',
    height=6,
    aspect=1.5
)
plt.subplots_adjust(top=0.85)
plt.suptitle('Bike Share Count by Weekday, Working Day, and Holiday', fontsize=16)
plt.xlabel('Weekday', fontsize=14)
plt.ylabel('Number of Bike Shares', fontsize=14)
plt.show()


# stacked bar chart for year 2011
df1['dteday'] = pd.to_datetime(df1['dteday'], format='%Y-%m-%d', errors='coerce')
df_2011 = df1[df1['dteday'].dt.year == 2011]
df_2011['month'] = df_2011['dteday'].dt.month
monthly_counts = df_2011.groupby('month')[['casual', 'registered']].sum()
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', stacked=True, color=['#66c2a5', '#fc8d62'], width=0.8)

plt.title('Month-wise Count of Casual and Registered Users (2011)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Users', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='User Type', labels=['Casual', 'Registered'], loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# stacked bar chart for year 2012
df1['dteday'] = pd.to_datetime(df1['dteday'], format='%Y-%m-%d', errors='coerce')
df_2012 = df1[df1['dteday'].dt.year == 2012]
df_2012['month'] = df_2012['dteday'].dt.month
monthly_counts = df_2012.groupby('month')[['casual', 'registered']].sum()
plt.figure(figsize=(10, 6))
monthly_counts.plot(kind='bar', stacked=True, color=['#66c2a5', '#fc8d62'], width=0.8)
plt.title('Month-wise Count of Casual and Registered Users (2012)', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Users', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='User Type', labels=['Casual', 'Registered'], loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


correlation_matrix = df1.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Features', fontsize=16)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df1[['casual', 'registered']], palette="Set2")
plt.title('Box Plot of Casual and Registered Users', fontsize=16)
plt.xlabel('User Type', fontsize=14)
plt.ylabel('Count of Users', fontsize=14)
plt.tight_layout()
plt.show()

columns_ = ['instant', 'casual', 'registered']
df1 = df1.drop(columns = columns_)


print(df1.dtypes)
categorical_vars = df1.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
continuous_vars = df1.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Categorical Variables:", categorical_vars)
print("Continuous Variables:", continuous_vars)


from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
df_minmax_scaled = df1.copy()
df_minmax_scaled[continuous_vars] = min_max_scaler.fit_transform(df1[continuous_vars])

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_array = encoder.fit_transform(df1[categorical_vars])
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_vars))


target = 'cnt'
X = df_minmax_scaled.drop(columns=[target])
y = df_minmax_scaled[target]
X_array = X.values
y_array = y.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


from scipy.linalg import lstsq
X_train_numeric = X_train.select_dtypes(include=[np.number]).values
X_test_numeric = X_test.select_dtypes(include=[np.number]).values
y_train_numeric = y_train.values.astype(float)
y_test_numeric = y_test.values.astype(float)

X_train_with_bias = np.c_[np.ones(X_train_numeric.shape[0]), X_train_numeric]
X_test_with_bias = np.c_[np.ones(X_test_numeric.shape[0]), X_test_numeric]

theta, residuals, rank, singular_values = lstsq(X_train_with_bias, y_train_numeric)
print("Coefficients (theta):", theta)

# Predict on the training and testing sets
y_train_pred = X_train_with_bias @ theta
y_test_pred = X_test_with_bias @ theta
from sklearn.metrics import mean_squared_error, r2_score

mse_train = mean_squared_error(y_train_numeric, y_train_pred)
mse_test = mean_squared_error(y_test_numeric, y_test_pred)
r2_train = r2_score(y_train_numeric, y_train_pred)
r2_test = r2_score(y_test_numeric, y_test_pred)
print(f"Training MSE: {mse_train}, R²: {r2_train}")
print(f"Testing MSE: {mse_test}, R²: {r2_test}")



# Ensure X_train and X_test contain only numeric data
X_train_numeric = X_train.select_dtypes(include=[np.number]).values
X_test_numeric = X_test.select_dtypes(include=[np.number]).values
y_train_numeric = y_train.values.astype(float)
y_test_numeric = y_test.values.astype(float)

X_train_with_bias = np.c_[np.ones(X_train_numeric.shape[0]), X_train_numeric]
X_test_with_bias = np.c_[np.ones(X_test_numeric.shape[0]), X_test_numeric]
m, n = X_train_with_bias.shape
theta = np.random.randn(n)  # Random initialization of coefficients
learning_rate = 0.01  # Set a learning rate
iterations = 1000  # Number of iterations

for i in range(iterations):
    y_pred = X_train_with_bias @ theta
    cost = np.mean((y_pred - y_train_numeric) ** 2)
    gradient = (2 / m) * X_train_with_bias.T @ (y_pred - y_train_numeric)
    theta -= learning_rate * gradient
    if i % 100 == 0:
        print(f"Iteration {i}, Cost: {cost}")
print("Optimized Coefficients (theta):", theta)
y_test_pred = X_test_with_bias @ theta
from sklearn.metrics import mean_squared_error, r2_score
mse_test = mean_squared_error(y_test_numeric, y_test_pred)
r2_test = r2_score(y_test_numeric, y_test_pred)
print(f"Testing MSE: {mse_test}, R²: {r2_test}")



from sklearn.linear_model import SGDRegressor
X_train_numeric = X_train.select_dtypes(include=[np.number])
X_test_numeric = X_test.select_dtypes(include=[np.number])
y_train_numeric = y_train.astype(float)
y_test_numeric = y_test.astype(float)

sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_regressor.fit(X_train_numeric, y_train_numeric)
y_test_pred = sgd_regressor.predict(X_test_numeric)
mse_test = mean_squared_error(y_test_numeric, y_test_pred)
r2_test = r2_score(y_test_numeric, y_test_pred)

print(f"Testing Mean Squared Error (MSE): {mse_test}")
print(f"Testing R² Score: {r2_test}")


from sklearn.linear_model import LinearRegression
X_train_numeric = X_train.select_dtypes(include=[np.number])
X_test_numeric = X_test.select_dtypes(include=[np.number])
y_train_numeric = y_train.astype(float)
y_test_numeric = y_test.astype(float)
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_numeric, y_train_numeric)
y_test_pred = linear_regressor.predict(X_test_numeric)
mse_test = mean_squared_error(y_test_numeric, y_test_pred)
print(f"Testing Mean Squared Error (MSE): {mse_test}")


from sklearn.metrics import r2_score
y_test_pred = linear_regressor.predict(X_test_numeric)
r2_value = r2_score(y_test_numeric, y_test_pred)
print(f"R² Score: {r2_value}")



import matplotlib.pyplot as plt
import pandas as pd
coefficients = linear_regressor.coef_
feature_names = X_train_numeric.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Linear Regression Coefficients')
plt.grid(True)
plt.tight_layout()
plt.show()

alpha_values = [0.0001, 0.001,0.01, 0.1, 1, 10, 100]
from sklearn.linear_model import Lasso
best_alpha = None
best_mse = float('inf')
best_r2 = -float('inf')
results = []
for alpha in alpha_values
    lasso_regressor = Lasso(alpha=alpha, random_state=42)
    lasso_regressor.fit(X_train_numeric, y_train_numeric)
    y_test_pred = lasso_regressor.predict(X_test_numeric)
    mse = mean_squared_error(y_test_numeric, y_test_pred)
    r2 = r2_score(y_test_numeric, y_test_pred)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_r2 = r2
    results.append((alpha, mse, r2))
print(f"Best Alpha: {best_alpha}")
print(f"Minimum MSE: {best_mse}")
print(f"Maximum R² Score: {best_r2}")
for alpha, mse, r2 in results:
    print(f"Alpha: {alpha}, MSE: {mse}, R² Score: {r2}")


from sklearn.linear_model import Ridge
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
best_alpha = None
best_mse = float('inf')
best_r2 = -float('inf')
results = []
for alpha in alpha_values:
    ridge_regressor = Ridge(alpha=alpha, random_state=42)
    ridge_regressor.fit(X_train_numeric, y_train_numeric)
    y_test_pred = ridge_regressor.predict(X_test_numeric)
    mse = mean_squared_error(y_test_numeric, y_test_pred)
    r2 = r2_score(y_test_numeric, y_test_pred)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_r2 = r2
    results.append((alpha, mse, r2))
print(f"Best Alpha: {best_alpha}")
print(f"Minimum MSE: {best_mse}")
print(f"Maximum R² Score: {best_r2}")
for alpha, mse, r2 in results:
    print(f"Alpha: {alpha}, MSE: {mse}, R² Score: {r2}")


from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
best_alpha = None
best_mse = float('inf')
best_r2 = -float('inf')
results = []
for alpha in alpha_values:
    elasticnet_regressor = ElasticNet(alpha=alpha, random_state=42)
    elasticnet_regressor.fit(X_train_numeric, y_train_numeric)
    y_test_pred = elasticnet_regressor.predict(X_test_numeric)
    mse = mean_squared_error(y_test_numeric, y_test_pred)
    r2 = r2_score(y_test_numeric, y_test_pred)
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha
        best_r2 = r2
    results.append((alpha, mse, r2))
print(f"Best Alpha: {best_alpha}")
print(f"Minimum MSE: {best_mse}")
print(f"Maximum R² Score: {best_r2}")
for alpha, mse, r2 in results:
    print(f"Alpha: {alpha}, MSE: {mse}, R² Score: {r2}")






