import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = np.arange(1, 25).reshape(12, 2)
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

print(X)

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4, random_state=4)
X_train

X_test

y_train

y_test

rng = np.random.RandomState(1)              
x = 10 * rng.rand(50)                       
y = 2 * x - 5 + rng.randn(50)               
plt.scatter(x, y, c='g');


model = LinearRegression(fit_intercept=True)                   
model.fit(x[:, np.newaxis], y)                                 
xfit = np.linspace(0, 10, 1000)                                
yfit = model.predict(xfit[:, np.newaxis])                      
plt.scatter(x, y, c='g')
plt.plot(xfit, yfit, 'k');

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y

X, y = make_data(40)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),
                                         X, y, cv=7,
                                         train_sizes=np.linspace(0.3, 1, 25))

    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1],
                 color='gray', linestyle='dashed')

    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree = {0}'.format(degree), size=14)
    ax[i].legend(loc='best')

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

print(df.columns)

df.describe()

df.info()

df.isna().sum()

plt.style.use('ggplot')
sns.pairplot(df)

plt.figure(figsize=(8, 8))
sns.heatmap(df.corr(), annot=True, linewidth=0.5, center=0)
plt.show()


df.head()
df.dtypes
df.isna().sum()
df = df.dropna()
X = df[['MedInc',	'HouseAge',	'AveRooms', 'AveBedrms',	'Population',	'AveOccup',	'Latitude',	'Longitude']]
y = df['target']
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state= 101)
X_train.head()

lr = LinearRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)

print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))
print('Coefficient of Determination:', r2_score(y_test, pred))

pred = lr.predict(X_test)

print('Predicted value:', pred[2])
print('Actual value:', y_test.values[2])


SPECIFIC TO AUTO_MPG DATASET 

# Removing '?' from horsepower column
auto = auto[auto['horsepower'] != '?']
auto['horsepower'].unique()


# Converting horsepower column datatype from string to float
auto['horsepower'] = auto['horsepower'].astype(float)
auto.dtypes


# Prediction features
X = auto[['displacement', 'horsepower', 'acceleration', 'model year', 'origin']]
#X = auto[['displacement', 'horsepower', 'acceleration', 'model-year']]
# Target feature
y = auto['mpg']

pred = lr.predict(X_test)
print('Predicted fuel consumption(mpg):', pred[2])
print('Actual fuel consumption(mpg):', y_test.values[2])




FOR THE REAL ESTATE CSV 

Import the same. 

df = pd.read_csv('/content/Real estate.csv')


df_binary = df[['X2 house age', 'Y house price of unit area']]

df.head()


columns_to_drop = ['X1 transaction date', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
df = df.drop(columns=columns_to_drop)

null_values = df.isnull()
print(null_values)

sns.scatterplot(data=df, x='X2 house age', y='Y house price of unit area', s=75)

X = df[['X2 house age']]
y = df['Y house price of unit area']
X.head()

Train test split, fit using lr

# Data scatter of predicted values
pred = lr.predict(X_test)
sns.scatterplot(data=df, x='X2 house age', y='Y house price of unit area', s=75)

RIDGE REGRESSION:

n_samples, n_features = 15, 10
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

rdg = linear_model.Ridge(alpha = 0.5)                  # instantiate Ridge regressor
rdg.fit(X, y)
rdg.score(X,y)



LASSO:

Lreg = linear_model.Lasso(alpha = 0.5)
Lreg.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])

Lreg.predict([[0,1]])

#weight vectors
Lreg.coef_

#Calculating intercept
Lreg.intercept_

#Calculating number of iterations
Lreg.n_iter_


ELASTIC NET 

ENreg = linear_model.ElasticNet(alpha = 0.5,random_state = 0)
ENreg.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])

ENreg.predict([[0,1]])

#weight vectors
ENreg.coef_

#Calculating intercept
ENreg.intercept_

#Calculating number of iterations
ENreg.n_iter_


SIGNIFICANCE OF ALPHA

Lreg = linear_model.Lasso(alpha = 0.25)
Lreg.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
Lreg.coef_  #weight vectors


Lreg = linear_model.Lasso(alpha = 0.5)
Lreg.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
Lreg.coef_  #weight vectors

Lreg = linear_model.Lasso(alpha = 0.75)
Lreg.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
Lreg.coef_  #weight vectors
