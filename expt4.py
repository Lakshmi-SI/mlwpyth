import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import support vector regressor algorithm
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Import modelling methods
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
# Import the model performance evaluation metrics
from sklearn import metrics
# Import Adaboost, Gradient Boost, Random Forest and Stacking algorithm
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import fetch_california_housing   # to import boston housing dataset
# to visualize decision boundaries
import graphviz
import xgboost as xgb
from xgboost import XGBRegressor


df = pd.read_csv('indian_liver_patient.csv')
df.head()


# Check for missing values
df.isnull().sum()


# Drop missing values
df1 = df.dropna()
df1.isnull().any()


# Visualize correlation matrix
fig, ax = plt.subplots(figsize=(7,7))
from sklearn.preprocessing import LabelEncoder
# Apply label encoding to categorical columns (assuming 'Gender' is a column with 'Male' and 'Female')
encoder = LabelEncoder()
df1['Gender'] = encoder.fit_transform(df1['Gender'])
sns.heatmap(abs(df1.corr()), annot=True, square=True, cbar=False, ax=ax, linewidths=0.25);


# Drop correlated features
df2 = df1.drop(columns= ['Direct_Bilirubin', 'Alamine_Aminotransferase', 'Total_Protiens'])


df2['Dataset'] = df2['Dataset'].replace(1,0)
df2['Dataset'] = df2['Dataset'].replace(2,1)


print('How many people have disease:', '\n', df2.groupby('Gender')[['Dataset']].sum(), '\n')
print('How many people participated in the study:', '\n', df2.groupby('Gender')[['Dataset']].count())


print('Percentage of people with the disease depending on gender:')
df2.groupby('Gender')[['Dataset']].sum()/ df2.groupby('Gender')[['Dataset']].count()


# defining the X and y variables
X = df2[['Gender', 'Total_Bilirubin','Alkaline_Phosphotase','Aspartate_Aminotransferase','Albumin','Albumin_and_Globulin_Ratio']]
y = pd.Series(df2['Dataset'])


labelencoder = LabelEncoder()
X['Gender'] = labelencoder.fit_transform(X['Gender'])


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


ADB = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                         n_estimators=125,
                         learning_rate = 0.6,
                         random_state=42)
ADB.fit(X_train, y_train)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)



# calculating model evaluation metrics using cross_val_score like accuracy, R2 score, etc.
n_scores = cross_val_score(ADB, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
('Accuracy: %.3f' % (np.mean(n_scores)*100))


labels = ADB.predict(X_test)
matrix = metrics.confusion_matrix(y_test, labels)
# creating a heat map to visualize confusion matrix
sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


logit_roc_auc = metrics.roc_auc_score(y_test, labels)
fpr, tpr, thresholds = metrics.roc_curve(y_test, ADB.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='(area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


boston = fetch_california_housing()
print(boston.keys())
print("shape of dataset",boston.data.shape)


print(boston.feature_names)

df = pd.DataFrame(boston.data)
df.columns = boston.feature_names

df.head()

df['PRICE'] = boston.target

df.info()

X, y = df.iloc[:,:-1],df.iloc[:,-1]

xtrain, xtest, ytrain, ytest=train_test_split(X, y, random_state=12, test_size=0.15)

# with new parameters
gbr1 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', n_estimators=600,
    max_depth=5,
    learning_rate=0.01,
    min_samples_split=4)
# with default parameters
gbr = GradientBoostingRegressor()


# fit with default parameters
gbr.fit(xtrain, ytrain)
ypred = gbr.predict(xtest)
# calculating Mean Squared Error
mse = metrics.mean_squared_error(ytest,ypred)
# mse for default model
print("MSE: %.2f" % mse)


# fit by passing hyperparameters
gbr1.fit(xtrain, ytrain)
ypred1 = gbr1.predict(xtest)
# calculating Mean Squared Error
mse1 = metrics.mean_squared_error(ytest, ypred1)
# mse for regularized model
print("MSE: %.2f" % mse1)


x_ax = range(len(ytest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


x_ax = range(len(ytest))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred1, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


xgb_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                           max_depth = 5, alpha = 10, n_estimators = 10)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xgb_reg.fit(X_train,y_train)
y_pred = xgb_reg.predict(X_test)

mse2 = metrics.mean_squared_error(y_test, y_pred)
print("MSE: %f" % (mse))


xgb.plot_importance(xgb_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
xgb = XGBRegressor()
rf = RandomForestRegressor(n_estimators=400, max_depth=5, max_features=6)
ridge = Ridge()
lasso = Lasso()
svr = SVR(kernel='rbf')







