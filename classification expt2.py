import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, 3].values # estimated salary
y = df.iloc[:, -1].values
X = X.reshape(-1, 1)
df.head()

#OR

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target
df.head()
X = df.iloc[:, 3].values 
y = df.iloc[:, -1].values
X = X.reshape(-1, 1)


print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(X_train)

print(X_test)

print(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)

print(X_test)

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.predict(sc.transform([[87000]])))

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Creating a confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Use sklearn to plot precision-recall curves

from sklearn.metrics import PrecisionRecallDisplay
display = PrecisionRecallDisplay.from_estimator(
    classifier,  # Your trained model
    X_test,      # Test features
    y_test,      # True labels
    name='Logistic Regression'  # Name to display on the plot
)



from sklearn.metrics import roc_curve
classifier.fit(X_train, y_train)
pred_prob1 = classifier.predict_proba(X_test)
# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


plt.style.use('ggplot') #using ggplot cause seaborn isn't working
# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();

#IF USING DIABETES.CSV############################3
DF = pd.read_csv('diabetes.csv')
print(DF.head())
print(DF.isnull().sum())
# Separating the data into independent and dependent variables
X = DF.iloc[:, :-1].values  # All rows, all columns except the last
y = DF.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42, max_iter=500)  # Increase max_iter to 500
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Print predictions (optional)
print("\nPredictions on the test set:")
print(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Print results
print("\nConfusion Matrix:")
print(cm)
print(f"\nAccuracy: {accuracy:.2f}"
################################################
      

from sklearn.metrics import roc_curve, roc_auc_score
# Obtain predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]
# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# Compute AUC
auc_score = roc_auc_score(y_test, y_prob)
# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line for random guessing
plt.title('ROC Curve for Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()


from sklearn.preprocessing import LabelEncoder
#label_encoder = LabelEncoder()
#df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Converts 'Male' to 1 and 'Female' to 0
# After encoding, concatenate 'Gender' with the 'Age' and 'EstimatedSalary' columns
X = df.iloc[:, [1, 2, 3]].values
y = df.iloc[:, -1].values
df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#sc = StandardScaler()
#X_train[:, 1:] = sc.fit_transform(X_train[:, 1:])  # Only scale Age and EstimatedSalary (columns 1 and 2)
#X_test[:, 1:] = sc.transform(X_test[:, 1:])

#FOR BREAST CANCER ONE
sc = StandardScaler()
X_train[:, :3] = sc.fit_transform(X_train[:, :3])  # Scale the first three columns
X_test[:, :3] = sc.transform(X_test[:, :3])       # Apply the same transformation to the test set


softmax_reg = LogisticRegression(multi_class='multinomial', # switch to Softmax Regression
                                     solver='lbfgs', # handle multinomial loss, L2 penalty
                                   C=10)
softmax_reg.fit(X, y)

softmax_reg.predict(sc.transform([[30,20,140]]))
softmax_reg.predict_proba(sc.transform([[30,20,140]]))


from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=1)

!pip install mlxtend
import matplotlib.gridspec as gridspec
from mlxtend.plotting import plot_decision_regions
gs = gridspec.GridSpec(3, 2)
fig = plt.figure(figsize=(14,10))
label = 'Logistic Regression'
clf = LogisticRegression()
clf.fit(X, y)
fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
plt.title(label)
plt.show()
