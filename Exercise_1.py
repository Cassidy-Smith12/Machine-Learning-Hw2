import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

df = pd.read_csv('hsbdemo.csv')

X = df.drop(columns=['id', 'prog', 'cid'])
X = pd.get_dummies(X)
y = df['prog']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=3) 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print('Model accuracy score: ', accuracy_score(y_test, pred))

conf_matrix = confusion_matrix(y_test, pred)
print(f'\nConfusion Matrix: \n{conf_matrix}')

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


misclassified = X_test[y_test != pred]
misclassified['Actual'] = y_test[y_test != pred]
misclassified['Predicted'] = pred[y_test != pred]
print("\nMisclassified data points (predicted vs actual):")
print(misclassified[['Predicted', 'Actual']])
