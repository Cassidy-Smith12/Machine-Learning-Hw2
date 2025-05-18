import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('hsbdemo.csv')
print(df)

X = df.drop(columns=['id', 'prog', 'cid'])
X = pd.get_dummies(X)
y = df['prog']

X = StandardScaler().fit_transform(X)
print(f'After Standrdization: {X}')

print('Standard')
print(X)

pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)

explained_variance = pca.explained_variance_ratio_
print(f'Variance: {explained_variance}')

cumulative_variance = np.cumsum(explained_variance)
print(f'Cumulative variance: {cumulative_variance}')

plt.plot(cumulative_variance)
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Variance Ratio')
plt.title('PC = 1-10')
plt.show()
