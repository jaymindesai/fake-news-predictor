import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import sqrt


train_data = pd.read_table('../../data/liar/train.tsv')[['label', 'statement']]
test_data = pd.read_table('../../data/liar/test.tsv')[['label', 'statement']]
data = pd.concat([train_data, test_data], ignore_index=True)

X = data['statement']

Sum_of_squared_distances = []

clusters = int(sqrt(len(X)))

K = range(1, clusters)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()