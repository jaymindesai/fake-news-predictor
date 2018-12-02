from functools import reduce

import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

dbs = ['label', 'barely_true', 'FALSE', 'half_true', 'mostly_true', 'pants_on_fire']

train_data = pd.read_table('../../data/liar/train.tsv')[dbs].dropna()

X_train = train_data.drop(columns='label')
y_train = train_data['label']

nn = NearestNeighbors(n_neighbors=4, metric='euclidean')

nn.fit(X_train)

distances, indices = nn.kneighbors(X_train)

print(distances)

sum = []
for d in distances:
    sum.append(reduce(lambda x, y: x + y, d) / (len(d) - 1))

frame = pd.DataFrame(sum)
print(frame)
print(frame.describe())
print('')
print('90th Percentile => {}'.format(np.percentile(frame.values, 90)))

# print(sum/len(distances))
