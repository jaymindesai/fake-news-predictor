import pandas as pd
import numpy as np

from kmodes.kmodes.kprototypes import KPrototypes

columns_k_proto = ['label', 'speaker', 'party', 'barely_true', 'FALSE', 'half_true', 'mostly_true', 'pants_on_fire']

train_data = pd.read_table('../../data/liar/train.tsv')[columns_k_proto].dropna()

columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party',
           'barely_true', 'FALSE', 'half_true', 'mostly_true', 'pants_on_fire', 'venue']

X_train = train_data.drop(columns='label')
y_train = train_data['label']

print(len(X_train))

NO_OF_CLUSTERS = 6

km = KPrototypes(n_clusters=NO_OF_CLUSTERS, init='Huang', n_jobs=-1, verbose=2, random_state=0)

print('Model => {}'.format(km))

km = km.fit(X_train.values, categorical=[0, 1])

X_df = pd.DataFrame({'cluster': km.labels_, 'label': y_train.values})

for c in range(NO_OF_CLUSTERS):
    print('\t')
    print('Cluster: {}'.format(c))
    print(X_df[X_df['cluster'] == c]['label'].value_counts())