import pandas as pd

from kmodes.kmodes.kmodes import KModes

columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party',
           'barely_true', 'FALSE', 'half_true', 'mostly_true', 'pants_on_fire', 'venue']

columns_k_modes = ['label', 'speaker', 'party']

train_data = pd.read_table('../../data/liar/train.tsv')[columns_k_modes].dropna()

X_train = train_data.drop(columns='label')
y_train = train_data['label']

print(len(X_train))

NO_OF_CLUSTERS = 6

km = KModes(n_clusters=NO_OF_CLUSTERS, n_init=10, init='Huang', n_jobs=-1, verbose=2, random_state=0)

print('Model => {}'.format(km))

km = km.fit(X_train.values)

X_df = pd.DataFrame({'cluster': km.labels_, 'label': y_train.values})

for c in range(NO_OF_CLUSTERS):
    print('\t')
    print('Cluster: {}'.format(c))
    print(X_df[X_df['cluster'] == c]['label'].value_counts())