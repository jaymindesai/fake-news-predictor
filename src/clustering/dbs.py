import pandas as pd
from sklearn.cluster import DBSCAN
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# dbs = ['label', 'speaker']
dbs = ['label', 'barely_true', 'FALSE', 'half_true', 'mostly_true', 'pants_on_fire']

train_data = pd.read_table('../../data/liar/train.tsv')[dbs].dropna()

X_train = train_data.drop(columns='label')
y_train = train_data['label']

tuner = {
    'eps': list(range(1, 30)),
    'min_samples': list(range(5, 20)),
    'metric': ['euclidean', 'minkowski'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

dbs_clf = DBSCAN(eps=4, min_samples=100, n_jobs=-1, metric='euclidean')

dbs_clf.fit(X_train)

X_df = pd.DataFrame({'cluster': dbs_clf.labels_, 'label': y_train.values})

N = len(X_df['cluster'].value_counts().index)

for c in range(N):
    print('\t')
    print('Cluster: {}'.format(c))
    print(X_df[X_df['cluster'] == c]['label'].value_counts())

