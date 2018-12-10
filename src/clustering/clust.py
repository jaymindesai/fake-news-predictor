from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, precision_score

from src.utils.tokenizer import tokenize


def prep_data():
    train_data = pd.read_table('../../data/liar/train.tsv')[['label', 'statement']]
    test_data = pd.read_table('../../data/liar/test.tsv')[['label', 'statement']]
    data = pd.concat([train_data, test_data], ignore_index=True)

    X = data['statement']
    y_six = data['label']
    y_two = data['label'].apply(lambda x: 'TRUE' if x in ['TRUE', 'mostly-true', 'half-true'] else 'FALSE')

    _print_vc(y_six)
    _print_vc(y_two)

    y_two = y_two.apply(lambda x: 0 if x is 'TRUE' else 1)

    prep = []
    for x in X.values:
        prep.append(' '.join([t for t in tokenize(x)]))

    X_prep = pd.Series(prep)

    return X_prep, y_six, y_two


def nb_helper(X_prep, y_six, y_two, classifier, n_folds, desc):
    tfidf_vectorizer = TfidfVectorizer(lowercase=False)
    count_vectorizer = CountVectorizer(lowercase=False)

    # _classify(X_prep, y_six, classifier, tfidf_vectorizer, n_folds, 6, desc + '_TFiDF')
    _classify(X_prep, y_six, classifier, count_vectorizer, n_folds, 6, desc + '_BoW')
    # _classify(X_prep, y_two, classifier, tfidf_vectorizer, n_folds, 2, desc + '_TFiDF')
    # _classify(X_prep, y_two, classifier, count_vectorizer, n_folds, 2, desc + '_BoW')


def _classify(X, y, clf, vec, n_folds, n_class, operation='_'):

    _find_elbow(vec.fit_transform(X).toarray(), desc=operation)

    # skfolds = StratifiedKFold(n_splits=n_folds, random_state=100)
    #
    # accuracy = []
    # recall = []
    # precision = []
    #
    # for train_index, test_index in skfolds.split(X, y):
    #     X_train_def = X[train_index]
    #     y_train = y[train_index]
    #
    #     X_test_def = X[test_index]
    #     y_test = y[test_index]
    #
    #     X_train = vec.fit_transform(X_train_def).toarray()
    #     X_test = vec.transform(X_test_def).toarray()
    #
    #     clust = True
    #
    #     if clust:
    #         km = KMeans(n_clusters=0, random_state=100, precompute_distances=True, verbose=0).fit(X_train)
    #
    #         X_train = X_train.assign(cluster=km.labels_)
    #         X_test = X_test.assign(cluster=km.predict(X_test))
    #
    #         y_true = []
    #         y_pred = []
    #
    #         clusters = X_test['cluster'].value_counts().index
    #         for c in clusters:
    #             X_train_new = X_train[X_train['cluster'] == c]
    #             y_train_new = y_train[X_train_new.index.values]
    #             X_test_new = X_test[X_test['cluster'] == c]
    #             y_test_new = y_test[X_test_new.index.values]
    #
    #             y_true.extend(y_test_new.values)
    #
    #             labels_train = set(y_train_new)
    #             if len(labels_train) == 1:
    #                 p = [labels_train.pop()] * len(X_test_new)
    #                 y_pred.extend(p)
    #             else:
    #                 cloned_clf = clone(clf)
    #                 cloned_clf.fit(X_train_new, y_train_new)
    #                 y_pred.extend(cloned_clf.predict(X_test_new))
    #
    #         accuracy.append(accuracy_score(y_true, y_pred))
    #         recall.append(recall_score(y_true, y_pred, average='binary' if n_class == 2 else 'macro'))
    #         precision.append(precision_score(y_true, y_pred, average='binary' if n_class == 2 else 'macro'))
    #
    # results = pd.DataFrame(data={'accuracy': accuracy, 'recall': recall, 'precision': precision})
    #
    # print('{} Fold CV Results for {} with {} Class Labels:'.format(n_folds, operation, n_class))
    # print('')
    # print(results.describe())
    # print('')
    # print('-----')
    # print('\t')


def _print_vc(y):
    vc = y.value_counts()
    n_class = len(vc)
    print('{} Class Labels:'.format(n_class))
    print('')
    for i in vc.index:
        print('{}: {}'.format(i, vc[i]))
    print('')
    print('-----')
    print('\t')


def _find_elbow(X, desc):
    sum_of_squared_distances = []

    clusters = int(sqrt(len(X)))

    K = range(2, 10)
    for k in K:
        km = KMeans(n_clusters=k, precompute_distances=True, n_jobs=-1)
        km = km.fit(X)
        sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For {}'.format(desc))
    plt.show()


if __name__ == '__main__':
    print('\t')
    X_prep, y_six, y_two = prep_data()

    # ...
    #
    # @Param: classifier => Any SciKit-Learn Classifier
    # @Param: n_folds => Number of Folds
    # @Param: desc => Name of Classifier
    #
    # ...

    nb_helper(X_prep, y_six, y_two, classifier=GaussianNB(), n_folds=10, desc='Gaussian_NB')
    # nb_helper(X_prep, y_six, y_two, classifier=MultinomialNB(), n_folds=10, desc='Multinomial_NB')
