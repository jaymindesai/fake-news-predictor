import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score

from src.utils.tokenizer import tokenize

train_data = pd.read_table('../../data/liar/train.tsv')[['label', 'statement']]
test_data = pd.read_table('../../data/liar/test.tsv')[['label', 'statement']]

data = pd.concat([train_data, test_data], ignore_index=True)

X = data['statement']
y_six = data['label']
y_two = data['label'].apply(lambda x: 0 if x in ['TRUE', 'mostly-true', 'half-true'] else 1)

print(y_six.value_counts())
print('')
print(y_two.value_counts())
print('')

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
count_vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False)

X_tfidf = tfidf_vectorizer.fit_transform(X).toarray()
X_bow = count_vectorizer.fit_transform(X).toarray()

pd.DataFrame(X_tfidf).to_csv(path_or_buf='tfidf.csv')
pd.DataFrame(X_bow).to_csv(path_or_buf='bow.csv')

skfolds = StratifiedKFold(n_splits=10, random_state=0)

# clf = GaussianNB()
clf = MultinomialNB()

accuracy_tfidf_six = []
accuracy_bow_six = []
recall_tfidf_six = []
recall_bow_six = []
precision_tfidf_six = []
precision_bow_six = []

for train_index, test_index in skfolds.split(X_tfidf, y_six):
    cloned_clf = clone(clf)

    X_train_folds = X_tfidf[train_index]
    y_train_folds = y_six[train_index]

    X_test_folds = X_tfidf[test_index]
    y_test_folds = y_six[test_index]

    cloned_clf.fit(X_train_folds, y_train_folds)

    y_pred = cloned_clf.predict(X_test_folds)

    accuracy_tfidf_six.append(accuracy_score(y_test_folds, y_pred))
    recall_tfidf_six.append(recall_score(y_test_folds, y_pred, average='macro'))
    precision_tfidf_six.append(precision_score(y_test_folds, y_pred, average='macro'))

for train_index, test_index in skfolds.split(X_bow, y_six):
    cloned_clf = clone(clf)

    X_train_folds = X_bow[train_index]
    y_train_folds = y_six[train_index]

    X_test_folds = X_bow[test_index]
    y_test_folds = y_six[test_index]

    cloned_clf.fit(X_train_folds, y_train_folds)

    y_pred = cloned_clf.predict(X_test_folds)

    accuracy_bow_six.append(accuracy_score(y_test_folds, y_pred))
    recall_bow_six.append(recall_score(y_test_folds, y_pred, average='macro'))
    precision_bow_six.append(precision_score(y_test_folds, y_pred, average='macro'))


tfidf_six = pd.DataFrame(data={'accuracy': accuracy_tfidf_six,
                               'recall': recall_tfidf_six,
                               'precision': precision_tfidf_six})

bow_six = pd.DataFrame(data={'accuracy': accuracy_bow_six,
                             'recall': recall_bow_six,
                             'precision': precision_bow_six})

print(tfidf_six.describe())
print('')
print(bow_six.describe())
print('')

# tfidf_perf_six = pd.Series(accuracy_tfidf_six)
# bow_perf_six = pd.Series(accuracy_bow_six)
#
# print(tfidf_perf_six.describe())
# print('')
# print(bow_perf_six.describe())
# print('')

accuracy_tfidf_two = []
accuracy_bow_two = []
recall_tfidf_two = []
recall_bow_two = []
precision_tfidf_two = []
precision_bow_two = []

for train_index, test_index in skfolds.split(X_tfidf, y_two):
    cloned_clf = clone(clf)

    X_train_folds = X_tfidf[train_index]
    y_train_folds = y_two[train_index]

    X_test_folds = X_tfidf[test_index]
    y_test_folds = y_two[test_index]

    cloned_clf.fit(X_train_folds, y_train_folds)

    y_pred = cloned_clf.predict(X_test_folds)

    accuracy_tfidf_two.append(accuracy_score(y_test_folds, y_pred))
    recall_tfidf_two.append(recall_score(y_test_folds, y_pred))
    precision_tfidf_two.append(precision_score(y_test_folds, y_pred))

for train_index, test_index in skfolds.split(X_bow, y_two):
    cloned_clf = clone(clf)

    X_train_folds = X_bow[train_index]
    y_train_folds = y_two[train_index]

    X_test_folds = X_bow[test_index]
    y_test_folds = y_two[test_index]

    cloned_clf.fit(X_train_folds, y_train_folds)

    y_pred = cloned_clf.predict(X_test_folds)

    accuracy_bow_two.append(accuracy_score(y_test_folds, y_pred))
    recall_bow_two.append(recall_score(y_test_folds, y_pred))
    precision_bow_two.append(precision_score(y_test_folds, y_pred))


tfidf_two = pd.DataFrame(data={'accuracy': accuracy_tfidf_two,
                               'recall': recall_tfidf_two,
                               'precision': precision_tfidf_two})

bow_two = pd.DataFrame(data={'accuracy': accuracy_bow_two,
                             'recall': recall_bow_two,
                             'precision': precision_bow_two})

print(tfidf_two)
print('')
print(bow_two)
print('')

# tfidf_perf_two = pd.Series(accuracy_tfidf_two)
# bow_perf_two = pd.Series(accuracy_bow_two)
#
# print(tfidf_perf_two.describe())
# print('')
# print(bow_perf_two.describe())
# print('')
