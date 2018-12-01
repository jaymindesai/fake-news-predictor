import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from src.utils.tokenizer import tokenize

train_data = pd.read_table('../../data/liar/train.tsv')[['label', 'statement']]
test_data = pd.read_table('../../data/liar/test.tsv')[['label', 'statement']]

X_train = train_data['statement']
y_train = train_data['label']

X_test = test_data['statement']
y_test = test_data['label']

vectorizer = TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=False).fit(X_train)
# vectorizer = CountVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=False).fit(X_train)

X_train_tfidf = vectorizer.transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

# print(X_train_tfidf.shape)
# print(X_test_tfidf.shape)

# print(X_train_tfidf.toarray())
# print(X_test_tfidf.toarray())

nb = MultinomialNB()
nb.fit(X_train_tfidf.toarray(), y_train)
Y_pred = nb.predict(X_test_tfidf.toarray())

# print(str(round(accuracy_score(y_test, Y_pred), 2) * 100) + '%')
print(accuracy_score(y_test, Y_pred))