import string

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score


def tokenize(document):
    lemmatizer = WordNetLemmatizer()

    "Break the document into sentences"
    for sent in sent_tokenize(document):

        "Break the sentence into part of speech tagged tokens"
        for token, tag in pos_tag(wordpunct_tokenize(sent)):

            "Apply preprocessing to the token"
            token = token.lower()  # Convert to lower case
            token = token.strip()  # Strip whitespace and other punctuations
            token = token.strip('_')  # remove _ if any
            token = token.strip('*')  # remove * if any

            "If stopword, ignore."
            if token in stopwords.words('english'):
                continue

            "If punctuation, ignore."
            if all(char in string.punctuation for char in token):
                continue

            "If number, ignore."
            if token.isdigit():
                continue

            # Lemmatize the token and yield
            # Note: Lemmatization is the process of looking up a single word form
            # from the variety of morphologic affixes that can be applied to
            # indicate tense, plurality, gender, etc.
            lemma = lemmatizer.lemmatize(token)
            yield lemma


train_data = pd.DataFrame({'label': ['x', 'x', 'y', 'y'],
                           'statement': ['This is a dog, damn!, dog',
                                         'This is a cat, damn!, dog',
                                         'This is a truck, wow!, dog, dog',
                                         'This is a car, wow!, dog, dog']})

test_data = pd.DataFrame({'label': ['x', 'y'],
                          'statement': ['This is a duck, damn!',
                                        'This is a aeroplane, wow!']})

X_train = train_data['statement']
y_train = train_data['label']

X_test = test_data['statement']
y_test = test_data['label']

vectorizer = TfidfVectorizer(tokenizer=tokenize, preprocessor=None, lowercase=False)

vectorizer.fit(X_train)

print(vectorizer.vocabulary_)

X_train_tfidf = vectorizer.transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print(X_train_tfidf.shape)
print(X_test_tfidf.shape)

print(X_train_tfidf.toarray())
print(X_test_tfidf.toarray())

# gnb = GaussianNB()
# gnb.fit(X_train_tfidf.toarray(), y_train)
# Y_pred = gnb.predict(X_test_tfidf.toarray())

# print(str(round(accuracy_score(y_test, Y_pred), 2) * 100) + '%')

# labels = {'barely-true': 0, 'FALSE': 1, 'half-true': 2, 'mostly-true': 3, 'TRUE': 4, 'pants-fire': 5}
#
# YTR = []
# YTS = []
#
# for rec in Y_train.values:
#     YTR.append(labels[rec])
#
# for rec in Y_test.values:
#     YTS.append(labels[rec])
#
# print(YTR)
