from collections import Counter
import pandas as pd
from src.utils.tokenizer import tokenize


def count_words():
    train_data = pd.read_table('../../data/liar/train.tsv')[['label', 'statement']]
    test_data = pd.read_table('../../data/liar/test.tsv')[['label', 'statement']]
    data = pd.concat([train_data, test_data], ignore_index=True)

    fake_counts = Counter()
    true_counts = Counter()

    for _, row in data.iterrows():
        # print(row)
        tokens = [t for t in tokenize(row['statement'])]
        if row['label'] in ['TRUE', 'mostly-true', 'half-true']:
            true_counts.update(tokens)
        else:
            fake_counts.update(tokens)

    print('True', true_counts.most_common(10))
    print('Fake', fake_counts.most_common(10))

    true_words = set()
    fake_words = set()

    with open('word_freq_true.txt', 'w') as file:
        for word, freq in true_counts.most_common():
            true_words.add(word)
            file.write('{}: {}\n'.format(word, freq))

    with open('word_freq_fake.txt', 'w') as file:
        for word, freq in fake_counts.most_common():
            fake_words.add(word)
            file.write('{}: {}\n'.format(word, freq))

    with open('true_fake_diff.txt', 'w') as file:
        for word in fake_words - true_words:
            file.write('{}: {}\n'.format(word, fake_counts[word]))


if __name__ == '__main__':
    count_words()
