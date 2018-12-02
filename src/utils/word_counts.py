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
        print(row)
        tokens = [t for t in tokenize(row['statement'])]
        if row['label'] in ['TRUE', 'mostly-true', 'half-true']:
            true_counts.update(tokens)
        else:
            fake_counts.update(tokens)

    print('True', true_counts.most_common(10))
    print('Fake', fake_counts.most_common(10))
