import pandas as pd
from collections import Counter

x = ['a a']
y = ['a b']

c = Counter(x)
c.update(y)
print(c)

# y = [1, 2]
# df = pd.DataFrame([x, y])
#
# print(df)

# s = pd.Series([1, 2, 3, 100], name='t')
# print(s)
