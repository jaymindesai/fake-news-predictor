import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_score
from sklearn.metrics import recall_score

submit=pd.read_csv('submit.csv')
predicted_id=submit.id

tsv_file_test='test.tsv'
csv_table_test=pd.read_table(tsv_file_test,sep='\t')
csv_table_test['label']=csv_table_test['label'].apply(lambda x: 'TRUE' if x in ['TRUE', 'mostly-true', 'half-true'] else 'FALSE')
target_id=csv_table_test['id']
label_array=['FALSE','TRUE']
def minimum(a, n):
    maxpos = a.index(max(a))
    return maxpos


X=[]
Y=[]
A=[]
B=[]
Z=[]
for index, row in submit.iterrows():
      a=[row['FALSE'],row['TRUE']]
      pos = minimum(a, len(a))
      X.append([row['id'],label_array.__getitem__(pos)])
      Y.append(row['id'])
      B.append(label_array.__getitem__(pos))

for index,row in csv_table_test.iterrows():
    if row['id'] in Y:
        Z.append([row['id'],row['label']])
        A.append(row['label'])


print("Accuracy is",accuracy_score(A,B))


print("Recall is ",recall_score(A,B,average='binary',pos_label="FALSE"))

print("F1-Score",f1_score(A,B,pos_label='FALSE'))
#average_precision = average_precision_score(A, B,pos_label='FALSE')
print(precision_score(A, B, average='macro'))