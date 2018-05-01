import pandas as pd

import seaborn as sns

from sklearn.linear_model import LogisticRegression

import os

import numpy

import matplotlib.pyplot as plt 


rtrain = pd.read_csv(os.path.join('train', 'in.tsv'), sep='\t', header=None)


print(rtrain.describe())

print('Zbior treningowy')

print('Rozkład próby %: ', end ='')
print(str(sum(rtrain[0] == 'g') / len(rtrain)))

print('Dokładność algorytmu zero rule na zbiorze treningowym:', end ='')
print(str(1 - sum(rtrain[0] == 'g') / len(rtrain)))


lr = LogisticRegression()
lr.fit(rtrain[1].values.reshape(-1, 1), rtrain[0])


TP = sum((lr.predict(rtrain[1].values.reshape(-1, 1)) == rtrain[0]) & (lr.predict(rtrain[1].values.reshape(-1, 1)) == 'g'))


TN = sum((lr.predict(rtrain[1].values.reshape(-1, 1)) == rtrain[0]) & (lr.predict(rtrain[1].values.reshape(-1, 1)) == 'b'))

FP = sum((lr.predict(rtrain[1].values.reshape(-1, 1)) != rtrain[0]) & (lr.predict(rtrain[1].values.reshape(-1, 1)) == 'g'))

FN = sum((lr.predict(rtrain[1].values.reshape(-1, 1)) != rtrain[0]) & (lr.predict(rtrain[1].values.reshape(-1, 1)) == 'b'))
print('True Positives: ', TP)
print('True Negatives: ', TN)
print('False Positives: ', FP)
print('True Negatives: ', TN)

print('Accurancy: ',end= ' ')  
print((TP+TN)/(TP+FP+FN+TN))

print('Specifcity: ', end=' ')
print(TN/(TN+FP))

print('Sensitivity: ',end=' ') 
print(TP/(TP+FN))
print('-'*100)
print('Zbior deweloperski')
rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', header=None)
rdev = pd.DataFrame(rdev)
rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names = ['y'])

print('Rozkład próby %: ', end ='')
print(str(sum(rdev_expected['y'] == 'g') / len(rdev_expected)))
print('Dokładność algorytmu zero rule na zbiorze deweloperskim:', end ='')
print(str(1 - sum(rdev_expected['y'] == 'g') / len(rdev)))

TP = sum((lr.predict(rdev[0].values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'g'))
TN = sum((lr.predict(rdev[0].values.reshape(-1, 1)) == rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'b'))
FP = sum((lr.predict(rdev[0].values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'g'))
FN = sum((lr.predict(rdev[0].values.reshape(-1, 1)) != rdev_expected['y']) & (lr.predict(rdev[0].values.reshape(-1, 1)) == 'b'))

print('True Positives: ', TP)
print('True Negatives: ', TN)
print('False Positives: ', FP)
print('True Negatives: ', TN)

print('Accurancy: ',end= ' ')  
print((TP+TN)/(TP+FP+FN+TN))

print('Specifcity: ', end=' ')
print(TN/(TN+FP))

print('Sensitivity: ',end=' ') 
print(TP/(TP+FN))
print('-'*100)


rtest = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', header=None)
rtest = rtest[0].values.reshape(-1, 1)
rdev = rdev[0].values.reshape(-1, 1)



file = open(os.path.join('dev-0', 'out.tsv'), 'w')

for line in list(lr.predict(rdev)):
    file.write(str(line)+ '\n')



file = open(os.path.join('test-A', 'out.tsv'), 'w')

for line in list(lr.predict(rtest)):
    file.write(str(line) + '\n')







