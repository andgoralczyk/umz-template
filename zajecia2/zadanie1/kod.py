
import pandas as pd

import seaborn as sns

from sklearn.linear_model import LogisticRegression

import os

import numpy

import matplotlib.pyplot as plt 




rtrain = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names=["Occupancy", "date", "Temperature","Humidity", "Light", "CO2", "HumidityRatio"])
print('training set')
print('Occupancy % :', end='')

print(sum(rtrain.Occupancy) / len(rtrain))

print('zero rule model accuracy on training set is',1 - sum(rtrain.Occupancy) / len(rtrain))

lr = LogisticRegression()
lr.fit(rtrain.CO2.values.reshape(-1, 1), rtrain.Occupancy) 
print('lr model on CO2 only accuracy on training data: ', end='')
print(sum(lr.predict(rtrain.CO2.values.reshape(-1, 1))
             == rtrain.Occupancy) / len(rtrain))

lr_full = LogisticRegression()

X = pd.DataFrame(rtrain, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
lr_full.fit(X, rtrain.Occupancy)

TP=(sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 1)))

TN=(sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 0)))

FP=(sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 1)))

FN=(sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 0)))


print('Accurancy: ',end= ' ')  
print((TP+TN)/(TP+FP+FN+TN))

print('Specifcity: ', end=' ')
print(TN/(TN+FP))

print('Sensitivity: ',end=' ') 
print(TP/(TP+FN))

print('Matrix: ')
print('True Positives: ', TP)
print('True Negatives: ', TN)
print('False Positives: ', FP)
print('True Negatives: ', TN)
CM_train=numpy.array([[TP, FP], [FN, TN]])
print(CM_train)
print('-'*100)


print('developmnet set')
rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

rdev = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'])

print('zero rule model accurancy on development set is', end= ' ')
print(1 - sum(rdev_expected['y']) / len(rdev))

print('accuracy on dev data (full model):', end = '')

print(sum(lr_full.predict(rdev) == rdev_expected['y'] ) / len(rdev_expected))


TP=(sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 1)))
TN=(sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 0)))
FP=(sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 1)))
FN=(sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 0)))

print('Accurancy= ' , (TP+TN)/(TP+FP+FN+TN))
print('Specifcity=' ,TN/(TN+FP))
print('Sensitivity=' ,TP/(TP+FN))
print('Matrix: ')
print('True Positives: ', TP)
print('True Negatives: ', TN)
print('False Positives: ', FP)
print('True Negatives: ', TN)
CM_dev=numpy.array([[TP, FP],[FN, TN]])
print(CM_dev)


print('writing to the expected file')
file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(lr_full.predict(rdev)):
    file.write(str(line)+'\n')

file.close()

print('-'*100) 


test = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
test = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

ol = lr_full.predict(test) 
file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(ol):
    file.write(str(line)+'\n') 

file.close()

print('-'*100)

print('plotting...')

sns.regplot(x=rdev.CO2, y=rdev_expected.y, logistic=True, y_jitter=.1)
plt.show()
