#!/usr/bin/python3
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model

p=pd.read_csv('/home/students/s407545/Desktop/cos/umz-template/zajecia1/zadanie3/train/train.tsv', sep='\t', names=['price','isNew','rooms','floor','location','sqrmeters'])

reg=linear_model.LinearRegression()

reg.fit(pd.DataFrame(p, columns=['sqrmeters','rooms']),p['price']) 

                                                                                     
import os
os.chdir('/home/students/s407545/Desktop/cos/umz-template/zajecia1/zadanie3/dev-0')

r=pd.read_csv('in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrmeters'])

x=pd.DataFrame(r, columns=['sqrmeters', 'rooms'])

y=reg.predict(x)


pd.DataFrame(y).to_csv('out.tsv', sep='\t', index=False, header=False)

os.chdir('/home/students/s407545/Desktop/cos/umz-template/zajecia1/zadanie3/test-A')

s=pd.read_csv('in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrmeters'])

x1=pd.DataFrame(s, columns=['sqrmeters', 'rooms'])

y1=reg.predict(x1)

pd.DataFrame(y1).to_csv('out.tsv', sep='\t', index=False,header= False)
    

sns.regplot(y=p['price'], x=p['sqrmeters'])

plt.show()

