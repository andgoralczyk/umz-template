
#!/usr/bin/python3
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from  sklearn import linear_model 


p= pd.read_csv('/home/students/s407545/Desktop/cos/umz-template/zajecia1/zadanie2/train/train.tsv', sep = '\t', names= ['price', 'isNew', 'rooms', 'floor', 'location', 'sqrmeters'])

reg=linear_model.LinearRegression()

p.shape
p.price

reg.fit(pd.DataFrame(p, columns=['sqrmeters']), p['price'])
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

x=p['sqrmeters']
x=x.values.reshape(-1,1)

os.chdir('/home/students/s407545/Desktop/cos/umz-template/zajecia1/zadanie2/dev-0')

r=pd.read_csv('in.tsv', sep='\t', names=['isNew', 'rooms', 'floor', 'location', 'sqrmeters'])

x=r['sqrmeters']

x=x.values.reshape(-1,1)

y=reg.predict(x)

pd.DataFrame(y).to_csv('out.tsv', sep='\t', index= False, header=False)
    
s=pd.read_csv('/home/students/s407545/Desktop/cos/umz-template/zajecia1/zadanie2/test-A/in.tsv', sep='\t', names=['isNew',' rooms', 'floor', 'location', 'sqrmeters'])
x1=s['sqrmeters']
x1=x1.values.reshape(-1,1)
y1=reg.predict(x1)

 pd.DataFrame(y1).to_csv('out.tsv', sep='\t', index=False, header= False)
 


sns.regplot(y=p['price'], x=p['sqrmeters']); plt.show()



