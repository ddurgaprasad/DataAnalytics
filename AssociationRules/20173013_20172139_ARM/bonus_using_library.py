

corpus=['I1, I2, I3, I4', 'I1, I2, I4', 'I1, I2', 'I2, I3, I4', 'I2, I3', 'I3, I4', 'I2, I4']

corpus = [item.split(',') for item in corpus]
                     
#Import all basic libray
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import time
# Task1 : Compute Frequent Item Set using  mlxtend.frequent_patterns
tran_encoder = TransactionEncoder()
from mlxtend.frequent_patterns import fpgrowth
tran_encode_arr = tran_encoder.fit(corpus).transform(corpus)
df = pd.DataFrame(tran_encode_arr, columns=tran_encoder.columns_)
print(df)
start_time = time.time()
frequent = fpgrowth(df, min_support=0.001, use_colnames=True)
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))
# Task 2&3: Find closed/max frequent itemset using frequent itemset found in task1
_support = frequent.support.unique()#all unique support count#Dictionay storing itemset with same support count key
print(_support)
fredic = {}
for i in range(len(_support)):
    inset = list(frequent.loc[frequent.support ==_support[i]]['itemsets'])
    fredic[_support[i]] = inset#Dictionay storing itemset with  support count <= key
fredic2 = {}
for i in range(len(_support)):
    inset2 = list(frequent.loc[frequent.support<=_support[i]]['itemsets'])
    fredic2[_support[i]] = inset2#Find Closed frequent itemset
start_time = time.time()
cl = []
for index, row in frequent.iterrows():
    isclose = True
    cli = row['itemsets']
    cls = row['support']
    checkset = fredic[cls]
    for i in checkset:
        if (cli!=i):
            if(frozenset.issubset(cli,i)):
                isclose = False
                break
    
    if(isclose):
        cl.append(row['itemsets'])
print('Time to find Close frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))  
    
#Find Max frequent itemset
start_time = time.time()
ml = []
for index, row in frequent.iterrows():
    isclose = True
    cli = row['itemsets']
    cls = row['support']
    checkset = fredic2[cls]
    for i in checkset:
        if (cli!=i):
            if(frozenset.issubset(cli,i)):
                isclose = False
                break
    
    if(isclose):
        ml.append(row['itemsets'])
print('Time to find Max frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))