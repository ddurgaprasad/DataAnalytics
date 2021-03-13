# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:06:09 2020

@author: E442282
"""

import pandas as pd
import numpy as np

df=pd.read_csv('earthquakes1.csv',encoding = "utf-8")
print(df.head())

missing_values_count = df.isnull().sum()
print(missing_values_count)

df.drop('ORIGIN_TIME', axis=1,inplace=True)

missing_values_count = df.isnull().sum()
print(missing_values_count)


df.replace(r'^\s+$', np.nan, regex=True,inplace=True)
df.replace(r'^\s*$', np.nan, regex=True,inplace=True)
df['MONTH'].replace('', np.nan, inplace=True)

df=df[~df['MONTH'].isnull()]
df=df[~df['DAY'].isnull()]

missing_values_count = df.isnull().sum()
print(missing_values_count)

df['DEPTH_KM'].fillna(df['DEPTH_KM'].mode()[0], inplace=True)
missing_values_count = df.isnull().sum()
print(missing_values_count)

df['MAGNITUDE']=df['MAGNITUDE'].astype(float)
df['MAGNITUDE'].fillna(df['MAGNITUDE'].mean(), inplace=True)
missing_values_count = df.isnull().sum()
print(missing_values_count)

searchfor = ['°', 'E','W','N','S']
df=  df[~df.LAT.str.contains('|'.join(searchfor),na=False)]
df=  df[~df.LONG.str.contains('|'.join(searchfor),na=False)]
missing_values_count = df.isnull().sum()
print(missing_values_count)


df['LAT']=df['LAT'].astype(float)
df['LONG']=df['LONG'].astype(float)

df=df[df['LAT']>0 ] 
df=df[df['LAT']<90] 
missing_values_count = df.isnull().sum()
print(missing_values_count)
print(df.dtypes)


df['YEAR']=df['YEAR'].astype(int)
df['DAY']=df['DAY'].astype(int)
# df['MONTH']=df['MONTH'].astype(int)

df.to_csv('earthquakes11.csv', encoding='utf-8', index=False)
df=pd.read_csv('earthquakes11.csv',encoding = "utf-8")

df['YEAR']=df['YEAR'].astype(int)
df['DAY']=df['DAY'].astype(int)
df['MONTH']=df['MONTH'].astype(int)

missing_values_count = df.isnull().sum()
print(missing_values_count)
print(df.dtypes)


df=df[(df['YEAR'] >= 1767) ]
df=df[(df['MONTH'] >= 1) & (df['MONTH'] <= 12)]
df=df[(df['DAY'] >= 1) & (df['DAY'] <= 31)]


missing_values_count = df.isnull().sum()
print(missing_values_count)
print(df.dtypes)


#df['DT']=pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

#df.drop(['YEAR', 'MONTH', 'DAY'], axis=1,inplace=True)

# https://civildigital.com/classification-earthquakes/
Threshold=4.9

df['IsEarthQuake']=np.where(df['MAGNITUDE']> Threshold, 1, 0)


print(df.head())
print(df.groupby('IsEarthQuake').count())
df.to_csv('earthquakes_final.csv', encoding='utf-8', index=False)















