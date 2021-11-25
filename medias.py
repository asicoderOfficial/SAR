import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join

df = pd.read_csv('tiempos1.csv')
df['media_trie-distancia'] = df.groupby(['trie', 'distancia', 'tamanyo', 'threshold'])['tiempo'].transform('mean')
df['mediana_trie-distancia'] = df.groupby(['trie', 'distancia', 'tamanyo', 'threshold'])['tiempo'].transform('median')
df['distancia'] = df['distancia'].str.replace('intermediate','i')
df['distancia'] = df['distancia'].str.replace('restricted','d')
df['distancia'] = df['distancia'].str.replace('levenshtein','l')

df['distancia-trie'] = df['distancia'] + df['trie']
del df['distancia']
del df['trie']
print(df)
df2 = pd.pivot_table(df, values='tiempo', index=['tamanyo', 'distancia-trie'], columns='threshold')
print(df2)
df2.to_csv('CSV_FINAL.csv')
#df.to_csv('tiempos_stats4.csv')
###############################################

"""
mypath = 'resultadosConMod/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
df = pd.DataFrame()
trie = []
for f in onlyfiles:
    curr_df = pd.read_csv(mypath + f, delimiter='\t', header=None)
    if 'trie' in f:
        trie += ['Yes'] * curr_df.shape[0]
    else:
        trie += ['No'] * curr_df.shape[0]
    df = pd.concat([df, pd.read_csv(mypath + f, delimiter='\t', header=None)])
df['media'] = df[4].str.split(' ').str[1]
df['media'] = pd.to_numeric(df['media'])
df['trie'] = trie
df = df.rename(columns={0:'distancia', 1:'palabra', 2:'threshold', 3:'tamanyo'})
df['media_trie-distancia'] = df.groupby(['trie', 'distancia', 'tamanyo', 'threshold'])['media'].transform('mean')
df['distancia'] = df['distancia'].str.replace('intermediate','i')
df['distancia'] = df['distancia'].str.replace('restricted','d')
df['distancia'] = df['distancia'].str.replace('levenshtein','l')
df['distancia-trie'] = df['distancia'] + df['trie']
del df['distancia']
del df['trie']
df2 = pd.pivot_table(df, values='media_trie-distancia', index=['tamanyo', 'distancia-trie'], columns='threshold')
print(df2)
df2.to_csv('CSV_FINAL_PRUEBA.csv')
"""
