import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('tiempos1.csv')
df['media_trie-distancia'] = df.groupby(['trie', 'distancia', 'tamanyo', 'threshold'])['tiempo'].transform('mean')
df['mediana_trie-distancia'] = df.groupby(['trie', 'distancia', 'tamanyo', 'threshold'])['tiempo'].transform('median')
df['distancia'] = df['distancia'].str.replace('intermediate','i')
df['distancia'] = df['distancia'].str.replace('restricted','d')
df['distancia'] = df['distancia'].str.replace('levenshtein','l')

df['distancia-trie'] = df['distancia'] + df['trie']
del df['distancia']
del df['trie']
df2 = pd.pivot_table(df, values='tiempo', index=['tamanyo', 'distancia-trie'], columns='threshold')
print(df2)
df2.to_csv('CSV_FINAL.csv')
#df.to_csv('tiempos_stats4.csv')
