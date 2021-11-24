import pandas as pd

df = pd.read_csv('tiempos1.csv')
df['media_trie-distancia'] = df.groupby(['trie', 'distancia'])['tiempo'].transform('mean')
df['mediana_trie-distancia'] = df.groupby(['trie', 'distancia'])['tiempo'].transform('median')
df.to_csv('tiempos_stats.csv')
