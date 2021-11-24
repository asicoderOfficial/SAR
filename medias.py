import pandas as pd

df = pd.read_csv('tiempos1.csv')
df['media_trie-distancia'] = df.groupby(['trie', 'distancia', 'tamanyo'])['tiempo'].transform('mean')
df['mediana_trie-distancia'] = df.groupby(['trie', 'distancia', 'tamanyo'])['tiempo'].transform('median')
df['distancia'] = df['distancia'].str.replace('intermediate','i')
df['distancia'] = df['distancia'].str.replace('restricted','d')
df['distancia'] = df['distancia'].str.replace('levenshtein','l')
print(df)
df.to_csv('tiempos_stats3.csv')
