# -*- coding: utf-8 -*-
import pandas as pd
import re
import distances as dist
from levels import level_flat
from trie import Trie
import collections
import time
import numpy as np


class SpellSuggester:

    """
    Clase que implementa el método suggest para la búsqueda de términos.
    """

    def __init__(self, vocab_file_path,vocab=None):
        """Método constructor de la clase SpellSuggester

        Construye una lista de términos únicos (vocabulario),
        que además se utiliza para crear un trie.

        Args:
            vocab_file (str): ruta del fichero de texto para cargar el vocabulario.

        """
        if vocab is not None:
            self.vocabulary = vocab
        else:
            self.vocabulary  = self.build_vocab(vocab_file_path, tokenizer=re.compile("\W+"))

    def build_vocab(self, vocab_file_path, tokenizer):
        """Método para crear el vocabulario.

        Se tokeniza por palabras el fichero de texto,
        se eliminan palabras duplicadas y se ordena
        lexicográficamente.

        Args:
            vocab_file (str): ruta del fichero de texto para cargar el vocabulario.
            tokenizer (re.Pattern): expresión regular para la tokenización.
        """
        with open(vocab_file_path, "r", encoding='utf-8') as fr:
            vocab = set(tokenizer.split(fr.read().lower()))
            vocab.discard('') # por si acaso
            return sorted(vocab)

    def suggest(self, term, distance="levenshtein", threshold=None):

        """Método para sugerir palabras similares siguiendo la tarea 3.

        A completar.

        Args:
            term (str): término de búsqueda.
            distance (str): algoritmo de búsqueda a utilizar
                {"levenshtein", "restricted", "intermediate"}.
            threshold (int): threshold para limitar la búsqueda
                puede utilizarse con los algoritmos de distancia mejorada de la tarea 2
                o filtrando la salida de las distancias de la tarea 2
        """
        if distance not in ["levenshtein", "restricted", "intermediate"]: raise ValueError("La distancia no es correcta")
        if threshold == None: threshold = 2**31
        results = {} # diccionario termino:distancia
        lengword = len(term) #Agilizar dentro del bucle.
        if (distance == "levenshtein"):
            DistUt= dist.dp_levenshtein_threshold
        elif (distance == "restricted" ):
            DistUt = dist.dp_restricted_damerau_threshold
        elif (distance == "intermediate"):
            DistUt = dist.dp_intermediate_damerau_threshold
        
        for w in self.vocabulary:
            if ((distance == "levenshtein" and level_flat(term,w) <= threshold) or distance != "levenshtein" ) \
                    and abs(len(w)-lengword) <= threshold:
                    Dist = DistUt(term,w, threshold)
                    if Dist <= threshold and Dist is not None and w not in results:
                    #Diccionario implementado para --> {word:distancia}
                        results[w] = Dist
        return results

class TrieSpellSuggester(SpellSuggester):
    def suggest(self, term, distance="levenshtein", threshold=2 ** 31):

        """Método para sugerir palabras similares siguiendo la tarea 3.

        A completar.

        Args:
            term (str): término de búsqueda.
            distance (str): algoritmo de búsqueda a utilizar
                {"levenshtein", "restricted", "intermediate"}.
            threshold (int): threshold para limitar la búsqueda
                puede utilizarse con los algoritmos de distancia mejorada de la tarea 2
                o filtrando la salida de las distancias de la tarea 2
        """
        if distance not in ["levenshtein", "restricted", "intermediate"]: raise ValueError(
            "La distancia no es correcta")
        results = {}  # diccionario termino:distancia
        lengword = len(term)  # Agilizar dentro del bucle.
        if distance == "levenshtein":
            DistUt = dist.dp_levenshtein_trie
        elif distance == "restricted":
            DistUt = dist.dp_restricted_damerau_trie
        elif distance == "intermediate":
            DistUt = dist.dp_intermediate_damerau_trie
        Dist = DistUt(term, self.trie, threshold)
        return Dist

    """
    Clase que implementa el método suggest para la búsqueda de términos y añade el trie
    """
    def __init__(self, vocab_file_path,vocab=None):
        super().__init__(vocab_file_path,vocab)
        self.trie = Trie(sorted(self.vocabulary))


if __name__ == "__main__":
    pal = ("casa", "senor", "constitución", "ancho", "savaedra", "quixot", "s3afg4ew")
    tam = (2500, 10000, 35000)
    th = (1, 2, 4, 5, 7)
    m = ("intermediate", "restricted", "levenshtein")

    #Listas con las configuraciones usadas, y el tiempo obtenido.
    palres = []
    tamres = []
    thres = []
    mres = []
    timeres = []
    trie = []

    vocab_file_path = "./quijote.txt"
    tokenizer = re.compile("\W+")
    with open(vocab_file_path, "r", encoding='utf-8') as fr:
        c = collections.Counter(tokenizer.split(fr.read().lower()))
    if '' in c:
        del c['']
    reversed_c = [(freq, word) for (word, freq) in c.items()]
    sorted_reversed = sorted(reversed_c, reverse=True)
    sorted_vocab = sorted([word for (freq, word) in sorted_reversed])
    for p in pal:
        print(p)
        for x in tam: 
            spellsuggester = SpellSuggester("./quijote.txt", sorted_vocab[:x])
            for z in m:
                for y in th:
                    tini = time.process_time()
                    spellsuggester.suggest(p, z, threshold = y)
                    t = time.process_time() - tini
                    #Obtenemos los datos a insertar luego en el DataFrame.
                    palres.append(p)
                    tamres.append(x)
                    thres.append(y)
                    mres.append(z)
                    timeres.append(t)
                    trie.append('No')
                    print('Trie: No   Tamaño: '+ str(x) + '   Threshold: ' + str(y) + '   Método: ' + z + '   :   ' + str(t))
            spellsuggester = TrieSpellSuggester("./quijote.txt", sorted_vocab[:x])
            for z in m:
                for y in th:
                    tini = time.process_time()
                    spellsuggester.suggest(p, z, threshold = y)
                    t = time.process_time() - tini
                    #Obtenemos los datos a insertar luego en el DataFrame.
                    palres.append(p)
                    tamres.append(x)
                    thres.append(y)
                    mres.append(z)
                    timeres.append(t)
                    trie.append('Yes')
                    print('Trie: Sí   Tamaño: '+ str(x) + '   Threshold: ' + str(y) + '   Método: ' + str(z)   + '   :    ' + str(t))

    #Creamos el DataFrame con los resultados de todas las ejecuciones.
    #Agrupamos por tam y m.
    df = pd.DataFrame({'palabra':palres, 'tamanyo':tamres, 'threshold':thres, 'distancia':mres, 'trie':trie, 'tiempo':timeres})
    df.to_csv('tiempos1.csv')
    
