# -*- coding: utf-8 -*-
import re
from distances import levenshtein_optimized_restinged, dp_restricted_damerau_threshold, dp_intermediate_damerau_threshold
from levels import level_flat
from trie import Trie

class SpellSuggester:

    """
    Clase que implementa el método suggest para la búsqueda de términos.
    """

    def __init__(self, vocab_file_path):
        """Método constructor de la clase SpellSuggester

        Construye una lista de términos únicos (vocabulario),
        que además se utiliza para crear un trie.

        Args:
            vocab_file (str): ruta del fichero de texto para cargar el vocabulario.

        """

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
        assert distance in ["levenshtein", "restricted", "intermediate"]
        if threshold == None: threshold = 2**31
        results = {} # diccionario termino:distancia
        lengword = len(term) #Agilizar dentro del bucle.
        MetDist = {}
        if (distance == "levensthein"):
            MetDist[distance] = levenshtein_optimized_restinged
        elif (distance == "restricted" ):
            MetDist[distance] = dp_restricted_damerau_threshold
        elif (distance == "intermediate"):
            MetDist[distance] = dp_intermediate_damerau_threshold
        
        for w in self.vocabulary:
            if (distance == "levensthein"):
                if(level_flat(term,w) <= threshold)
                    if (abs(len(w)-lengword) <= threshold):
                        MetDist[distance](term,w, threshold)
                        if (Dist <= threshold and Dist != None):
                        #No estoy seguro de si esto está ok, pero implemento diccionario para --> word:distancia}
                            if (w not in results):
                                results[w] = Dist
        return results

class TrieSpellSuggester(SpellSuggester):
    def suggest(self, term, distance="levenshtein", threshold=None):
        if distance == "levenshtein":
            results = {}
            if threshold == None: 
                threshold = 2**31
            a = self.trie.get_num_states()
            b = len(term)
            M1 = np.zeros(a)
            M2 = np.zeros(a)
            for i in range(1,a):
                V1[i]= M1[self.trie.get_parent(i)] + 1

            for col in range(1,b + 1):
                V2[0]=col
                for fil in range(1,a) :
                    cost = not term[col-1] == self.trie.get_label(fil)
                    V2[fil] = min(V1[fil] + 1,
                                V2[self.trie.get_parent(fil)] + 1,
                                V1[self.trie.get_parent(fil)] + cost)
                if min(M2) > threshold:
                     return {}
                M1, M2 = M2, M1

            for i in range(n):
                if self.trie.is_final(i):
                    if V1[i] <= threshold: results[self.trie.get_output(i)] = V1[i]
            return results
        else: return super().suggest(term, distance, threshold)

    """
    Clase que implementa el método suggest para la búsqueda de términos y añade el trie
    """
    def __init__(self, vocab_file_path):
        super().__init__(vocab_file_path)
        self.trie = Trie(self.vocabulary)
    
if __name__ == "__main__":
    pal = ("casa", "senor", "jabón", "constitución", "ancho", "savaedra", "vicios", "quixot", "s3afg4ew")
    tam = (2500,5000, 10000, 15000, 20000, 25000, 30000, 35000)
    th = (1, 2, 3, 4, 5, 7)
    m = ("intermediate", "restricted", "levenshtein")
    for p in pal:
        print(p)
        for x in tam: 
            spellsuggester = SpellSuggester("./corpora/quijote.txt", x)    
            for z in m:
                for y in th:
                    tini = time.process_time()
                    spellsuggester.suggest(p, z, threshold = y)
                    t = time.process_time() - tini
                    print('Trie: No   Tamaño: '+ str(x) + '   Threshold: ' + str(y) + '   Método: ' + z + '   :   ' + str(t))
            spellsuggester = TrieSpellSuggester("./corpora/quijote.txt", x)
            for y in th:
                tini = time.process_time()
                spellsuggester.suggest(p, "levenshtein", threshold = y)
                t = time.process_time() - tini
                print('Trie: Sí   Tamaño: '+ str(x) + '   Threshold: ' + str(y) + '   Método: levenshtein   :   ' + str(t))

    
