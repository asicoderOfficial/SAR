# -*- coding: utf-8 -*-
import re
from distances import levenshtein_optimized_restinged, dp_restricted_damerau_iterative, dp_intermediate_damerau_threshold
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

        results = {} # diccionario termino:distancia
        lengword = len(term) #Agilizar dentro del bucle.
        MetDist = {}
        if (distance == "levensthein"):
            MetDist[distance] = levenshtein_optimized_restinged
        elif (distance == "restricted" ):
            MetDist[distance] = dp_restricted_damerau_iterative
        elif (distance == "intermediate"):
            MetDist[distance] = dp_intermediate_damerau_threshold
        
        for w in self.vocabulary:
            if (distance == "levensthein"):
                if(level_flat(term,w) <= threshold)
                    if (abs(len(w)-lengword) <= threshold):
                        MetDist[distance](term,w, threshold)
                        if (Dist <= threshold and Dist != None):
                        #No estoy seguro de si esto está ok, pero implemento diccionario para --> distancia:[word]}
                            if (Dist not in results):
                                results[Dist] = [w]
                            else:
                                results[Dist].append(w)
        return results

class TrieSpellSuggester(SpellSuggester):
    """
    Clase que implementa el método suggest para la búsqueda de términos y añade el trie
    """
    def __init__(self, vocab_file_path):
        super().__init__(vocab_file_path)
        self.trie = Trie(self.vocabulary)
    
if __name__ == "__main__":
    spellsuggester = TrieSpellSuggester("./corpora/quijote.txt")
    print(spellsuggester.suggest("alábese"))
    # cuidado, la salida es enorme print(suggester.trie)

    
