import numpy as np


def crear_matriz_levenshtein(begin, end):
    """
    Dadas 2 strings, la inicial (begin) y en la que se quiere convertir (end),
    devolver la matriz necesaria para calcular la distancia de Levenshtein.
    Ejemplo -> para begin=benyam y end=ephrem, se genera:
    [[0. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0.]
     [2. 0. 0. 0. 0. 0. 0.]
     [3. 0. 0. 0. 0. 0. 0.]
     [4. 0. 0. 0. 0. 0. 0.]
     [5. 0. 0. 0. 0. 0. 0.]
     [6. 0. 0. 0. 0. 0. 0.]]
    """
    levmatrix = np.zeros((len(end)+1, len(begin)+1))
    levmatrix[:,0] = np.asarray(list(range(len(end)+1)))
    return levmatrix



