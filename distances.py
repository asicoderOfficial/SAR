import numpy as np


def crear_matriz_levenshtein(begin, end):
    """
    Dadas 2 strings, la inicial (begin) y en la que se quiere convertir (end),
    devolver la matriz necesaria para calcular la distancia de Levenshtein.
    Ejemplo -> para begin=benyam y end=ephrem, se genera:
    [[0. 1. 2. 3. 4. 5. 6.]
     [1. 0. 0. 0. 0. 0. 0.]
     [2. 0. 0. 0. 0. 0. 0.]
     [3. 0. 0. 0. 0. 0. 0.]
     [4. 0. 0. 0. 0. 0. 0.]
     [5. 0. 0. 0. 0. 0. 0.]
     [6. 0. 0. 0. 0. 0. 0.]]
    """
    levmatrix = np.zeros((len(end)+1, len(begin)+1))
    levmatrix[:,0] = np.asarray(list(range(len(end)+1)))
    levmatrix[0,:] = np.asarray(list(range(len(begin)+1)))
    return levmatrix


"""
LEVENSHTEIN
"""

def levenshtein_basic(begin, end):
    """
    Distancia de levenshtein basica.
    """
    levmatrix = crear_matriz_levenshtein(begin, end)
    for i in range(1,len(begin)+1):
        for j in range(1,len(end)+1):
            if begin[i-1] == end[j-1]:
                #Caracteres iguales.
                levmatrix[i,j] = levmatrix[i-1, j-1]
            else:
                #No es igual, cogemos el minimo y sumamos 1.
                levmatrix[i,j] = min(levmatrix[i-1, j-1], min(levmatrix[i-1, j], levmatrix[i, j-1])) + 1
    return levmatrix[len(end), len(begin)]
            

