import numpy as np

def occurences_dict(s):
    """
    Funcion auxiliar, devuelve el numero de ocurrencias en un string para cada caracter.
    """
    d = {}
    for c in s:
        if c not in d:
            d[c] = 1
        else:
            d[c] += 1
    return d


def level_flat(begin, end):
    """
    Calcula la cota optimista de la tarea 3, para 2 cadenas dadas sobre la distancia de levenshtein.
    """
    #Caracteres que aparecen en una o ambas cadenas.
    chars = list(set(begin).union(end))
    #Indices de dichos caracteres en la lista resultante.
    indexes = {chars[c] : c for c in range(len(chars))}
    #Creacion de las listas de frecuencias.
    begin_occ = [0] * len(chars)
    end_occ = [0] * len(chars)
    b = occurences_dict(begin)
    e = occurences_dict(end)
    #Rellenamos con las frecuencias.
    for c in chars:
        if c in b:
            begin_occ[indexes[c]] = b[c]
        if c in e:
            end_occ[indexes[c]] = e[c]
    #Conversion a array de numpy.
    begin_occ = np.array(begin_occ)
    end_occ = np.array(end_occ)
    #Resta.
    substraction = begin_occ - end_occ
    return max(np.sum(substraction[substraction > 0]), np.sum(substraction[substraction < 0]))

