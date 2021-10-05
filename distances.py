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
            

def levenshtein_restringed(begin, end, threshold):
    """
    Distancia de levenshtein restringida por un threshold.
    Si se supera, devolvemos el threshold + 1.
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
            if levmatrix[i,j] > threshold:
                #Hemos superado el threshold!!
                return threshold + 1
    return levmatrix[len(end), len(begin)]


"""
Damerau-Levenshtein
"""
def dp_restricted_damerau_iterative(x, y):
    """
    Distancia de Damerau-Levenshtein restringida entre dos cadenas (algoritmo iterativo bottom-up).
    
    """
    n = len(x)
    m = len(y)
    columnas0 = np.zeros(m + 1, dtype=int)
    columnas1 = np.zeros(m + 1, dtype=int)
    columnasI = np.zeros(m + 1, dtype=int) # Columna que utilizaremos para el intercambio (columna más anterior)
    #Inicializamos las columnas
    for i in range(m+1):
        columnas0[i] = i
    #Recorremos las columnas
    for i in range(1, n+1):
        # El primer elemento de la columna será el coste acumulado
        columnas1[0] = i
        #Recorremos las filas
        for j in range(1, m+1):
            if x[i-1] == y[j-1]: #Si es el mismo elemento el coste es 0 y cogemos el de la diagonal
                columnas1[j] = columnas0[j-1]
            elif i > 1 and j > 1 and x[i-2] == y[j-1] and x[i-1] == y[j-2]: #Si podemos aplicar intercambio
                columnas1[j] = min(columnas1[j-1], columnas0[j], columnas0[j-1], columnasI[j-2]) + 1
            else:
                columnas1[j] = min(columnas1[j-1], columnas0[j], columnas0[j-1]) + 1

        #Avanzamos las columnas (estados)
        columnasI = np.copy(columnas0)
        columnas0 = np.copy(columnas1)

    return columnas1[-1]


"""
Damerau-Levenshtein (algoritmo recursivo top-down con memorizacion)
"""
def dp_restricted_damerau_backwards(x, y):

 R = {}
    def dr(z,k):
        minn = None
        if (not z) and (not k): #Caso base
            R[z,k] = 0
            return 0
        if len(z) >= 1:
            if (z[:-1], k) not in R: R[z[:-1], k] = dr(z[:-1], k)
            minn = R[z[:-1], k] + 1
        if len(k) >= 1:
            if (z, k[:-1]) not in R: R[z, k[:-1]] = dr(z, k[:-1])
            if minn is None:
                minn = R[z, k[:-1]] + 1
            else:
                minn = min(minn, R[z, k[:-1]]+1)
        if len(z) >= 1 and len(k) >= 1:
            if len(z) > 1 and len(k) > 1:
                if z[-2] == k[-1] and z[-1] == k[-2]: #Intercambio
                    if (z[:-2], k[:-2]) not in R: R[z[:-2], k[:-2]] = dr(z[:-2], k[:-2])
                    minn = min(minn, R[z[:-2], k[:-2]] + 1)
            if z[-1] == k[-1]:
                if (z[:-1], k[:-1]) not in R: R[z[:-1], k[:-1]] = dr(z[:-1], k[:-1]) #Sustitucion sin coste
                minn = min(minn, R[z[:-1], k[:-1]])
            else:
                if (z[:-1], k[:-1]) not in R: R[z[:-1], k[:-1]] = dr(z[:-1], k[:-1])
                minn = min(minn, R[z[:-1], k[:-1]] + 1)
        return minn
    return dr(x,y)