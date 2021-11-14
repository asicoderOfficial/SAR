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
    Distancia de levenshtein basica,
    sin optimizacion de memoria, O(n*m),
    siendo n=len(begin) y m=len(end).
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
            

def levenshtein_optimized(begin, end):
    """
    Distancia de levenshtein basica,
    con optimizacion de memoria, O(n),
    siendo n=len(begin).
    """
    a = list(range(len(begin)+1))
    b = [1] + list(range(len(begin)))
    for j in range(1, len(end)+1):
        for i in range(1,len(begin)+1):
            if begin[i-1] == end[i-1]:
                b[i] = a[i-1]
            else:
                b[i] = min(b[i-1],
                            a[i-1],
                            a[i])
        a = b
        b = [1] + list(range(len(begin)))
    return b[-1]


def levenshtein_optimized_restinged(begin, end, threshold):
    """
    Distancia de levenshtein basica,
    con optimizacion de memoria, O(n),
    siendo n=len(begin).
    """
    a = list(range(len(begin)+1))
    b = [1] + list(range(len(begin)))
    for j in range(1, len(end)+1):
        for i in range(1,len(begin)+1):
            if begin[i-1] == end[i-1]:
                b[i] = a[i-1]
            else:
                b[i] = min(b[i-1],
                            a[i-1],
                            a[i])
            if b[i] > threshold:
                return threshold + 1
        a = b
        b = [1] + list(range(len(begin)))
    return b[-1]


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
Version iterativa de la distancia damerau restringida  con threshold.

"""
def dp_restricted_damerau_threshold(x, y, threshold=2**30):
    n = len(x)
    m = len(y)

    # Creamos un vector inicializado a [0 .. m+1]
    col1 = np.fromiter((i for i in range(m + 1)), dtype=int)
    # Reservamos un vector para el cómputo de las siguientes columnas
    col2 = np.full(m + 1, threshold + 1)
    colI = np.full(m + 1, threshold + 1)

    for i in range(1, n + 1):
        for j in range(0, m + 1):
            if i - threshold <= j <= i + threshold:
                if j == 0:
                    col2[0] = i
                else:
                    if x[i - 1] == y[j - 1]:
                        col2[j] = col1[j - 1]
                    elif i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:
                        col2[j] = 1 + min(col2[j - 1],  # Insertar
                                         col1[j],  # Eliminar
                                         col1[j - 1],  # Reemplazar
                                         colI[j - 2])  # Intercambiar
                    else:
                        col2[j] = 1 + min(col2[j - 1],  # Insertar
                                         col1[j],  # Eliminar
                                         col1[j - 1])  # Reemplazar

        if np.min(col2) > threshold:
            return np.min(col2)

        colI = np.copy(col1)
        col1 = np.copy(col2)
        col2 = np.full(m + 1, threshold + 1)

    return col1[-1]
    return 0



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
Damerau-levenstein intermedia hacia atras sin threshold.

"""
def dp_intermediate_damerau_backwards(x,y):
    cte = 1  # Lo pone en el boletin.
    n = len(x)
    m = len(y)

    # Vector de [0 .. m+1]
    col1 = np.fromiter((x for x in range(m + 1)), dtype=int)
    # Vectores para las siguientes columnas
    col2 = np.zeros(m + 1, dtype=int)
    col0 = np.zeros(m + 1, dtype=int)
    col4 = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        col2[0] = i
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                col2[j] = col1[j - 1]
            elif i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:
                col2[j] = 1 + min(col2[j - 1],  # Insertar
                                 col1[j],  # Eliminar
                                 col1[j - 1],  # Reemplazar
                                 col0[j - 2])  # Intercambiar

            elif (i > 1 and j > 1 + cte) and (x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2 - cte]):
                # de ab a bda
                col2[j] = 1 + min(col2[j - 1],  # Insertar
                                 col1[j],  # Eliminar
                                 col1[j - 1],  # Reemplazar
                                 col0[j - 3] + 1)  # Intercambiar

            elif (i > 1 + cte and j > 1) and (x[i - 2 - cte] == y[j - 1] and x[i - 1] == y[j - 2]):
                # de bda a ab
                col2[j] = 1 + min(col2[j - 1],  # Insertar
                                 col1[j],  # Eliminar
                                 col1[j - 1],  # Reemplazar
                                 col4[j - 2] + 1)  # Intercambiar

            else:
                col2[j] = 1 + min(col2[j - 1],  # Insertar
                                 col1[j],  # Eliminar
                                 col1[j - 1])  # Reemplazar
        col4 = np.copy(col0)
        col0 = np.copy(col1)
        col1 = np.copy(col2)

    return col1[-1]

"""
Damerau-levenstein intermedia hacia atras con threshold.

"""
def dp_intermediate_damerau_threshold(x,y,threshold=2**30):
    cte = 1  # Lo pone en el boletin.
    n = len(x)
    m = len(y)

    # Vector de [0 .. m+1]
    col1 = np.fromiter((x for x in range(m + 1)), dtype=int)
    # Vectores para las siguientes columnas
    col2 = np.full(m + 1, threshold + 1)
    col0 = np.full(m + 1, threshold + 1)
    col4 = np.full(m + 1, threshold + 1)

    for i in range(1, n + 1):
        for j in range(0, m + 1):
            if i - threshold <= j <= i + threshold:
                if j == 0:
                    col2[0] = i
                else:
                    if x[i - 1] == y[j - 1]:
                        col2[j] = col1[j - 1]
                    elif i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:

                        col2[j] = 1 + min(col2[j - 1],  # Insertar
                                         col1[j],  # Eliminar
                                         col1[j - 1],  # Reemplazar
                                         col0[j - 2])  # Intercambiar

                    elif (i > 1 and j > 1 + cte) and (
                            x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2 - cte]):
                        # de ab a bda
                        col2[j] = 1 + min(col2[j - 1],  # Insertar
                                         col1[j],  # Eliminar
                                         col1[j - 1],  # Reemplazar
                                         col0[j - 3] + cte)  # Intercambiar

                    elif (i > 1 + cte and j > 1) and (
                            x[i - 2 - cte] == y[j - 1] and x[i - 1] == y[j - 2]):
                        # de bda a ab
                        col2[j] = 1 + min(col2[j - 1],  # Insertar
                                         col1[j],  # Eliminar
                                         col1[j - 1],  # Reemplazar
                                         col4[j - 2] + cte)  # Intercambiar

                    else:

                        col2[j] = 1 + min(col2[j - 1],  # Insertar
                                         col1[j],  # Eliminar
                                         col1[j - 1])  # Reemplazar

        if np.min(col2) > threshold:
            return np.min(col2)

        col4 = np.copy(col0)
        col0 = np.copy(col1)
        col1 = np.copy(col2)
        col2 = np.full(m + 1, threshold + 1)

    return col1[-1]


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

def dist_levenshtein_trie(str1, tr2, thres=2**31):
    """
    Distancia de levenshtein para la clase trie.
    """
    n = len(str1)
    m = tr2.get_num_states()

    dic = {}
    # Creamos una matriz donde guardar resultados
    dp = np.full((n+1, m), thres + 1)
    # Inicializamos la primera fila
    dp[0][0] = 0
    for x in range(1, m):
        dp[0][x] = dp[0][tr2.get_parent(x)] + 1
    for i in range(1, n + 1):
        for j in range(0, m):
            if j == 0:
                dp[i][0] = i
            else:
                if str1[i-1] == tr2.get_label(j):
                    dp[i][j] = dp[i-1][tr2.get_parent(j)]
                else:
                    dp[i][j] = 1 + min(dp[i][tr2.get_parent(j)],    # Insertar
                                    dp[i-1][j],                     # Eliminar
                                    dp[i-1][tr2.get_parent(j)])     # Reemplazar
        if np.min(dp[i]) > thres:
            break
    nodos = []
    for i in tr2.iter_children(0):
        nodos.append(i)
    while(nodos):
        nodo = nodos.pop()
        if tr2.is_final(nodo):
            if dp[-1][nodo] <= thres:
                dic[tr2.get_output(nodo)] = dp[-1][nodo]
        for i in tr2.iter_children(nodo):
            nodos.append(i)
    return dic


def damerau_general(begin, end):
    """
    Distancia de levenshtein basica,
    con optimizacion de memoria, O(n),
    siendo n=len(begin).
    """
    d = {}
    lenbegin = len(begin)
    lenend = len(end)
    for i in xrange(-1,lenbegin+1):
        d[(i,-1)] = i+1
    for j in xrange(-1,lenend+1):
        d[(-1,j)] = j+1

    for i in xrange(lenbegin):
        for j in xrange(lenend):
            if begin[i] == end[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and begin[i]==end[j-1] and begin[i-1] == end[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenbegin-1,lenend-1]
