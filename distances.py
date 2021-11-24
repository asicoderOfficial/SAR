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
    levmatrix = np.transpose(levmatrix)
    for i in range(1,len(begin)+1):
        for j in range(1,len(end)+1):
            if begin[i-1] == end[j-1]:
                #Caracteres iguales.
                levmatrix[i,j] = levmatrix[i-1, j-1]
            else:
                #No es igual, cogemos el minimo y sumamos 1.
                levmatrix[i,j] = min(levmatrix[i-1, j-1], min(levmatrix[i-1, j], levmatrix[i, j-1])) + 1
    return levmatrix[len(begin), len(end)]
            

def levenshtein_optimized(begin, end):
    """
     Distancia de levenshtein basica,
     con optimizacion de memoria, O(n),
     siendo n=len(begin).
     begin = benyam -> i -> a -> b
     end = ephre -> j
     """
    a = list(range(len(end) + 1))
    for j in range(1, len(begin) + 1):
        b = [j] + [0] * len(end)
        for i in range(1, len(end) + 1):
            if begin[j - 1] == end[i - 1]:
                b[i] = a[i - 1]
            else:
                b[i] = min(b[i - 1],
                           a[i - 1],
                           a[i]) + 1
        a = b
    return b[-1]

def dp_levenshtein_threshold(begin, end, threshold=2**30):
    """
     Distancia de levenshtein basica,
     con optimizacion de memoria, O(n),
     siendo n=len(begin).
     begin = benyam -> i -> a -> b
     end = ephre -> j
     """
    a = list(range(len(end) + 1))
    for j in range(1, len(begin) + 1):
        b = [j] + [0] * len(end)
        for i in range(1, len(end) + 1):
            if begin[j - 1] == end[i - 1]:
                b[i] = a[i - 1]
            else:
                b[i] = min(b[i - 1],
                           a[i - 1],
                           a[i]) + 1
        if min(b) > threshold:
            return threshold+1
        a = b
    return b[-1]


def levenshtein_restringed(begin, end, threshold):
    """
    Distancia de levenshtein restringida por un threshold.
    Si se supera, devolvemos el threshold + 1.
    """

    levmatrix = crear_matriz_levenshtein(begin, end)
    levmatrix = np.transpose(levmatrix)
    for i in range(1,len(begin)+1):
        for j in range(1,len(end)+1):
            if begin[i-1] == end[j-1]:
                #Caracteres iguales.
                levmatrix[i,j] = levmatrix[i-1, j-1]
            else:
                #No es igual, cogemos el minimo y sumamos 1.
                levmatrix[i,j] = min(levmatrix[i-1, j-1],
                                     min(levmatrix[i-1, j],
                                         levmatrix[i, j-1])) + 1
            if levmatrix[i,j] > threshold:
                #Hemos superado el threshold!!
                return threshold + 1
    return levmatrix[len(begin), len(end)]

"""
Version iterativa de la distancia damerau restringida  con threshold.

"""
def dp_restricted_damerau_threshold(x, y, threshold=2**30):
    n = len(x)
    m = len(y)

    # Creamos un vector inicializado a [0 .. m+1]
    columnas1 = np.fromiter((i for i in range(m + 1)), dtype=int)
    # Reservamos un vector para el cómputo de las siguientes columnas
    columnas2 = np.full(m + 1, threshold + 1) #Ahora en vez de inicializarlo a 0 lo inicializamos a threshold+1
    columnasI = np.full(m + 1, threshold + 1)

    for i in range(1, n + 1): # Desplazamos la matriz n veces.
        for j in range(0, m + 1):
            if i - threshold <= j and j <= i + threshold:
                if j == 0:
                    columnas2[0] = i
                else:
                    if x[i - 1] == y[j - 1]: #Si es el mismo elemento el coste es 0 y cogemos el de la diagonal
                        columnas2[j] = columnas1[j - 1]
                    elif i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]: #Si podemos aplicar intercambio
                        columnas2[j] = 1 + min(columnas2[j - 1], columnas1[j], columnas1[j - 1], columnasI[j - 2])
                    else:
                        columnas2[j] = 1 + min(columnas2[j - 1], columnas1[j], columnas1[j - 1])

        if np.min(columnas2) > threshold:
            return np.min(columnas2)

        columnasI = np.copy(columnas1)
        columnas1 = np.copy(columnas2)
        columnas2 = np.full(m + 1, threshold + 1)

    return columnas1[-1]



"""
Damerau-Levenshtein restringidahacia atras sin threshold.
"""
def dp_restricted_damerau_backwards(x, y):
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
    colI = np.zeros(m + 1, dtype=int)
    col4 = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        col2[0] = i
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                col2[j] = col1[j - 1]
            elif i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:
                col2[j] = 1 + min(col2[j - 1], col1[j],  col1[j - 1], colI[j - 2])
            elif (i > 1 and j > 1 + cte) and (x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2 - cte]):
                # de ab a bda
                col2[j] = 1 + min(col2[j - 1], col1[j], col1[j - 1], colI[j - 3] + 1)

            elif (i > 1 + cte and j > 1) and (x[i - 2 - cte] == y[j - 1] and x[i - 1] == y[j - 2]):
                # de bda a ab
                col2[j] = 1 + min(col2[j - 1], col1[j], col1[j - 1], col4[j - 2] + 1)
            else:
                col2[j] = 1 + min(col2[j - 1], col1[j], col1[j - 1])
        col4 = np.copy(colI)
        colI = np.copy(col1)
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
    colI = np.full(m + 1, threshold + 1)
    col4 = np.full(m + 1, threshold + 1)

    for i in range(1, n + 1):
        for j in range(0, m + 1):
            if i - threshold <= j and j <= i + threshold:
                if j == 0:
                    col2[0] = i
                else:
                    if x[i - 1] == y[j - 1]:
                        col2[j] = col1[j - 1]
                    elif i > 1 and j > 1 and x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2]:

                        col2[j] = 1 + min(col2[j - 1], col1[j], col1[j - 1], colI[j - 2])
                    elif (i > 1 and j > 1 + cte) and \
                        (x[i - 2] == y[j - 1] and x[i - 1] == y[j - 2 - cte]):
                        # de ab a bda
                        col2[j] = 1 + min(col2[j - 1], col1[j], col1[j - 1], colI[j - 3] + cte)

                    elif (i > 1 + cte and j > 1) and \
                            (x[i - 2 - cte] == y[j - 1] and x[i - 1] == y[j - 2]):
                        # de bda a ab
                        col2[j] = 1 + min(col2[j - 1], col1[j],  col1[j - 1], col4[j - 2] + cte)

                    else:

                        col2[j] = 1 + min(col2[j - 1],  col1[j], col1[j - 1])

        if np.min(col2) > threshold:
            return np.min(col2)
        col4 = np.copy(colI)
        colI = np.copy(col1)
        col1 = np.copy(col2)
        col2 = np.full(m + 1, threshold + 1)

    return col1[-1]



def dp_intermediate_damerau_trie(x, trie, threshold = 2**30):

    nodes = np.fromiter((i for i in range(0,trie.get_num_states())),dtype=int)
    col0 = np.zeros(trie.get_num_states(), dtype=int)
    col1 = np.zeros(trie.get_num_states(), dtype=int)
    col2 = np.zeros(trie.get_num_states(), dtype=int)
    col3 = np.zeros(trie.get_num_states(), dtype=int)
    col0[0] = col1[0] + 1
    for j in nodes[1:]:
        col1[j] = col1[trie.get_parent(j)] + 1


    for j in nodes[1:]:
        col0[j] = min(col0[trie.get_parent(j)] + 1, col1[j] + 1, col1[trie.get_parent(j)] + (trie.get_label(j) != x[0]))

    for i in range(1, len(x)):
        col3, col2, col1, col0 = col2, col1, col0, col3
        col0[0] = col1[0] + 1
        for j in nodes[1:]:
            aux = min(col0[trie.get_parent(j)] + 1, col1[j] + 1, col1[trie.get_parent(j)] + (trie.get_label(j) != x[i]))

            if (i > 1 and trie.get_parent(trie.get_parent(j)) != -1 and x[i] == trie.get_label(trie.get_parent(j))
                    and x[i - 2] == trie.get_label(j)
                    and col3[trie.get_parent(trie.get_parent(j))] + 2 < aux):
                aux = col3[trie.get_parent(trie.get_parent(j))] + 2

            if (trie.get_parent(trie.get_parent(trie.get_parent(j))) != -1 and trie.get_label(j) == x[i - 1]
                    and trie.get_label(trie.get_parent(trie.get_parent(j))) == x[i]
                    and col2[trie.get_parent(trie.get_parent(trie.get_parent(j)))] + 2 < aux):
                aux = col2[trie.get_parent(trie.get_parent(trie.get_parent(j)))] + 2

            if (x[i] == trie.get_label(trie.get_parent(j)) and x[i - 1] == trie.get_label(j) and
                    trie.get_parent(trie.get_parent(j)) != -1 and col2[trie.get_parent(trie.get_parent(j))] <= aux):
                aux = col2[trie.get_parent(trie.get_parent(j))] + 1
            col0[j] = aux
        if min(col3) > threshold:
            break
    dic = {}
    for node in list(range(0, trie.get_num_states())):
        if trie.is_final(node) and col0[node] <= threshold:
            dic[trie.get_output(node)] = col0[node]
    return [(k, v) for k, v in dic.items()]

def dp_restricted_damerau_trie(x, tri, threshold = 2**30):
    n = len(x)
    m = tri.get_num_states()
    dic = {}
    # Creamos un vector inicializado a [0 .. m+1]
    #columnas1 = np.fromiter((i for i in range(m + 1)), dtype=int)
    columnas1 = np.zeros(m+1,dtype=int)
    for i in range(1,m):
        columnas1[i] = columnas1[tri.get_parent(i)] + 1
    # Reservamos un vector para el cómputo de las siguientes columnas
    columnas2 = np.full(m + 1, 0)  # Ahora en vez de inicializarlo a 0 lo inicializamos a threshold+1
    columnasI = np.full(m + 1, 0)

    for i in range(1, n + 1):  # Desplazamos la matriz n veces.
        for j in range(0, m + 1):
                if j == 0:
                    columnas2[0] = i
                else:
                    if x[i - 1] == tri.get_label(j):  # Si es el mismo elemento el coste es 0 y cogemos el de la diagonal
                        columnas2[j] = columnas1[tri.get_parent(j)]
                    elif i > 1 and j > 1 and x[i - 2] == tri.get_label(j) and \
                            x[i - 1] == tri.get_label(tri.get_parent(j)):  # Si podemos aplicar intercambio
                        columnas2[j] = 1 + min(columnas2[tri.get_parent(j)], columnas1[j], columnas1[tri.get_parent(j)],
                                               columnasI[tri.get_parent(tri.get_parent(j))])
                    else:
                        columnas2[j] = 1 + min(columnas2[tri.get_parent(j)], columnas1[j], columnas1[tri.get_parent(j)])
        if np.min(columnas2) > threshold:
            break

        columnasI = np.copy(columnas1)
        columnas1 = np.copy(columnas2)
        columnas2 = np.full(m + 1, threshold + 1)
    nodos = []
    for i in tri.iter_children(0):
        nodos.append(i)
    while (nodos):
        nodo = nodos.pop()
        if tri.is_final(nodo):
            if columnas1[nodo] <= threshold:
                dic[tri.get_output(nodo)] = columnas1[nodo]
        for i in tri.iter_children(nodo):
            nodos.append(i)
    if dic == {}: return []
    return [(k, v) for k, v in dic.items()]

def dp_levenshtein_trie(str1, tr2, thres=2**31):
    """
    Distancia de levenshtein para la clase trie.
    """
    n = len(str1)
    m = tr2.get_num_states()

    dic = {}
    # Creamos una matriz donde guardar resultados
    dp = np.full((n + 1, m), thres + 1)
    # Inicializamos la primera fila
    dp[0][0] = 0
    for x in range(1, m):
        dp[0][x] = dp[0][tr2.get_parent(x)] + 1
    for i in range(1, n + 1):
        for j in range(0, m):
            if j == 0:
                dp[i][0] = i
            else:
                if str1[i - 1] == tr2.get_label(j):
                    dp[i][j] = dp[i - 1][tr2.get_parent(j)]
                else:
                    dp[i][j] = 1 + min(dp[i][tr2.get_parent(j)],  # Insertar
                                       dp[i - 1][j],  # Eliminar
                                       dp[i - 1][tr2.get_parent(j)])  # Reemplazar
        if np.min(dp[i]) > thres:
            break
    nodos = []
    for i in tr2.iter_children(0):
        nodos.append(i)
    while (nodos):
        nodo = nodos.pop()
        if tr2.is_final(nodo):
            if dp[-1][nodo] <= thres:
                dic[tr2.get_output(nodo)] = dp[-1][nodo]
        for i in tr2.iter_children(nodo):
            nodos.append(i)
    return [(k, v) for k, v in dic.items()]

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
"""
print(levenshtein_basic('benyam', 'ephre'))
print(levenshtein_optimized('benyam', 'ephre'))
print(levenshtein_optimized_restringed('benyam', 'ephre', 10))
print(levenshtein_restringed('benyam', 'ephre', 10))
"""