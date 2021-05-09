import json
from nltk.stem.snowball import SnowballStemmer
import os
import re

class SAR_Project:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de noticias
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm + ranking de resultado

    Se deben completar los metodos que se indica.
    Se pueden aÃ±adir nuevas variables y nuevos metodos
    Los metodos que se aÃ±adan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [("title", True), ("date", False),
              ("keywords", True), ("article", True),
              ("summary", True)]
    
    
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes aÃ±adir mÃ¡s variables si las necesitas 

        """
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list.
                        # Si se hace la implementacion multifield, se pude hacer un segundo nivel de hashing de tal forma que:
                        # self.index['title'] seria el indice invertido del campo 'title'.
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de documentos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados. puede no utilizarse
        self.news = {} # hash de noticias --> clave entero (newid), valor: la info necesaria para diferenciar la noticia dentro de su fichero (doc_id y posiciÃ³n dentro del documento)
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()


    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_snippet(self, v):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v


    def set_stemming(self, v):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v


    def set_ranking(self, v):
        """

        Cambia el modo de ranking por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON RANKING DE NOTICIAS

        si self.use_ranking es True las consultas se mostraran ordenadas, no aplicable a la opcion -C

        """
        self.use_ranking = v




    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################


    def index_dir(self, root, **args):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root" e indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """

        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']
        if self.multifield is not None:
            new_self_index = {'article':{}, 'title':{}, 'summary':{}, 'keywords':{}, 'date':{}}
            self.index = new_self_index

        docid = 1
        for dir, subdirs, files in os.walk(root):
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    self.index_file(fullname, docid)
                    docid += 1

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        

    def fill_posting_list(self, new, index_key, docid):
        content = new['date'] if index_key == 'date' else self.tokenize(new[index_key])
        pos = 1
        for token in content:
            if token not in self.index[index_key]:
                self.index[index_key][token] = [(new['id'], docid, pos)]
            else:
                self.index[index_key][token] += [(new['id'], docid, pos)]
            pos += 1
        

    def index_file(self, filename, docid):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Indexa el contenido de un fichero.

        Para tokenizar la noticia se debe llamar a "self.tokenize"

        Dependiendo del valor de "self.multifield" y "self.positional" se debe ampliar el indexado.
        En estos casos, se recomienda crear nuevos metodos para hacer mas sencilla la implementacion

        input: "filename" es el nombre de un fichero en formato JSON Arrays (https://www.w3schools.com/js/js_json_arrays.asp).
                Una vez parseado con json.load tendremos una lista de diccionarios, cada diccionario corresponde a una noticia

        """

        with open(filename) as fh:
            jlist = json.load(fh)

        index_keys = self.index.keys()
        for new in jlist:
            for curr_index_key in index_keys:
                self.fill_posting_list(new, curr_index_key, docid)


        #
        # "jlist" es una lista con tantos elementos como noticias hay en el fichero,
        # cada noticia es un diccionario con los campos:
        #      "title", "date", "keywords", "article", "summary"
        #
        # En la version basica solo se debe indexar el contenido "article"
        #
        #
        #
        #################
        ### COMPLETAR ###
        #################



    def tokenize(self, text):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()



    def make_stemming(self):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING.

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        self.stemmer.stem(token) devuelve el stem del token

        """
        
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################


    
    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        """
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        








    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query, prev={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        if query is None or len(query) == 0:
            return []

        newquery = shunting_yard(infix_notation(query))  # Pasamos la query de a notación infija y la pasamos
                                                         # al algoritmo shunting_yard para obtener la postfija
        operadores = []
        newquery = list(map(lambda x: get_posting(format_terms(x)[2], format_terms(x)[0]) if isinstance(x, list)
                            else x, newquery))  # Formateamos los términos de la lista y obtenemos sus posting list.

        i = 0
        while len(newquery) != 1: #Vamos a analizar hasta que obtengamos 1 lista resultado.
            token = newquery[i]
            if isinstance(token, list):
                operadores.append(token) #Añadimos a operadores y seguimos analizando
                i += 1
            elif token == "NOT": #Si vemos una NOT, haremos reverse posting del ultimo elemento de la lista
                newquery[i - 1] = reverse_posting(operadores.pop())
                newquery.pop(i)
                operadores = [] #Volvemos a analizar
                i = 0
            elif token == "AND" or token == "OR": #Si vemos una and o una or cogemos los ultimos 2
                newquery[i] = and_posting(operadores.pop(), operadores.pop()) if token == "AND" \
                    else or_posting(operadores.pop(), operadores.pop())
                newquery.pop(i-2)
                newquery.pop(i-2)
                operadores = [] #Volvemos a analizar
                i = 0

        return newquery

    def shunting_yard(self, inputt):
        """
        Convierte una cadena en notación de postfijo (Fácilmente analizable) usando el algoritmo shunting_yard
        (con op unarios)

        :param inputt: consulta en notacion de infijo
        :return: consulta en notación de postfijo
        """
        stack = [] # Cola pperadores
        out = [] # Salida
        ops = ["OR", "AND", "NOT"] # Operadores
        precs = [1, 1, 2] # Precedencias, mayor valor mas precedencia
        for token in inputt:
            if isinstance(token, list): # Si es un operando
                out.append(token) # A la salida
            elif token in ops[:2]: # Si es un operador
                prec = precs[ops.index(token)]  # Precedencia del token
                while stack and stack[-1] not in ["("] and precs[ops.index(stack[-1])] >= prec:  # Precedencia de la cola mayor que el token
                    out.append(stack.pop()) # A la salida
                stack.append(token) # A la cola de operadores
            elif token == "NOT": # Si es un operador unario NOT, entonces no quitamos del stack.
                stack.append(token)
            elif token == "(":
                stack.append(token)
            elif token == ")":
                while stack and stack[-1] != "(":  # Buscamos el parentesis abierto
                    out.append(stack.pop())
                stack.pop()  # Descartamos el parentesis
        stack.reverse()
        for op in stack:  # Añadimos el resto de operadores
            out.append(op)

        return out

    def format_terms(self, terms):
        """
        Elimina los caracteres " y las palabras clave keywords:, title: etc para dejar los términos en una lista.
        Si contiene keywords, title... será incluido al principio de la lista.
        :param term: lista con los terminos
        :return:  lista en formato: [campo, si aplicar o no stemming (1 si 0 no), [terminos]]
        """
        fterms = []  # Variable para almacenar los terminos formateados
        multifield = [i[0] for i in fields] # Lista de campos
        fieldr = "article"  # Campo default
        if terms[0].find(":") != -1:  # Buscamos la primera aparaicion de : y asignamos el campo
                                      # a lo que haya a la izquierda
            field = terms[0][0:terms[0].find(":")] 
        else:
            field = 1  # Si no encontramos : no hay campos por tanto colocamos un valor arbitrario int
        if field in multifield:  # Si esta el campo en la lista
            fieldr = field  # El campo default pasa a ser el campo encontrado
            terms[0] = terms[0][len(field) + 1:]  # Cambiamos el termino para que ya no contenga "campo:"
            fterms.extend(terms[1:])  # Añadimos los elementos de
        if not fterms:  # Si no hay keywords
            fterms = [*terms]  # Copia
        if self.use_stemming and fterms[0][0] == "\"":  # Si usamos stemming y el primer caracter es "
            result = [fieldr, 0, fterms]  # Añadimos al resultado el campo, 0 para representar que no se usa stemming
                                          # en estos los terminos
        elif self.use_stemming:  # Si usamos stemming y no se ha cumplido lo anterior, usamos 1 para representar que
                                 # sí se usa stemming
            result = [fieldr, 1, fterms]
        else:
            result = [fieldr, 0, fterms] # En cualquier otro caso no se usa stemming
        fterms[0] = fterms[0][1:] if fterms[0][0] == "\"" else fterms[0]  # Eliminamos el primer caracter si es "
        fterms[-1] = fterms[-1][:-1] if fterms[-1][-1] == "\"" else fterms[-1]  # Eliminamos el ultimo caracter si es "

        return result

    def infix_notation(self, query):
        """
        Devuelve la consulta con notación de infijo.
        La consulta no debe tener los caracteres " y ( ) como terminos
        Útil para ser convertido a notación de postfijo.
        :param query: consulta a realizar
        :return: consulta en notacion de infijo NOT [Termino] OR [Termino]
        """
        query = query.replace("(", "( ")  # Separamos los parentesis
        query = query.replace(")", " )")
        query = query.split(" ")  # Obtenemos los tokens separados por espacios en una lista
        ops = []  # Lista para las operaciones
        term = []  # Lista para los terminos
        for i in query:
            if i not in ["NOT", "OR", "AND", "(", ")"]:  # Si no es un operador
                term.append(i)  # Lo añadimos a la lista de terminos
            elif term:  # Si es un operador y la lista de terminos no esta vacia
                ops.append(term)  # Añadimos a la lista de operaciones los terminos
                ops.append(i)  # Añadimos a la lista de operadores el operador
                term = []  # Reiniciamos la lista de terminos
            else:
                ops.append(i) # Si es un operador y esta vacia lo añadimos a la lista de operaciones
        if term: ops.append(term) # Si aún hay terminos los añadimos a
        return ops

    def get_posting(self, terms, field='article'):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming

        param:  "terms": lista con el primer elemento indicando si aplicamos o no stemming y sus terminos en otra
                field: campo a buscar
        return: posting list

        """
        stemming = terms[0]
        if stemming: # Si se requiere stemming del termino:
            return self.get_stemming(terms[1], field)
        elif len(terms)[1] > 1:
            return self.get_permuterm(terms[1], field)
        elif "*" in terms[1] or "?" in terms[1]:
            return self.get_positionals(terms[1], field)
        else:
            return self.index[field][terms[1][0]]



    def get_positionals_recursive(self, terms, new_pos, new_id, terms_pos, field, positional_list):
        """
        Metodo que recursivamente atraviesa el arbol de ngramas, obteniendo todos para todas las noticias.
        """
        for new in self.index[field][terms[terms_pos]]:
            # Caso base: llegamos a un nodo raiz.
            # Es el ultimo termino de la lista y sigue al anterior.
            if terms_pos == len(terms) - 1 and new[2] == new_pos - 1 and new[0] == new_id:
                return positional_list + [new]
            # Nos encontramos en el primer termino (nodos raiz).
            # Se generan tantos arboles como noticias que contienen el primer termino existen.
            elif terms_pos == 0:
                self.get_positionals_recursive(terms, new[2], new[0], terms_pos+1, field, [new])
            # Nodo intermedio, continuamos.
            elif new[0] == new_id and new[2] == new_pos - 1:
                self.get_positionals_recursive(terms, new[2], new[0], terms_pos+1, field, positional_list + [new])
        # No hay continuacion posible para la lista de terminos que buscamos en esta noticia. Devolvemos vacio.
        return []


    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list

        """
        positionals = []
        positionals += self.get_positionals_recursive(terms, 0, '', 0, field, positionals)
        return positionals


    def get_stemming(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING

        Devuelve la posting list asociada al stem de un termino.

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        stem = self.stemmer.stem(term)
        return [self.index[field][curr_term] for curr_term in self.index[field].keys() if stem in term]

    def get_permuterm(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################




    def reverse_posting(self, p):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los newid exceptos los contenidos en p

        """
        # Convertir la lista p a un set para mejorar el tiempo de busqueda, pasando de O(n) a O(1), siendo n, len(p).
        p = set(p)
        reversed_posting_list = []
        for k in self.index['article'].keys():
            for new in self.index['article'][k]:
                if new[0] not in p:
                    reversed_posting_list.append(new[0])
        
        return reversed_posting_list



    def and_posting(self, p1 ,p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos en p1 y p2

        """
        answer = []
        p1c = sorted(p1)
        p2c = sorted(p2)
        while p1c and p2c:
            if p1c[0] == p2c[0]:
                answer.append(p1c[0])
                p1c.pop(0)
                p2c.pop(0)
            elif p1c[0] < p2c[0]:
                p1c.pop(0)
            else:
                p2c.pop(0)
        return answer



    def or_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 o p2

        """
        # Como se indica en el boletin, seguimos la estructura de "merge".
        answer = []
        p1c = sorted(p1)
        p2c = sorted(p2)
        while p1c and p2c:
            if p1c[0] == p2c[0]:
                answer.append(p1c[0])
                p1c.pop(0)
                p2c.pop(0)
            elif p1c[0] < p2c[0]:
                answer.append(p1c[0])
                p1c.pop(0)
            else:
                answer.append(p2c[0])
                p2c.pop(0)
        while p1c:
            answer.append(p1c[0])
            p1c.pop(0)
        while p2c:
            answer.append(p2c[0])
            p2c.pop(0)
        return answer
        
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################


    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se propone por si os es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 y no en p2

        """

        
        pass
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################


    def solve_and_count(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T

        """
        result = self.solve_query(query)
        print("%s\t%d" % (query, len(result)))
        return len(result)  # para verificar los resultados (op: -T)


    def solve_and_show(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra informacion de las noticias recuperadas.
        Consideraciones:

        - En funcion del valor de "self.show_snippet" se mostrara una informacion u otra.
        - Si se implementa la opcion de ranking y en funcion del valor de self.use_ranking debera llamar a self.rank_result

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T
        
        """
        result = self.solve_query(query)
        if self.use_ranking:
            result = self.rank_result(result, query)   

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################




    def rank_result(self, result, query):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING

        Ordena los resultados de una query.

        param:  "result": lista de resultados sin ordenar
                "query": query, puede ser la query original, la query procesada o una lista de terminos


        return: la lista de resultados ordenada

        """

        pass
        
        ###################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE RANKING ##
        ###################################################
