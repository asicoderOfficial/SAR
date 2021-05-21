import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import time
import bisect
import math
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
        self.weight_noti = {} #Hash para pesados usado en rank_result.
        self.perterms = {} #Hash para permuterm. La clave es el permuterm y el valor es la lista de términos de ese permuterm. 
        self.stemterms = {} #Hash para stemming. La clave es el stem y el valor es la lista de términos asociados a ese stem.
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()
        #Variables adicionales para las partes opcionales.
        self.multifield = False
        self.positional = False
        self.permuterm = False

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


    def set_multifield(self, v):
        """

        Cambia el modo de campos por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON MULTIPLES CAMPOS

        si self.multifield es True se emplean todos los campos de la variable global fields.

        """
        self.multifield = v


    def set_positional(self, v):
        """

        Cambia el modo de positional por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON BUSQUEDA POSICIONAL DE NOTICIAS

        si self.positional es True, se realiza una busqueda posicional de un conjunto de terminos.

        """
        self.positional = v


    def set_permuterm(self, v):
        """

        Cambia el modo de permuterm por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON BUSQUEDA CON PERMUTERM DE NOTICIAS

        si self.permuterm es True, se realiza una busqueda por permuterm.

        """
        self.permuterm = v



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
        #Comprobamos las opciones seleccionadas y anadimos las correspondientes.
        self.set_multifield(args.multifield)
        self.set_positional(args.positional)
        self.set_stemming(args.stem)
        self.set_ranking(args.rank)
        self.set_showall(args.all)
        self.set_snippet(args.snippet)
        self.set_permuterm(args.permuterm)

        self.index = {'article':{}, 'title':{}, 'summary':{}, 'keywords':{}, 'date':{}} if self.multifield else {'article':{}}
        
        docid = 1
        for dir, _, files in os.walk(root):
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    self.index_file(fullname)
                    self.docs[docid] = filename
                    docid += 1

        if self.use_stemming:
            self.make_stemming()
        self.make_permuterm()
        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        

    def fill_posting_list(self, new, field):
        """
        Metodo para rellenar la posting list correspondiente a cada termino.
        El formato del indice es el siguiente:
        self.index = {field:{token:{new_id,[position1,position2,…]}}}
        """
        #No tokenizamos y solamente almacenamos la id de la noticia correspondiente a la fecha dada.
        if field == 'date':
            if new['date'] not in self.index[field]:
                self.index[field][new['date']] = {new['id']:[]}
            else:
                self.index[field][new['date']][new['id']] = []
        #Tokenizamos y guardamos las posiciones de cada token, empezando por 1.
        else:
            
            content = self.tokenize(new[field])
            pos = 1
            for token in content:
                if token not in self.index[field]:
                    self.index[field][token] = {new['id']:[pos]}
                else:
                    if new['id'] not in self.index[field][token]:
                        self.index[field][token][new['id']] = [pos]
                    else:
                        self.index[field][token][new['id']] += [pos]
                pos += 1
        

    def index_file(self, filename):
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

        fields = [f[0] for f in self.fields] if len(self.index.keys()) > 1 else ['article']
        for new in jlist:
            for field in fields:
                self.fill_posting_list(new, field)
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
        for i in self.index.keys():
            self.sindex[i] = {}
            for j in self.index[i].keys():
                self.sindex[i][self.stemmer.stem(j)] = j




    
    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        """
        """
        for i in self.index.keys():
            if i not in self.ptindex:
                self.ptindex[i] = []
            for j in self.index[i].keys():
                term = j
                j = j+'$'
                for k in range(len(j)):
                    tupla = (j, term)
                    self.ptindex[i].append(tupla)
                    
                    if j not in self.ptindex[i]:
                        self.ptindex[i][j] = [term]
                    else:
                        self.ptindex[i][j] = self.ptindex[i][j].append(term)
                        
                    aux = j[1:]
                    j = aux + j[0]
                    
            self.ptindex[i].sort(key=lambda x: x[0])
        """
        for i in self.index.keys():
            self.ptindex[i] = {}
            for j in self.index[i].keys():
                term = j
                self.ptindex[i][term] = []
                j = j + '$'
                for k in range(len(j)):
                    self.ptindex[i][term].append(j)
                    aux = j[1:]
                    j = aux + j[0]
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """ 
        print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')

        print('Number of indexed days: '+ str(len(self.docs)))

        print('----------------------------------------')
    
        print('Number of indexed news: ' + str(len(self.news)))

        print('----------------------------------------')

        print('TOKENS:')

        for a in self.index.keys():
            print("\t# of tokens in '{}': {}".format(a, len(self.index[a])))
        print('----------------------------------------')
        if (self.permuterm):
            print('PERMUTERMS:')
            for b in self.ptindex.keys():
                suma = 0
                for c in self.ptindex[b].keys():
                    suma += len(self.ptindex[b][c])
                print("\t# of tokens in '{}': {}".format(b, suma))
            print('----------------------------------------')
        if (self.use_stemming):
            print('STEMS:')
            for c in self.sindex.keys():
                print("\t# of tokens in '{}': {}".format(c, len(self.sindex[c])))
            print('----------------------------------------')
        if (self.positional):
            print('Positional queries are allowed')
        else:
            print(print('Positional queries are NOT allowed'))

        print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')

        








    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################

    def mapquery(self, query):
        if isinstance(query, list):  # Si son terminos
            ft = self.format_terms(query)  # Los formateamos
            terms = self.get_posting(ft[1], ft[0])  # Obtenemos sus posting list
            lista = sorted(list(set(i for i in terms)))  # Nos quedamos con las noticias únicas
            return lista

        else:
            return query



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

        newquery = self.shunting_yard(self.infix_notation(query))  # Pasamos la query de a notación infija y la pasamos
                                                         # al algoritmo shunting_yard para obtener la postfija
        operandos = []
        newquery = list(map(self.mapquery, newquery))  # Formateamos los términos de la lista y obtenemos sus posting list.

        i = 0
        while len(newquery) != 1: #Vamos a analizar hasta que obtengamos 1 lista resultado.
            token = newquery[i]
            if isinstance(token, list):
                operandos.append(token) #Añadimos a operandos y seguimos analizando
                i += 1
            elif token == "NOT": #Si vemos una NOT, haremos reverse posting del ultimo elemento de la lista
                newquery[i - 1] = self.reverse_posting(operandos.pop())
                newquery.pop(i)
                operandos = [] #Volvemos a analizar
                i = 0
            elif token == "AND" or token == "OR": #Si vemos una and o una or cogemos los ultimos 2
                newquery[i] = self.and_posting(operandos.pop(), operandos.pop()) if token == "AND" \
                    else self.or_posting(operandos.pop(), operandos.pop())
                newquery.pop(i-2)
                newquery.pop(i-2)
                operandos = [] #Volvemos a analizar
                i = 0
        print(len(newquery[0]))
        return newquery[0]


    def shunting_yard(self, inputt):
        """
        Convierte una cadena en notación de postfijo (Fácilmente analizable) usando el algoritmo shunting_yard
        (con op unarios)

        :param inputt: consulta en notacion de infijo
        :return: consulta en notación de postfijo
        """
        stack = [] # Pila pperadores
        out = [] # Salida
        ops = ["OR", "AND", "NOT"]  # Operadores
        precs = [1, 1, 2]  # Precedencias, mayor valor mas precedencia
        for token in inputt:
            if isinstance(token, list):  # Si es un operando
                out.append(token)  # A la salida
            elif token in ops[:2]:  # Si es un operador
                prec = precs[ops.index(token)]  # Precedencia del token
                while stack and stack[-1] not in ["("] and precs[ops.index(stack[-1])] >= prec:  # Precedencia de la cola mayor que el token
                    out.append(stack.pop()) # A la salida
                stack.append(token)  # A la pila de operadores
            elif token == "NOT":  # Si es un operador unario NOT, entonces no quitamos del stack.
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
        :return:  lista en formato: [campo, [terminos]]
        """
        fterms = []  # Variable para almacenar los terminos formateados
        multifield = [i[0] for i in SAR_Project.fields]  # Lista de campos
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
        fterms[0] = fterms[0][1:] if fterms[0][0] == "\'" else fterms[0]  # Eliminamos el primer caracter si es "
        fterms[-1] = fterms[-1][:-1] if fterms[-1][-1] == "\'" else fterms[-1]  # Eliminamos el ultimo caracter si es "

        return [fieldr, [x.lower() for x in fterms]]

    def make_posting_list(self, p):
        """
        Crea una lista con el elemento y sus posiciones. [Key,pos]
        """
        pass

    def infix_notation(self, query):
        """
        Devuelve la consulta con notación de infijo.
        La consulta no debe tener los caracteres " y ( ) como terminos
        Útil para ser convertido a notación de postf
        ....ppijo.
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

        param:  "terms": lista con los términos
                field: campo a buscar
        return: posting list

        """
        if any(d for d in terms if any(ds in d for ds in ["*", "?"])):# Usamos la función any porque solo requiere que aparezca 1 elemento
            return self.get_permuterm(terms, field)
        elif len(terms) > 1:
            pos = self.get_positionals(terms, field)
            return pos
        elif self.use_stemming:  # Si se requiere stemming del termino:
            return self.get_stemming(terms[0], field)
        else:
            return self.index[field][terms[0]].keys() if terms[0] in self.index[field] else []



    def get_positionals_recursive(self, terms, new_pos, new_id, terms_pos, field, positional_list):
        """
        Metodo que recursivamente atraviesa el arbol de ngramas, obteniendo todos para todas las noticias.
        """
        """
        result = []
        if terms[terms_pos] not in self.index[field]:
            return positional_list
        for new in self.index[field][terms[terms_pos]]:
            # Caso base: llegamos a un nodo raiz.
            # Es el ultimo termino de la lista y sigue al anterior.
            if terms_pos == len(terms) - 1 and new[2] == new_pos + 1 and new[0] == new_id:
                return positional_list + [new]
            # Nos encontramos en el primer termino (nodos raiz).
            # Se generan tantos arboles como noticias que contienen el primer termino existen.
            elif terms_pos == 0:
                result.extend(self.get_positionals_recursive(terms, new[2], new[0], terms_pos+1, field, [new]))
            # Nodo intermedio, continuamos.
            elif new[0] == new_id and new[2] == new_pos + 1:
                result.extend(self.get_positionals_recursive(terms, new[2], new[0], terms_pos+1, field, positional_list + [new]))
        # No hay continuacion posible para la lista de terminos que buscamos en esta noticia. Devolvemos vacio
        return result
        """



    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list

        """
        #Obtenemos todas las noticias en las que aparece cada termino, con su correspondiente posting list.
        #Filtramos, nos quedamos con las noticias comunes a todos los terminos en el campo seleccionado.
        common_news = self.index[field][terms[0]]
        for curr_term in terms[1:]:
            common_news = self.and_posting(common_news, self.index[field][curr_term])
        terms_postings = [{k: v for k, v in self.index[field][curr_term].items() if k in common_news} for curr_term in terms]
        
        #Recorremos las noticias, encontrando en ellas para el campo dado, los ngramas dados.
        for curr_new in common_news:
            #Buscamos los ngramas que se encuentran en la actual noticia.
            #Lo visualizamos como una matriz, donde las filas son las posting list, y las columnas las posiciones.
            #Ejemplo: [[1,3,8],[2,4]]; Aqui, habria 2 parejas, [1,2] y [3,4].
            curr_term_row = curr_term_col = next_term_col = 0
            next_term_row = 1

            
        #Filtramos, quedando las noticias 
        positionals = []
        positionals = self.get_positionals_recursive(terms, 0, '', 0, field, positionals)
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
        tokens = self.sindex[stem]
        return [self.index[field][curr_term][t] for t in tokens]


    def get_permuterm(self, term, field='article'):
        term = term[0] + '$'
        while term[-1] != '*' and term[-1] != '?':
            term = term[-1] + term[:-1]

        result = []
        if term[-1] == '*':
            term = term[:-1]
            for key in self.ptindex[field]:
                permuterms = self.ptindex[field][key]
                i = 0
                end = False
                while i < len(permuterms) and not end:
                    if term in permuterms[i]:
                        result = self.or_posting(result, sorted(self.index[field][key].keys()))
                        end = True
                    i = i+1
        else:
            term = term[:-1]
            for key in self.ptindex[field]:
                if len(key) == len(term):
                    permuterms = self.ptindex[field][key]
                    i=0
                    end = False
                    while i < len(permuterms) and not end:
                        if term in permuterms[i]:
                            result = self.or_posting(result, sorted(self.index[field][key].keys()))
                            end = True
                        i = i+1
        return result



    def reverse_posting(self, p):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los newid exceptos los contenidos en p

        """
        # Convertir la lista p a un set para mejorar el tiempo de busqueda.
        p1 = set(p.keys()) if isinstance(p, dict) else set(p)
        reversed_posting_list = set()
        for k in self.index['article'].keys():
            for new in self.index['article'][k]:
                if new not in p1:
                    reversed_posting_list.add(new)
        
        return list(reversed_posting_list)


    def and_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular (diccionario o lista con id de las noticias)

        return: posting list con los newid incluidos en p1 y p2

        """
        answer = []
        p1c = list(p1.keys()) if isinstance(p1, dict) else [*p1]
        p2c = list(p2.keys()) if isinstance(p2, dict) else [*p2]

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

        param:  "p1", "p2": posting lists sobre las que calcular (diccionario o lista con id de las noticias)


        return: posting list con los newid incluidos de p1 o p2

        """
        # Como se indica en el boletin, seguimos la estructura de "merge".
        answer = []
        p1c = list(p1.keys()) if isinstance(p1, dict) else [*p1]
        p2c = list(p2.keys()) if isinstance(p2, dict) else [*p2]
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
        #Variables auxiliares:
        noticiasprocesadas = 1 #Contador usado más adelante para indicar el número de noticia procesada.
        #Resolvemos la query y en caso de que se aplique ranking aplicamos para las noticias resultantes.

        result = self.solve_query(query)
        if result is None:
            return 0
        #Si la consulta usa ranking, aplicamos para el resultado de la query.
        if self.use_ranking:
            result = self.rank_result(result, query)   

        print("Query: " + str(query))
        print("Number of results: " + str(len(result)))

        q = query.lower()
        #Iterar sobre cada noticia resultante de la query...
        for ID in result:
            if not self.use_ranking:
                rank = 0
            else: rank = round(self.weight_noti[ID],4)

            IDDocumento = self.news[ID]['doc_id']
            PosicionDocumento = self.news[ID]['posición']
            PathDocumento = self.docs['doc_id']

            #Leer el documento que contiene la noticia que queremos obtener la información
            with open(PathDocumento) as fl:
                list = json.load(fl)

                noticiait = list[PosicionDocumento]

                #Ahora obtenemos los datos requeridos de la noticia (Keywords, Id de noticia(ya presente en el iterador), la fecha y el título de esta)
                keywords_noticia = noticiait['keywords']
                titulo_noticia = noticiait['title']
                fecha_noticia = noticiait['date']
                noticiasprocesadas += 1

                #Distinguimos entre si se ha usado la opción -N o no, y según ello mostramos la información de la noticia por pantalla
                if not self.show_snippet:
                    print("#{}      ({})  ({})  ({})   {}      ({})".format(noticiasprocesadas,rank,ID,fecha_noticia,titulo_noticia,keywords_noticia))
                else: print("#{} \nScore:{} \nNewsID: {}  \nDate: {}   \nTitle: {}   \nNKeywords: {}".format(noticiasprocesadas,rank,ID,fecha_noticia,titulo_noticia,keywords_noticia))































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

        #Términos de la query
        t = {}

        for a in query.keys():

            term = a[0]

            field = a[1]

            #Ahora cabe buscar si se ha usado stemming o si se trata de consulta permuterm
            if (self.use_stemming):
                tt = self.stemterms[self.stemmer.stem(term)] 
                for t1 in tt:
                    t.setdefault(t1,field)
            
            #En caso de consulta permuterm

            elif ("?" in term or "*" in term):
                permuterms = self.get_permuterm(term,field)

                for elemento in permuterms:
                    aux = self.perterms[elemento]
                    for  t2 in aux:
                        t.setdefault(t2,field)
            
            else:
                t.setdefault(term,field)
            
        #Declaramos lista de pesados para las noticias
            Masquepesados = []

        #Por cada noticia en el resultado...
        for noticia in result:

            #Por cada término y campo (Pues hay que calcular una por una para todas la noticias) usando las calculadas anteriormente.
            for tpla in t:

                term = tpla[0]

                field = tpla[1]

            #Para el pesado usaremos el explicado en el tema 4 de teoría.

                #1er paso: Pesado del término
                tf = 0
                ft = self.weight[field][term].get(noticia,0)

                if ft > 0:
                    tf = math.log10(ft)
                
                #2do paso: Función global idf
                df = len(self.weight[field][term])
                idf = math.log10(self.news/df)

                #3er paso: Acumular el pesado de cada término para obtener el total de la noticia
                pesado_noticia = pesado_noticia + (tf*idf)

            #Añadir los pesados sobre cada noticia
            self.weight_noti[noticia] = self.weight_noti.get(noticia,0) + pesado_noticia
            Masquepesados.append(pesado_noticia)
            
            #Para finalizar, antes de devolver la lista, se ordena según el ranking de noticias.
            res = [i for _,i in sorted(zip(Masquepesados,result), reverse = True)]

            pass
        
        ###################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE RANKING ##
        ###################################################

def and_posting(p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos en p1 y p2

        """
        answer = []
        p1c = list(p1.keys()) if isinstance(p1, dict) else [*p1]
        p2c = list(p2.keys()) if isinstance(p2, dict) else [*p2]
        while p1c and p2c:
            if p1c[0][1] == p2c[0][1]:
                answer.append(p1c[0])
                p1c.pop(0)
                p2c.pop(0)
            elif p1c[0][1] < p2c[0][1]:
                p1c.pop(0)
            else:
                p2c.pop(0)
        return answer
