
import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
from spellsuggest import SpellSuggester
from spellsuggest import TrieSpellSuggester
import math


class SAR_Project:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de noticias
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm + ranking de resultado

    Se deben completar los metodos que se indica.
    Se pueden aadir nuevas variables y nuevos metodos
    Los metodos que se anyadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [("title", True), ("date", False),
              ("keywords", True), ("article", True),
              ("summary", True)]
    
    #Identificador de noticias empleado como clave del diccionario self.news.
    newid = 1
    #Identificador de documento empleado como clave en self.docs y como primer valor de tupla en self.news.
    docid = 1
    #Posicion de la noticia en el documento.
    newpos = 1
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes aadir ms variables si las necesitas

        """
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list.
                        # Si se hace la implementacion multifield, se pude hacer un segundo nivel de hashing de tal forma que:
                        # self.index['title'] seria el indice invertido del campo 'title'.
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de documentos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados. puede no utilizarse
        self.news = {} # hash de noticias --> clave entero (newid), valor: la info necesaria para diferenciar la noticia dentro de su fichero (doc_id y posicin dentro del documento)
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.weight_noti = {} #Hash para pesados usado en rank_result.
        self.perterms = {} #Hash para permuterm. La clave es el permuterm y el valor es la lista de trminos de ese permuterm. 
        self.stemterms = {} #Hash para stemming. La clave es el stem y el valor es la lista de trminos asociados a ese stem.
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()
        #Variables adicionales para las partes opcionales.
        self.multifield = False # valor por defecto, se cambia con self.set_multifield()
        self.positional = False # valor por defecto, se cambia con self.set_positional()
        self.permuterm = False # valor por defecto, se cambia con self.set_permuterm()
        self.busq = None  # valor por defecto, cuando no se usa la busqueda, se cambia con self.set_busq()
        self.threshold = 3  # valor por defecto se cambia con self.set_threshold
        self.trie = None # valor por defecto se cambia con self.make_trie
        self.use_trie = False # Valor por defecto, indica si en la consulta se busca con trie o no.
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

    def set_trie(self, v):
        """

        Indica si se usa trie o no para las distancias de edicion.
        input BOOLEANO
        """
        self.use_trie = v

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

    def set_busq(self, v):
        """
        Establece el modo de busqueda aproximada de cadenas

        """
        if v not in [None, 'levenshtein', 'intermediate', 'restricted']: raise ValueError("La distancia no es correcta")
        self.busq = v
    def set_threshold(self, v):
        """

        Establece el threshold para la busqueda aproximada de cadenas

        """
        self.threshold = v



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
        if 'multifield' in args:
            self.set_multifield(args['multifield'])
        if 'positional' in args:
            self.set_positional(args['positional'])
        if 'stem' in args:
            self.set_stemming(args['stem'])
        if 'rank' in args:
            self.set_ranking(args['rank'])
        if 'all' in args:
            self.set_showall(args['all'])
        if 'snippet' in args:
            self.set_snippet(args['snippet'])
        if 'permuterm' in args:
            self.set_permuterm(args['permuterm'])

        self.index = {'article':{}, 'title':{}, 'summary':{}, 'keywords':{}, 'date':{}} if self.multifield else {'article':{}}
        
        for dir, _, files in os.walk(root):
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    self.index_file(fullname)
                    self.docs[self.docid] = fullname
                    self.docid += 1

        if self.use_stemming:
            self.make_stemming()
        if self.permuterm:
            self.make_permuterm()
        if args['suggest']:
            self.make_trie()
            self.busq = "levenshtein"


    def fill_posting_list(self, new, field):
        """
        Metodo para rellenar la posting list correspondiente a cada termino.
        El formato del indice es el siguiente:
        self.index = {field:{token:{new_id,[position1,position2,]}}}
        """
        #No tokenizamos y solamente almacenamos la id de la noticia correspondiente a la fecha dada.
        if field == 'date':
            if new['date'] not in self.index[field]:
                self.index[field][new['date']] = {self.newid:[]}
            else:
                self.index[field][new['date']][self.newid] = []
        #Tokenizamos y guardamos las posiciones de cada token, empezando por 1.
        else:
            content = self.tokenize(new[field])
            pos = 1
            for token in content:
                if token not in self.index[field]:
                    self.index[field][token] = {self.newid:[pos]}
                else:
                    if self.newid not in self.index[field][token]:
                        self.index[field][token][self.newid] = [pos]
                    else:
                        self.index[field][token][self.newid] += [pos]
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
        newpos = 1
        for new in jlist:
            #Rellenamos el diccionario de noticias.
            self.news[self.newid] = (self.docid, newpos)
            for field in fields:
                self.fill_posting_list(new, field)
            self.newid += 1
            newpos += 1
        #
        # "jlist" es una lista con tantos elementos como noticias hay en el fichero,
        # cada noticia es un diccionario con los campos:
        #      "title", "date", "keywords", "article", "summary"
        #
        # En la version basica solo se debe indexar el contenido "article"
        #
        #
        #



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
                if self.stemmer.stem(j) not in self.sindex[i]:
                    self.sindex[i][self.stemmer.stem(j)] = [j]
                else:
                    self.sindex[i][self.stemmer.stem(j)] += [j]



    
    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        """
        for i in self.index.keys():
            self.ptindex[i] = {}
            for j in self.index[i].keys():
                term = j
                self.ptindex[i][term] = []
                j = j + '$'
                for _ in range(len(j)):
                    self.ptindex[i][term].append(j)
                    aux = j[1:]
                    j = aux + j[0]



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
            print('Positional queries are NOT allowed')

        print('-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')

    def make_trie(self):
        """
        NECESARIO PARA BUSQUEDA APROXIMADA

        """
        self.trie = TrieSpellSuggester("",list(self.index['article'].keys()))


    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################

    def mapquery(self, query):
        if isinstance(query, list):  # Si son terminos
            ft = self.format_terms(query)  # Los formateamos
            terms = self.get_posting(ft[1], ft[0])  # Obtenemos sus posting list
            lista = sorted(list(set(i for i in terms)))  # Nos quedamos con las noticias nicas
            return lista
        else:
            return query

    def cleanquery(self, query):
        if isinstance(query, list):  # Si son terminos
            ft = self.format_terms(query)  # Los formateamos
            return ft

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

        newquery = self.shunting_yard(self.infix_notation(query))  # Pasamos la query de notacin infija y la pasamos
                                                         # al algoritmo shunting_yard para obtener la postfija
        operandos = []
        newquery = list(map(self.mapquery, newquery))  # Formateamos los trminos de la lista y obtenemos sus posting list.

        i = 0
        while len(newquery) != 1: #Vamos a analizar hasta que obtengamos 1 lista resultado.
            token = newquery[i]
            if isinstance(token, list):
                operandos.append(token) #Aadimos a operandos y seguimos analizando
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
        Convierte una cadena en notacin de postfijo (Fcilmente analizable) usando el algoritmo shunting_yard
        (con op unarios)

        :param inputt: consulta en notacion de infijo
        :return: consulta en notacin de postfijo
        """
        stack = [] # Pila operadores
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
        for op in stack:  # Aadimos el resto de operadores
            out.append(op)

        return out

    def format_terms(self, terms):
        """
        Elimina los caracteres " y las palabras clave keywords:, title: etc para dejar los trminos en una lista.
        Si contiene keywords, title... ser incluido al principio de la lista.
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
            fterms.extend(terms[1:])  # Aadimos los elementos de
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
        Devuelve la consulta con notacin de infijo.
        La consulta no debe tener los caracteres " y ( ) como terminos
        til para ser convertido a notacin de postf
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
                term.append(i)  # Lo aadimos a la lista de terminos
            elif term:  # Si es un operador y la lista de terminos no esta vacia
                ops.append(term)  # Aadimos a la lista de operaciones los terminos
                ops.append(i)  # Aadimos a la lista de operadores el operador
                term = []  # Reiniciamos la lista de terminos
            else:
                ops.append(i) # Si es un operador y esta vacia lo aadimos a la lista de operaciones
        if term: ops.append(term) # Si an hay terminos los aadimos a
        return ops


    def get_posting(self, terms, field='article'):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming

        param:  "terms": lista con los trminos
                field: campo a buscar
        return: posting list

        """
        if any(d for d in terms if any(ds in d for ds in ["*", "?"])):  # Usamos la funcin any porque solo requiere que aparezca 1 elemento
            return self.get_permuterm(terms, field)
        elif len(terms) > 1:
            pos = self.get_positionals(terms, field)
            if not pos:  # -> ALT ESTO NO ERA REQUERIDO PERO ES UNA IDEA QUE HEMOS TENIDO:
                """
                En los posicionales (aunque no fuera requerido) hemos implementado algo similar. 
                Si no encontramos la palabra en el vocabulario del ??ndice entonces vamos variando las palabras de la 
                lista de posicionales con palabras sugeridas del m??todo suggest hasta que encontremos alguna noticia.
                 Si a??n as?? no encontramos noticias (ser??a realmente un caso extremo), por ejemplo buscamos ???una casa??? 
                 y nos hemos equivocado y hemos escrito ???una caso??? y no encontramos ninguna noticia, entonces vamos a
                  hacer permutaciones un poco ???a fuerza bruta??? con palabras sugeridas del m??todo suggest, por ejemplo 
                  una y un son parecidas-> un caso. Y caso y casa son parecidas -> una casa.
                    
                    Por desgracia, me di cuenta m??s tarde de que nosotros no hab??amos implementado la busqueda 
                    posicional, de todas formas dejo esto aqu??, que no me parece que sea incorrecto.
                    Atentamente, el grupo de SAR.

                """

                pos = []   # por si acaso
                if self.use_trie:
                    if self.trie is None:
                        raise ValueError("Error: se indic busqueda con trie pero el indice no se cre con -G")
                    spg = self.trie
                else:
                    spg = SpellSuggester("", list(self.index['article'].keys()))
                termsaux = []
                for t in terms: #SI una palabra no se encuentra en los articulos de las noticias (posiblemente sea un error)
                    if t not in list(self.index['article'].keys()):
                        palabras = [w for w,_ in spg.suggest(t, self.busq, self.threshold)]
                        for p in palabras:
                            termsaux = terms
                            termsaux[terms.index(t)] = p
                            pos += self.get_posting(termsaux, field)
                print(pos)
                if pos: #Si hemos encontrado noticias
                    return pos
                # Fuerza bruta (Si aun asi no hemos encontrado noticias, por ejemplo nos hemos equivocado en caso por
                # "casa" y la busqueda posicional no da resultados.
                for t in terms:
                    palabras = [w for w,_ in spg.suggest(t, self.busq, self.threshold)]
                    for p in palabras:
                        termsaux = terms
                        termsaux[terms.index(t)] = p
                        pos += self.get_posting(termsaux, field)
                        if pos:
                            break
                    if pos:
                        break

            return pos
        elif self.use_stemming:  # Si se requiere stemming del termino:
            return self.get_stemming(terms[0], field)
        elif terms[0] in self.index[field]:
            return list(self.index[field][terms[0]].keys())
        else:  # Si no hemos encontrado ningun resultado en las posting lists -> ALT
            if self.busq is None:
                return []
            if self.use_trie:
                if self.trie is None:
                    raise ValueError("Error: se indic busqueda con trie pero el indice no se cre con -G")
                spg = self.trie
            else:
                spg = SpellSuggester("", list(self.index['article'].keys()))

            postinglists = []
            palabras = [w for w,_ in spg.suggest(terms[0], self.busq, self.threshold)]
            for t in palabras:
                postinglists += self.get_posting([t],field)

            return postinglists





    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list

        """
        return []


    def get_stemming(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING

        Devuelve la posting list asociada al stem de un termino.

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        #Realizamos el stem del termino dado.
        stem = self.stemmer.stem(term)
        #Buscamos los tokens asociados a dicho stem.
        tokens = self.sindex[field][stem] if stem in self.sindex[field] else []
        #Buscamos las noticias que contienen dicho token, y devolvemos la lista de las mismas.
        return [b for t in tokens for b in list(self.index[field][t].keys())]


    def get_permuterm(self, term, field='article'):
        """
            Busca en el diccionario de permuterms las palabras que se ajustan a la wildcard.
            Se hace una bsqueda exhaustiva.
        :param term: termino a buscar (con wildcard)
        :param field: campo donde buscar
        :return: posting list con newid
        """
        term = term[0] + '$' #Indicamos el final del termino y luego rotamos la palabra hasta que la wildcard quede al final
        while term[-1] != '*' and term[-1] != '?':
            term = term[-1] + term[:-1]

        result = []
        if term[-1] == '*':
            term = term[:-1]
            for key in self.ptindex[field]:
                permuterms = self.ptindex[field][key]
                i = 0
                end = False
                while i < len(permuterms) and not end: #Recorremos toda la lista de terminos y sus rotaciones para encontrar las posting list de los trminos.
                    if term in permuterms[i]:
                        result = self.or_posting(result, sorted(self.index[field][key].keys()))
                        end = True
                    i = i+1
        else:
            term = term[:-1]
            for key in self.ptindex[field]:
                if len(key) == len(term): #Si es ? la longitud tendr que ser la misma.
                    permuterms = self.ptindex[field][key]
                    i=0
                    end = False
                    while i < len(permuterms) and not end:
                        if term in permuterms[i]:
                            result = self.or_posting(result, sorted(self.index[field][key].keys())) #Recorremos toda la lista de terminos y sus rotaciones para encontrar las posting list de los trminos.
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
        # Convertir la lista p a un set para mejorar el tiempo de busqueda, O(1) en Python.
        p1 = set(p.keys()) if isinstance(p, dict) else set(p)
        reversed_posting_list = set()
        for k in self.index['article'].keys():
            for new in self.index['article'][k]:
                if new not in p1:
                    reversed_posting_list.add(new)
        
        return sorted(list(reversed_posting_list))


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

    def get_terms_stemming(self, term, field='article'):
        """

        Devuelve los terminos asociados al stem de un termino.

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: terminos

        """
        stem = self.stemmer.stem(term)
        tokens = self.sindex[field][stem] if stem in self.sindex[field] else [term]
        return tokens



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

    def get_terms_permuterm(self, term, field="article"):
        """
        Busca en el diccionario de permuterms las palabras que se ajustan a la wildcard.
        Se hace una bsqueda exhaustiva.
        :param term: termino a buscar (con wildcard)
        :param field: campo donde buscar
        :return: los trminos que se asocian a esa busqueda.
        """
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
                        result += [key]
                        end = True
                    i = i + 1
        else:
            term = term[:-1]
            for key in self.ptindex[field]:
                if len(key) == len(term):
                    permuterms = self.ptindex[field][key]
                    i = 0
                    end = False
                    while i < len(permuterms) and not end:
                        if term in permuterms[i]:
                            result += [key]
                            end = True
                        i = i + 1
        return result


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
        noticiasprocesadas = 1 #Contador usado ms adelante para indicar el nmero de noticia procesada.
        #Resolvemos la query y en caso de que se aplique ranking aplicamos para las noticias resultantes.
        result = self.solve_query(query)
        q_sep = list(map(self.cleanquery, self.shunting_yard(self.infix_notation(query))))
        if not result:
            return 0
        #Si la consulta usa ranking, aplicamos para el resultado de la query.
        if self.use_ranking:
            result = self.rank_result(result, q_sep)

        print("Query: " + query)
        print("Number of results: " + str(len(result)))

        #Iterar sobre cada noticia resultante de la query...
        for ID in result:
            if not self.use_ranking:
                rank = 0
            else: rank = round(self.weight_noti[ID],4)

            #IDDocumento = self.news[ID]['doc_id']
            PosicionDocumento = self.news[ID][1]
            PathDocumento = self.docs[self.news[ID][0]]

            #Leer el documento que contiene la noticia que queremos obtener la informacin
            with open(PathDocumento) as fl:
                lista = json.load(fl)
                noticiait = lista[PosicionDocumento-1]
                #Ahora obtenemos los datos requeridos de la noticia (Keywords, Id de noticia(ya presente en el iterador), la fecha y el ttulo de esta)
                keywords_noticia = noticiait['keywords']
                titulo_noticia = noticiait['title']
                fecha_noticia = noticiait['date']
                #Distinguimos entre si se ha usado la opcin -N o no, y segn ello mostramos la informacin de la noticia por pantalla
                if not self.show_snippet:
                    print("#{}      ({})  ({})  ({})   {}      ({})".format(noticiasprocesadas, rank, ID, fecha_noticia,
                                                                            titulo_noticia, keywords_noticia))
                else:
                    print("#{} \nScore:{} \nNewsID: {}  \nDate: {}   \nTitle: {}   \nNKeywords: {}".format(
                    noticiasprocesadas, rank, ID, fecha_noticia, titulo_noticia, keywords_noticia))
                noticiasprocesadas += 1
        # Ahora viene la parte bonita, que es calcular el snippet en caso de ser requerido. Asimismo, lo implementaremos segn la segunda forma sugerida en el boletn.
            cuerpoST = []
            if self.show_snippet:
                cuerpoST = noticiait['article']
                cuerpoST = self.tokenize(cuerpoST)

            # Antes de retirar los espacios de la query, debemos separar los parntesis de la query


            # Aadimos el ndice de la palabra contenida en la query a la lista
            if self.permuterm:
                for part in q_sep:
                    if isinstance(part,list):
                        if any(d for d in part[1] if any(ds in d for ds in ["*", "?"])):
                            part[1] = self.get_terms_permuterm(part[1])
            if self.use_stemming:
                for part in q_sep:
                    if isinstance(part, list):
                        if len(part[1]) == 1:
                            part[1] = self.get_terms_stemming(part[1][0])

            aux_id = []
            for part in q_sep:
                if isinstance(part,list):
                    for term in part[1]:
                        if term in self.index["article"]:
                            if ID in self.index["article"][term]:
                                aux_id += [self.index["article"][term][ID][0]]

            aux_id.sort()  # Ordenamos ids por orden ascendente.
            a_devolver_snippet = ""  # Cadena vaca para devolver...
            Proc = False

            for i in range(len(aux_id)):
                # Comprobar que no estamos sobre el ltimo ndice
                if i < len(aux_id)-1 and not Proc:

                    id1 = aux_id[i]
                    id2 = aux_id[i + 1]

                    # Ahora hay que comprobar si se solapan. La distancia escogida arbitrariamente ser de 4 palabras. Para contexto usaremos 2 palabras a izquierda y derecha.
                    if id2 - id1 <= 4:
                        Proc = True
                        if id1 < 2:
                            a_devolver_snippet += " ".join(cuerpoST[id1 - id1:id2 + 2])
                        if id2 > len(cuerpoST) + 2:
                            a_devolver_snippet += "[...]" + " ".join(cuerpoST[id1 - 2:id2 + (len(cuerpoST) - id2)])
                        if id1 > 2 and id2 < len(cuerpoST) + 2:
                            a_devolver_snippet += "[...]" + " ".join(cuerpoST[id1 - 2:id2 + 2]) + "[...]"
                    else:
                        Proc = False
                        id1 = aux_id[i]
                        if id1 < 2:
                            a_devolver_snippet += " ".join(cuerpoST[id1 - id1:id1 + 2])
                        if id1 > len(cuerpoST) + 2:
                            a_devolver_snippet += "[...]" + " ".join(cuerpoST[id1 - 2:id1 + (len(cuerpoST) - id1)])
                        if id1 > 2 and id1 < len(cuerpoST) + 2:
                            a_devolver_snippet += "[...]" + " ".join(cuerpoST[id1 - 2:id1 + 2]) + "[...]"
                if i == len(aux_id)-1:
                    Proc = False
                    id1 = aux_id[len(aux_id)-1]
                    if id1 < 2:
                        a_devolver_snippet += " ".join(cuerpoST[id1 - id1:id1 + 2])
                    if id1 > len(cuerpoST) + 2:
                        a_devolver_snippet += "[...]" + " ".join(cuerpoST[id1 - 2:id1 + (len(cuerpoST) - id1)])
                    if id1 > 2 and id1 < len(cuerpoST) + 2:
                        a_devolver_snippet += "[...]" + " ".join(cuerpoST[id1 - 2:id1 + 2]) + "[...]"
            if self.show_snippet:
                print("Snippet: {} ".format(a_devolver_snippet))
                print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
        return len(result)


    def rank_result(self, result, query):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING

        Ordena los resultados de una query.

        param:  "result": lista de resultados sin ordenar
                "query": query, puede ser la query original, la query procesada o una lista de terminos


        return: la lista de resultados ordenada

        """

        #Trminos de la query
        t = {}

        if self.permuterm:
            for part in query:
                if isinstance(part, list):
                    if any(d for d in part[1] if any(ds in d for ds in ["*", "?"])):
                        part[1] = self.get_terms_permuterm(part[1])
        if self.use_stemming:
            for part in query:
                if isinstance(part, list):
                    if len(part[1]) == 1:
                        part[1] = self.get_terms_stemming(part[1][0])
        #Declaramos lista de pesados para las noticias
        Masquepesados = []

        #Por cada noticia en el resultado...
        for noticia in result:
            pesado_noticia = 0
            #Por cada trmino y campo (Pues hay que calcular una por una para todas la noticias) usando las calculadas anteriormente.
            for part in query:
                if isinstance(part,list):
                    for term in part[1]:
                        field = part[0]
                    #Para el pesado usaremos el explicado en el tema 4 de teora.
                        #1er paso: Pesado del trmino
                        tf = 0
                        ft = 0
                        if term in self.index[field]:
                            for c in self.index[field][term].keys():
                                if c == noticia:
                                    ft += len(self.index[field][term][c])
                            if ft > 0:
                                tf = math.log10(ft)
                            # 2do paso: Funcin global idf
                            df = len(self.index[field][term].keys())
                            idf = math.log10(len(self.news) / df)

                            # 3er paso: Acumular el pesado de cada trmino para obtener el total de la noticia
                            pesado_noticia = pesado_noticia + (tf * idf)

            #Aadir los pesados sobre cada noticia
            self.weight_noti[noticia] = self.weight_noti.get(noticia,0) + pesado_noticia
            Masquepesados.append(pesado_noticia)
            
        #Para finalizar, antes de devolver la lista, se ordena segn el ranking de noticias.
        res = [i for _,i in sorted(zip(Masquepesados,result), reverse = True)]

        return res
        

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
