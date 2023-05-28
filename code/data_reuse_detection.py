#----------------------------------------------------------------#
"""
Nombre del Proyecto: Detección de Texto Reutilizado en 
                     documentos y textos cientificos

Nombre de Integrantes de Equipo:
                                Roberto Valdez Jasso A01746863
                                Renata Montserrat de Luna Flores
                                Eduardo Acosta Hernández
Proposito de este entregable: 
                                Generar un modelo de detección
                                de reutilización de textos 
                                cientificios y estudiantiles
                                a nivel profesional
Fecha de Inicio: 24/05/2023
Fecha de Finalizado_ N/A
"""
#----------------------------------------------------------------#
#                          LIBRERIAS                             #
#----------------------------------------------------------------#
# Librerias que se utilizaran para este prototipo/proyecto
import re
import os
import nltk
from nltk.tokenize import  word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.util import ngrams
#----------------------------------------------------------------#
#                          DESCARGAS                             #
#----------------------------------------------------------------#
#Descarga necesaria para la actividad 
# estar descarga nos apoyar en tener las herramientas nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

#----------------------------------------------------------------#
#                        VARIABLES                               #
#----------------------------------------------------------------#
# Variables base para la ubicacion LOCAL ABSOLUTA de los textos de este proyecto
file_path_G = r"C:\Users\rober\OneDrive\Escritorio\Repositorios\PlagiarismDetectorTeam4\code\documents\documentos-genuinos"
file_path_SUS = r"C:\Users\rober\OneDrive\Escritorio\Repositorios\PlagiarismDetectorTeam4\code\documents\documentos-sospechosos"
file_path_OH = r"C:\Users\rober\OneDrive\Escritorio\Repositorios\PlagiarismDetectorTeam4\code\documents\documentos-con texto de otros"

#----------------------------------------------------------------#
#                           CODIGO                               #
#----------------------------------------------------------------#

# Clase reuse_Text_Detection
class reuse_Text_Detection:
    #-------------------------------------------------------#
    #                FUNCIONES  AUXILIARES                  #
    #-------------------------------------------------------#
    
    #Autor: Roberto Valdez Jasso
    # Función Auxiliar stemming_Text
    # Se encarga de generar el stemming de las oracciones
    # con el formato LancasterStemmer, el cual es el mas agresivo
    # de los formatos stemming
    def stemming_Text(text_list):
        LS = [] # Lista que guardara todas las palabras raices de cada oración
        LSU = [] # Lista que guardara todas las palabras raices de cada parrafo }
        # Llamamos a la funcion Lancaster para realizar el stemming    
        ss = LancasterStemmer() 
        data_list = text_list # variable que contiene la lista que contiene el texto
        # ciclamos por las palabras del documento
        for words in data_list:
            # Tokenizamos la palabra
            words = word_tokenize(words)
            # ciclamos por cada una de las palabras tokenizadas
            for w in words:
                # adjuntamos la palabra raiz a la lista
                LS.append(ss.stem(w))
                #print(ss.stem(w))
                # A la vez, generamos una lista de palabras unica raiz
                if w not in  LSU and w != "":
                    LSU.append(ss.stem(w))
        # regresamos a lista que contiene las palabras raiz por oración
        # como tambien la que contiene las palabras raiz de cada parrafo
        return LS, LSU 
    
    #Autor: Roberto Valdez Jasso
    # Función Auxiliar lemmatizer_Text
    # Se encarga de generar el lematización de las oracciones
    # con el formato lematizer, la cual  nos apoya encontrar
    # la palabra raiz exacta de acuerdo  a las reglas acordadas
    # dentro de esta función
    def lemmatizer_Text(text_list):
        LL = [] # Lista que guardara todas las palabras raices de cada oración
        LLU = []  # Lista que guardara todas las palabras raices de cada parrafo 
        # Llamamos a la funcion WordNetLemmatizer para realizar la lematización        
        lemmatizer = WordNetLemmatizer()
        data_list = text_list # variable que contiene la lista que contiene el texto
        # ciclamos por las palabras del documento
        for words in data_list:
            for w,tag in  pos_tag(word_tokenize(words)):
                if tag.startswith("NN"):
                    LL.append(lemmatizer.lemmatize(w, pos = "n"))
                    # A la vez, generamos una lista de palabras unica raiz
                    if w not in  LLU and w != "":
                        LLU.append(lemmatizer.lemmatize(w, pos = "n"))
                elif tag.startswith('VB'):
                    LL.append(lemmatizer.lemmatize(w, pos = "v"))
                    # A la vez, generamos una lista de palabras unica raiz
                    if w not in  LLU and w != "":
                        LLU.append(lemmatizer.lemmatize(w, pos = "v"))
                elif tag.startswith('JJ'):
                    LL.append(lemmatizer.lemmatize(w, pos = "a"))
                    # A la vez, generamos una lista de palabras unica raiz
                    if w not in  LLU and w != "":
                        LLU.append(lemmatizer.lemmatize(w, pos = "a"))
                else: 
                    LL.append(lemmatizer.lemmatize(w))
                    # A la vez, generamos una lista de palabras unica raiz
                    if w not in  LLU and w != "":
                        LLU.append(lemmatizer.lemmatize(w))
        # regresamos a lista que contiene las palabras raiz por oración
        # como tambien la que contiene las palabras raiz de cada parrafo
        return LL, LLU
    
    #Autor: Roberto Valdez Jasso
    # Función Auxiliar cleaning_Text
    # se encarga de tomar el texto y el diccionario
    # proporcionado y cambiar todos los simbolos disponibles
    # y encontrados en ambos elementos
    def cleaning_Text(text,dic):
        # Ciclo que analiza los elementos del diccionario
        for i, j in dic.items():
                # Si se encuentra el elemento dentro del texto se intercambia
                # por el un espacio vacio ""
                text = text.replace(i, j)
        # regresamos el texto limpio
        return text
    
    #Autor: Roberto Valdez Jasso
    # Función Auxiliar text_to_list
    # Se encarga de dividir el texto en oraciones y
    # generar la lista de oraciones de los parrafos
    def text_to_list(text):
        # Divide la oraciones del parrafo una vez encontrado un punto
        # y cambia las letras mayusculas a minusculas
        sentences=text.lower().split(".")
        # regresamos una lista de las oraciones encontradas por parrafo.
        return list(sentences)
    
    #-------------------------------------------------------#
    #              FIN DE FUNCIONES AUXILIARES              #
    #-------------------------------------------------------#

    #-------------------------------------------------------#
    #                FUNCIONES  N-GRAMAS                    #
    #-------------------------------------------------------#

    #Autor: Roberto Valdez Jasso
    # Función Auxiliar uni_grams_by_text
    # se encarga de realizar los unigramas de cada oración de cada texto
    def uni_grams_by_text(list_of_lists): # ya esta
        data_list = list_of_lists # Variable que contiene la lista de oraciones de cada texto
        #Unigramas
        uni = [] # Lista que guadara los unigramas existentes de cada texto
        #ciclamos por cada lista
        for lists in data_list:
            #ciclamos por cada palabra dentro de la lista
            for words in lists:
                # generamos los trigramas con ngramas
                unigrams =ngrams(words.split(),1)
                #print(unigrams)
                for w in unigrams:
                    #print(w)
                    uni.append(w) # adjuntamos el resultado
        return uni # regresamos la lista de unigramas

    #Autor: Roberto Valdez Jasso
    # Función Auxiliar bi_grams_by_text
    # se encarga de realizar los bigramas de cada oración de cada texto
    def bi_grams_by_text(list_of_lists): # ya quedo
        data_list = list_of_lists # Variable que contiene la lista de oraciones de cada texto
        #Bigramas
        bi = [] # Lista que guadara los bigramas  existentes de cada texto
        #print(data_list)
        #ciclamos por cada lista
        for lists in data_list:
            #print(lists)
            #ciclamos por cada palabra dentro de la lista
            for words in lists: 
                #print(words)
                # generamos los trigramas con ngramas
                bigrams = ngrams(words.split(),2)
                for w in bigrams:
                    #print(w)
                    bi.append(w) # adjuntamos el resultado
        return bi # regresamos la lista de bigramas

    #Autor: Roberto Valdez Jasso
    # Función Auxiliar tri_grams_by_text
    # se encarga de realizar los trigramas de cada oración de cada texto
    def tri_grams_by_text(list_of_lists): # ya quedo
        data_list = list_of_lists # Variable que contiene la lista de oraciones de cada texto 
        #Trigramas
        tri = [] # Lista que guadara los unigramas existentes de cada texto
        #ciclamos por cada lista
        for lists in data_list:
            #ciclamos por cada palabra dentro de la lista
            for words in lists:
                # generamos los trigramas con ngramas
                trigrams =ngrams(words.split(),3)
                for w in trigrams:
                    #print(w)
                    tri.append(w) # adjuntamos el resultado
        return tri # regresamos la lista de trigramas

    #-------------------------------------------------------#
    #                FIN FUNCIONES  N-GRAMAS                #
    #-------------------------------------------------------#

    #-------------------------------------------------------#
    #             FUNCIONES  LECTURA MAESTRA                #
    #-------------------------------------------------------#

    #Autor: Roberto Valdez Jasso
    # Función Reading_Files
    # Se encarga de limpiar,ajustar y preparar el texto
    # que se usara  para el entrenamiento y detección de plagio
    def reading_Files(path,data):
        #print(data) # Imprime los textos (Titulo de los textos) presentes dentro de la carpeta
        # Generamos el Path Completo a documento a analizar
        URL =  f"{path}\{data}"
        #print(URL)
        # Diccionario de elementos a borrar de los textos
        replacements = {"(": "", ")": "",",": "",":": "",";": "","¿": "","?": "","¡": "","!": "","*": "","+": "","-": "","%": "","@": "","$": "","#": "","=": "","/": "","{": "","}": "","[": "","]": "","|": "","&": ""}

        # Abrimos cada uno de los documentos 
        with open(URL, mode='r', encoding='utf-8') as f:
            text_no_line_breaks = ''
            for lines in f.readlines():
                if not lines.isspace(): 
                    stripped_line = lines.rstrip()
                    text_no_line_breaks += stripped_line +" " 
            # Limpieza de todo aquello que que tenga un arroba (como correo eletronico)
            # y ligas de internet
            re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "",text_no_line_breaks)
            # Llamamos a la funcion de limpieza de texto y le mandamos el texto
            # mas un diccionario con los simbolos que deseamos eliminar dentro de los textos
            text_no_symbols =  reuse_Text_Detection.cleaning_Text(text_no_line_breaks,replacements)
            # Pasamos todas las mayusculas a minusculas
            text_lower = text_no_symbols.lower()  # Listas completas y limpias
                                                    # primer resultado de las listas sin generar Stemming o lematización
            # Teniendo ya el texto limpio, lo mandamos a lista para su uso mas adelante
            lists = reuse_Text_Detection.text_to_list(text_lower) # Listas completas,separadas por oraciones y limpias
                                            # segundo resultado de las listas sin generar Stemming o lematización
            #print(lists) # Imprimimos los resultados

            # Ya teniendo las listas preparadas
            # Generamos stemming y lematización de los documentos
            # Stemming
            ST,STU =  reuse_Text_Detection.stemming_Text(lists)  # Listas completas,con palabras raices "agresivas" y limpias
                                            # tercer resultado de las listas con Stemming de Lancaster
            #print(ST) # Imprimimos los resultados de stemming
            # Lematización
            LEMA,LEMAU = reuse_Text_Detection.lemmatizer_Text(lists)  # Listas completas,con palabras raices "suaves" y limpias
                                                # tercer resultado de las listas con lematización
            #print(LEMA) # Imprimimos los resultados de lematización 
            
            # Finalmente cerramos el documento una vez creado todas la listas necesarias
            f.close()

            """
            Regresamos la variables que contiene lo siguiente:
            * text_lower = texto limpio sin separar (para generar los nuevos documentos)
            * lists = texto separado por oraciones para la realización de stemming, lematización y distancias
            * ST = Lista de textos modificados por stemming de Lancaster
            * LEMA =  Lista de textos modificados por Lematizacion y las reglas creadas en su función
            """
            return  text_lower, lists, ST, STU, LEMA, LEMAU

    #-------------------------------------------------------#
    #             FIN FUNCIONES  LECTURA MAESTRA            #
    #-------------------------------------------------------#

    #-------------------------------------------------------#
    #                FUNCIONES  LISTALES                    #
    #-------------------------------------------------------#

    #-------------------------------------------------------#
    #              FUNCIONES  DATOS GENUINOS                #
    #-------------------------------------------------------#

    #Autor: Roberto Valdez Jasso
    # Función training_Texts_G
    # funcion principal que se encarga de enlistar
    # todos los documentos presentes en la carpeta 
    # y  generar una lista de listas  de documentos genuinos
    # cada texto presente dentro de la misma
    def training_Texts_G():
        #print("Estos son los Genuinos")
        #--------------------------------------------#
        # Listas (Variables)
        #--------------------------------------------#
        # Lista para guardar los textos por documento limpio
        list_clean_text_G = []
        # Lista de lista de las oraciones de los textos por documento limpio
        list_of_lists_text_G = []
        # Lista con resultado de stemming por oracion de cada texto por documento limpio
        list_of_stemming_text_G = []
        # Lista con resultado de stemming por parrafo completo de cada texto por documento limpio
        list_of_stemming_text_unique_G = []
        # Lista con resultado de lematizer por oracion de cada texto por documento limpio
        list_of_lema_text_G = []
        # Lista con resultado de lematizer por parrafo completo de cada texto por documento limpio
        list_of_lema_text_unique_G = []
        #--------------------------------------------------#
        # Lista de N-gramas por las oraciones de cada texto
        #--------------------------------------------------#
        # Unigramas
        list_of_unigrams_by_text_G = [] # por oración dentro del documento
        #--------------------------------------------------#
        # Bigramas
        list_of_bigrams_by_text_G = [] # por oración dentro del documento
        #--------------------------------------------------#
        # Trigramas
        list_of_trigrams_by_text_G = [] # por oración dentro del documento
        #--------------------------------------------------#


        # Abrimos cada archivo presente dentro de la carperta presente en el Path seleccionado
        for file in os.listdir(file_path_G):
            # revisamos que cada documento tenga la terminación .txt
            if file.endswith(".txt"):
                # Obtenemos el Path completo 
                absolute_path = os.path.abspath(file)
                relative_path = file_path_G # path relativo (carpeta)
                full_path = os.path.join(absolute_path, relative_path) # juntamos ambos paths para tener el path completo al documento
                # llamamos a la funcion reading_files para la limpieza 

                # llamamos a la funcion reading_files para la limpieza 
                # y lista de la misma
                text_lower, lists, ST, STU, LEMA, LEMAU = reuse_Text_Detection.reading_Files(full_path, file)

                #-----------------------------------------------------#
                # Listas
                #-----------------------------------------------------#
                # Texto Limpio
                # Adjutamos todos los textos limpios de caracteres y simbolos por cada texto en la carpeta seleccionada
                list_clean_text_G.append(text_lower) 

                # Lista de listas limpias
                # Ajuntamos una lista que contiene todas las oraciones divididas por cada texto en la carpeta seleccionada
                list_of_lists_text_G.append(lists)
                #-----------------------------------------------------#
                # Lista de Stemming por oración
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_stemming_text_G.append(ST) 
                # Lista de Stemming por parrafo
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_stemming_text_unique_G.append(STU)
                #-----------------------------------------------------#
                # Lista de Lemitazer por oración
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_lema_text_G.append(LEMA)
                # Lista de Stemming por parrafo
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_lema_text_unique_G.append(LEMAU)
                #-----------------------------------------------------#
                #N-gramas
                #-----------------------------------------------------#
                """ 
                En las siguientes lineas de codigo de esta función
                se dara a conocer la llamada de las función de ngrama
                deseada, ya sea:
                1. Unigrama
                2. Bigrama
                3. Trigrama

                , las cuales se le llamara para poder generar los 
                patrones de n-grama deseado en cada una de las listas
                de oraciones y despues guardar los resultados
                en una lista de cada uno de las listas antes ya comentadas
                """
                #-----------------------------------------------------#
                # Unigramas
                #-----------------------------------------------------#
                # Realizamos la función de unigramas con las oraciones
                unigram = reuse_Text_Detection.uni_grams_by_text(list_of_lists_text_G)
                # y la abjuntamos a su lista destino
                list_of_unigrams_by_text_G.append(unigram)
                #-----------------------------------------------------#
                # Bigramas
                #-----------------------------------------------------#
                # Realizamos la función de bigramas con las oraciones
                bigram = reuse_Text_Detection.bi_grams_by_text(list_of_lists_text_G)
                # y la abjuntamos a su lista destino
                list_of_bigrams_by_text_G.append(bigram)
                #-----------------------------------------------------#
                # Trigramas
                #-----------------------------------------------------#
                # Realizamos la función de trigramas con las oraciones
                triagram = reuse_Text_Detection.tri_grams_by_text(list_of_lists_text_G)
                # y la abjuntamos a su lista destino
                list_of_trigrams_by_text_G.append(triagram)
                #-----------------------------------------------------#
                #      AQUI SE ACABA EL CICLO DE DOCUMENTOS           #
                #-----------------------------------------------------#
        #-----------------------------------------------------------------------#
        # Finalmente imprimimos los resultados de cada una de las listas generadas
        #-----------------------------------------------------------------------#
        # Listas
        #-----------------------------------------------------------------------#
        #print(list_clean_text_G) 
        #print(list_of_lists_text_G) 
        #print(list_of_stemming_text_G) 
        #print(list_of_stemming_text_unique_G) 
        #print(list_of_lema_text_G) 
        #print(list_of_lema_text_unique_G)
        #-----------------------------------------------------------------------#
        # N-gramas
        #-----------------------------------------------------------------------# 
        # Unigrama
        #-----------------------------------------------------------------------#
        #print(list_of_unigrams_by_text_G)
        #-----------------------------------------------------------------------#
        # Bigrama
        #-----------------------------------------------------------------------#
        #print(list_of_bigrams_by_text_G)
        #-----------------------------------------------------------------------#
        # Trigrama
        #print(list_of_trigrams_by_text_G)
        #-----------------------------------------------------------------------#

        # Por ultimo regramos
        return list_clean_text_G, list_of_lists_text_G, list_of_stemming_text_G, list_of_stemming_text_unique_G, list_of_lema_text_G, list_of_lema_text_unique_G,list_of_unigrams_by_text_G, list_of_bigrams_by_text_G, list_of_trigrams_by_text_G   
        
    #-------------------------------------------------------#
    #           FUNCIONES  DATOS SOSPECHOSOS                #
    #-------------------------------------------------------#

    #Autor: Roberto Valdez Jasso
    # Función training_Texts_SUS
    # funcion principal que se encarga de enlistar
    # todos los documentos presentes en la carpeta 
    # y  generar una lista de listas  de documentos sospechosos
    # cada texto presente dentro de la misma
    def training_Texts_SUS():
        #print("Estos son los Sospechosos")
        #--------------------------------------------#
        # Listas (Variables)
        #--------------------------------------------#
        # Lista para guardar los textos por documento limpio
        list_clean_text_SUS = []
        # Lista de lista de las oraciones de los textos por documento limpio
        list_of_lists_text_SUS = []
        # Lista con resultado de stemming por oracion de cada texto por documento limpio
        list_of_stemming_text_SUS = []
        # Lista con resultado de stemming por parrafo completo de cada texto por documento limpio
        list_of_stemming_text_unique_SUS = []
        # Lista con resultado de lematizer por oracion de cada texto por documento limpio
        list_of_lema_text_SUS = []
        # Lista con resultado de lematizer por parrafo completo de cada texto por documento limpio
        list_of_lema_text_unique_SUS = []
        #--------------------------------------------------#
        # Lista de N-gramas por las oraciones de cada texto
        #--------------------------------------------------#
        # Unigramas
        list_of_unigrams_by_text_SUS = [] # por oración dentro del documento
        #--------------------------------------------------#
        # Bigramas
        list_of_bigrams_by_text_SUS = [] # por oración dentro del documento
        #--------------------------------------------------#
        # Trigramas
        list_of_trigrams_by_text_SUS = [] # por oración dentro del documento
        #--------------------------------------------------#


        # Abrimos cada archivo presente dentro de la carperta presente en el Path seleccionado
        for file in os.listdir(file_path_SUS):
            # revisamos que cada documento tenga la terminación .txt
            if file.endswith(".txt"):
                # Obtenemos el Path completo 
                absolute_path = os.path.abspath(file)
                relative_path = file_path_SUS # path relativo (carpeta)
                full_path = os.path.join(absolute_path, relative_path) # juntamos ambos paths para tener el path completo al documento
                
                # llamamos a la funcion reading_files para la limpieza 
                # y lista de la misma
                text_lower, lists, ST, STU, LEMA, LEMAU = reuse_Text_Detection.reading_Files(full_path,file)

                #-----------------------------------------------------#
                # Listas
                #-----------------------------------------------------#
                # Texto Limpio
                # Adjutamos todos los textos limpios de caracteres y simbolos por cada texto en la carpeta seleccionada
                list_clean_text_SUS.append(text_lower) 

                # Lista de listas limpias
                # Ajuntamos una lista que contiene todas las oraciones divididas por cada texto en la carpeta seleccionada
                list_of_lists_text_SUS.append(lists)
                #-----------------------------------------------------#
                # Lista de Stemming por oración
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_stemming_text_SUS.append(ST) 
                # Lista de Stemming por parrafo
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_stemming_text_unique_SUS.append(STU)
                #-----------------------------------------------------#
                # Lista de Lemitazer por oración
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_lema_text_SUS.append(LEMA)
                # Lista de Stemming por parrafo
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_lema_text_unique_SUS.append(LEMAU)
                #-----------------------------------------------------#
                #N-gramas
                #-----------------------------------------------------#
                """ 
                En las siguientes lineas de codigo de esta función
                se dara a conocer la llamada de las función de ngrama
                deseada, ya sea:
                1. Unigrama
                2. Bigrama
                3. Trigrama

                , las cuales se le llamara para poder generar los 
                patrones de n-grama deseado en cada una de las listas
                de oraciones y despues guardar los resultados
                en una lista de cada uno de las listas antes ya comentadas
                """
                #-----------------------------------------------------#
                # Unigramas
                #-----------------------------------------------------#
                # Realizamos la función de unigramas con las oraciones
                unigram = reuse_Text_Detection.uni_grams_by_text(list_of_lists_text_SUS)
                # y la abjuntamos a su lista destino
                list_of_unigrams_by_text_SUS.append(unigram)
                #-----------------------------------------------------#
                # Bigramas
                #-----------------------------------------------------#
                # Realizamos la función de bigramas con las oraciones
                bigram = reuse_Text_Detection.bi_grams_by_text(list_of_lists_text_SUS)
                # y la abjuntamos a su lista destino
                list_of_bigrams_by_text_SUS.append(bigram)
                #-----------------------------------------------------#
                # Trigramas
                #-----------------------------------------------------#
                # Realizamos la función de trigramas con las oraciones
                triagram = reuse_Text_Detection.tri_grams_by_text(list_of_lists_text_SUS)
                # y la abjuntamos a su lista destino
                list_of_trigrams_by_text_SUS.append(triagram)
                #-----------------------------------------------------#
                #      AQUI SE ACABA EL CICLO DE DOCUMENTOS           #
                #-----------------------------------------------------#
        #-----------------------------------------------------------------------#
        # Finalmente imprimimos los resultados de cada una de las listas generadas
        #-----------------------------------------------------------------------#
        # Listas
        #-----------------------------------------------------------------------#
        #print(list_clean_text_SUS) 
        #print(list_of_lists_text_SUS) 
        #print(list_of_stemming_text_SUS) 
        #print(list_of_stemming_text_unique_SUS) 
        #print(list_of_lema_text_SUS) 
        #print(list_of_lema_text_unique_SUS)
        #-----------------------------------------------------------------------#
        # N-gramas
        #-----------------------------------------------------------------------# 
        # Unigrama
        #-----------------------------------------------------------------------#
        #print(list_of_unigrams_by_text_SUS)
        #-----------------------------------------------------------------------#
        # Bigrama
        #-----------------------------------------------------------------------#
        #print(list_of_bigrams_by_text_SUS)
        #-----------------------------------------------------------------------#
        # Trigrama
        #print(list_of_trigrams_by_text_SUS)
        #-----------------------------------------------------------------------#

        # Por ultimo regramos
        return list_clean_text_SUS, list_of_lists_text_SUS, list_of_stemming_text_SUS, list_of_stemming_text_unique_SUS, list_of_lema_text_SUS, list_of_lema_text_unique_SUS,list_of_unigrams_by_text_SUS, list_of_bigrams_by_text_SUS, list_of_trigrams_by_text_SUS   
        
    #------------------------------------------------------#
    #       FUNCIONES  DATOS CON TEXTO DE OTROS            #
    #-------------------------------------------------------#

    #Autor: Roberto Valdez Jasso
    # Función training_Texts_OH
    # funcion principal que se encarga de enlistar
    # todos los documentos presentes en la carpeta 
    # y  generar una lista de listas  de documentos de texto de otros
    # cada texto presente dentro de la misma
    def training_Texts_OH():
        #print("Estos son Otros textos")
        #--------------------------------------------#
        # Listas (Variables)
        #--------------------------------------------#
        # Lista para guardar los textos por documento limpio
        list_clean_text_OH = []
        # Lista de lista de las oraciones de los textos por documento limpio
        list_of_lists_text_OH = []
        # Lista con resultado de stemming por oracion de cada texto por documento limpio
        list_of_stemming_text_OH = []
        # Lista con resultado de stemming por parrafo completo de cada texto por documento limpio
        list_of_stemming_text_unique_OH = []
        # Lista con resultado de lematizer por oracion de cada texto por documento limpio
        list_of_lema_text_OH = []
        # Lista con resultado de lematizer por parrafo completo de cada texto por documento limpio
        list_of_lema_text_unique_OH = []
        #--------------------------------------------------#
        # Lista de N-gramas por las oraciones de cada texto
        #--------------------------------------------------#
        # Unigramas
        list_of_unigrams_by_text_OH = [] # por oración dentro del documento
        #--------------------------------------------------#
        # Bigramas
        list_of_bigrams_by_text_OH = [] # por oración dentro del documento
        #--------------------------------------------------#
        # Trigramas
        list_of_trigrams_by_text_OH = [] # por oración dentro del documento
        #--------------------------------------------------#


        # Abrimos cada archivo presente dentro de la carperta presente en el Path seleccionado
        for file in os.listdir(file_path_OH):
            # revisamos que cada documento tenga la terminación .txt
            if file.endswith(".txt"):
                # Obtenemos el Path completo 
                absolute_path = os.path.abspath(file)
                relative_path = file_path_OH # path relativo (carpeta)
                full_path = os.path.join(absolute_path, relative_path) # juntamos ambos paths para tener el path completo al documento
                # llamamos a la funcion reading_files para la limpieza 
                # y lista de la misma
                text_lower, lists, ST, STU, LEMA, LEMAU = reuse_Text_Detection.reading_Files(full_path, file)

                #-----------------------------------------------------#
                # Listas
                #-----------------------------------------------------#
                # Texto Limpio
                # Adjutamos todos los textos limpios de caracteres y simbolos por cada texto en la carpeta seleccionada
                list_clean_text_OH.append(text_lower) 

                # Lista de listas limpias
                # Ajuntamos una lista que contiene todas las oraciones divididas por cada texto en la carpeta seleccionada
                list_of_lists_text_OH.append(lists)
                #-----------------------------------------------------#
                # Lista de Stemming por oración
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_stemming_text_OH.append(ST) 
                # Lista de Stemming por parrafo
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_stemming_text_unique_OH.append(STU)
                #-----------------------------------------------------#
                # Lista de Lemitazer por oración
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_lema_text_OH.append(LEMA)
                # Lista de Stemming por parrafo
                # Ajuntamos una lista que contiene todas las palabras raiz por cada texto en la carpeta seleccionada 
                list_of_lema_text_unique_OH.append(LEMAU)
                #-----------------------------------------------------#
                #N-gramas
                #-----------------------------------------------------#
                """ 
                En las siguientes lineas de codigo de esta función
                se dara a conocer la llamada de las función de ngrama
                deseada, ya sea:
                1. Unigrama
                2. Bigrama
                3. Trigrama

                , las cuales se le llamara para poder generar los 
                patrones de n-grama deseado en cada una de las listas
                de oraciones y despues guardar los resultados
                en una lista de cada uno de las listas antes ya comentadas
                """
                #-----------------------------------------------------#
                # Unigramas
                #-----------------------------------------------------#
                # Realizamos la función de unigramas con las oraciones
                unigram = reuse_Text_Detection.uni_grams_by_text(list_of_lists_text_OH)
                # y la abjuntamos a su lista destino
                list_of_unigrams_by_text_OH.append(unigram)
                #-----------------------------------------------------#
                # Bigramas
                #-----------------------------------------------------#
                # Realizamos la función de bigramas con las oraciones
                bigram = reuse_Text_Detection.bi_grams_by_text(list_of_lists_text_OH)
                # y la abjuntamos a su lista destino
                list_of_bigrams_by_text_OH.append(bigram)
                #-----------------------------------------------------#
                # Trigramas
                #-----------------------------------------------------#
                # Realizamos la función de trigramas con las oraciones
                triagram = reuse_Text_Detection.tri_grams_by_text(list_of_lists_text_OH)
                # y la abjuntamos a su lista destino
                list_of_trigrams_by_text_OH.append(triagram)
                #-----------------------------------------------------#
                #      AQUI SE ACABA EL CICLO DE DOCUMENTOS           #
                #-----------------------------------------------------#
        #-----------------------------------------------------------------------#
        # Finalmente imprimimos los resultados de cada una de las listas generadas
        #-----------------------------------------------------------------------#
        # Listas
        #-----------------------------------------------------------------------#
        #print(list_clean_text_OH) 
        #print(list_of_lists_text_OH) 
        #print(list_of_stemming_text_OH) 
        #print(list_of_stemming_text_unique_OH) 
        #print(list_of_lema_text_OH) 
        #print(list_of_lema_text_unique_OH)
        #-----------------------------------------------------------------------#
        # N-gramas
        #-----------------------------------------------------------------------# 
        # Unigrama
        #-----------------------------------------------------------------------#
        #print(list_of_unigrams_by_text_OH)
        #-----------------------------------------------------------------------#
        # Bigrama
        #-----------------------------------------------------------------------#
        #print(list_of_bigrams_by_text_OH)
        #-----------------------------------------------------------------------#
        # Trigrama
        #print(list_of_trigrams_by_text_OH)
        #-----------------------------------------------------------------------#

       # Por ultimo regramos
        return list_clean_text_OH, list_of_lists_text_OH, list_of_stemming_text_OH, list_of_stemming_text_unique_OH, list_of_lema_text_OH, list_of_lema_text_unique_OH,list_of_unigrams_by_text_OH, list_of_bigrams_by_text_OH, list_of_trigrams_by_text_OH   

#----------------------------------------------------------------#
#                         FIN  CODIGO                            #
#----------------------------------------------------------------#