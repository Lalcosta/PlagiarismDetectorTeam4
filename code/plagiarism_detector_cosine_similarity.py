
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
#Importamos las librerías necesarias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#importamos la función para realizar el tratamiento de los datos
import data_reuse_detection


#Creamos las variables para el texto de genuino
list_clean_text_SUS, list_of_lists_text_SUS, list_of_stemming_text_SUS, list_of_stemming_text_unique_SUS, list_of_lema_text_SUS, list_of_lema_text_unique_SUS,list_of_unigrams_by_text_SUS, list_of_bigrams_by_text_SUS, list_of_trigrams_by_text_SUS = data_reuse_detection.reuse_Text_Detection.training_Texts_SUS()

#Creamos las variables para el texto sospechoso
list_clean_text_G, list_of_lists_text_G, list_of_stemming_text_G, list_of_stemming_text_unique_G, list_of_lema_text_G, list_of_lema_text_unique_G,list_of_unigrams_by_text_G, list_of_bigrams_by_text_G, list_of_trigrams_by_text_G = data_reuse_detection.reuse_Text_Detection.training_Texts_G()

#Almacenamos los bigramas, trigramas, lemmas y stems correspondientes a cada tipo de texto en dos diccionarios
#text1 corresponde a las variables de los datos genuinos
text1 = {
    'bigrams': list_of_bigrams_by_text_G,
    'trigrams': list_of_trigrams_by_text_G,
    'lemmas': list_of_lema_text_G,
    'stems': list_of_stemming_text_G
    
}
#text2 corresponde a las variables de los datos sospechosos
text2 = {
    'bigrams': list_of_bigrams_by_text_SUS,
    'trigrams': list_of_trigrams_by_text_SUS,
    'lemmas': list_of_lema_text_SUS,
    'stems': list_of_stemming_text_SUS
    }

#Creamos una funcion para convertir los ngramas en listas 
def convert_ngrams_to_str(bigrams):
    return [' '.join(map(str, bigram)) for bigram in bigrams]

#Función para detectar plagio
def plagiarism_detector(text1, text2):
    # Combina las listas en un solo texto para cada conjunto de características
    
    #Variables para almacenar los bigramas
    text1_bigrams = convert_ngrams_to_str(text1['bigrams'])
    text2_bigrams = convert_ngrams_to_str(text2['bigrams'])
    
    #Variables para almacenar los trigramas
    text1_trigrams = convert_ngrams_to_str(text1['trigrams'])
    text2_trigrams = convert_ngrams_to_str(text2['trigrams'])
    
    #Variables para almacenar los lemmas
    text1_lemmas = ' '.join([lemma for sublist in text1['lemmas'] for lemma in sublist])
    text2_lemmas = ' '.join([lemma for sublist in text2['lemmas'] for lemma in sublist])
    
    #Variables para alamcenar stems
    text1_stems = ' '.join([stem for sublist in text1['stems'] for stem in sublist])
    text2_stems = ' '.join([stem for sublist in text2['stems'] for stem in sublist])
    
    
    # Crea el vectorizador y ajusta el texto de entrenamiento
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(text1_bigrams + text2_bigrams +
                                       text1_trigrams + text2_trigrams + [text1_lemmas, text2_lemmas] +
                                       [text1_stems, text2_stems])
    
    
    # Calcula la similitud del coseno entre los textos
    similarity_scores = cosine_similarity(vectors)
    
    # Imprime los puntajes de similitud para cada conjunto de características
    print("Unigrams similarity score:", similarity_scores[0, 1])
    print("Bigrams similarity score:", similarity_scores[1, 2])
    print("Trigrams similarity score:", similarity_scores[2, 3])
    print("Lemmas similarity score:", similarity_scores[3, 4])
    print("Stems similarity score:", similarity_scores[4, 5])
    
    # Verifica si hay plagio basado en los puntajes de similitud en este caso consideramos un ubral de 0.9
    #Es decir que si la similitud de coseno es mayor a 0.9 es indice de plágio
    is_plagiarism = any(score > 0.9 for score in similarity_scores.flatten())
    if is_plagiarism:
        print("Plagiarism detected!")
    else:
        print("No plagiarism detected.")
    
plagiarism_detector(text1,text2)