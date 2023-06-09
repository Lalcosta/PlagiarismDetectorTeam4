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

from sentence_transformers import SentenceTransformer, util
import data_reuse_detection


#Creamos las variables para el texto de genuino
list_clean_text_SUS, list_of_lists_text_SUS, list_of_stemming_text_SUS, list_of_stemming_text_unique_SUS, list_of_lema_text_SUS, list_of_lema_text_unique_SUS,list_of_unigrams_by_text_SUS, list_of_bigrams_by_text_SUS, list_of_trigrams_by_text_SUS = data_reuse_detection.reuse_Text_Detection.training_Texts_SUS()

#Creamos las variables para el texto sospechoso
list_clean_text_G, list_of_lists_text_G, list_of_stemming_text_G, list_of_stemming_text_unique_G, list_of_lema_text_G, list_of_lema_text_unique_G,list_of_unigrams_by_text_G, list_of_bigrams_by_text_G, list_of_trigrams_by_text_G = data_reuse_detection.reuse_Text_Detection.training_Texts_G()

def deteccion_plagio(genuin, suspicious):
    # Cargar el modelo BERT 
    model = SentenceTransformer('LaBSE')

    for i, suspect_text in enumerate(suspicious):
        max_similarity = 0.0
        max_similarity_index = -1

        # Vectorizar el texto sospechoso
        suspect_vector = model.encode([' '.join(suspect_text)], convert_to_tensor=True)

        for j, genuine_text in enumerate(genuin):
            # Vectorizar el texto genuino
            genuine_vector = model.encode([' '.join(genuine_text)], convert_to_tensor=True)

            # Calcular la similitud de coseno entre los vectores
            similarity = util.pytorch_cos_sim(suspect_vector, genuine_vector)[0][0].item()
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = j
                
        result=""
        if max_similarity>=0.81:
            result="PLAGIO DETECTADO"
        else:
            result="DOCUMENTO GENUINO"
            
        print(f"Texto sospechoso {i+1} Similitud de: {max_similarity} con el texto genuino {max_similarity_index+1}----{result}")



print("lemma")
print("-----------------------------")
deteccion_plagio(list_of_lema_text_G, list_of_lema_text_SUS)
print("lemma unique")
print("-----------------------------")
deteccion_plagio(list_of_lema_text_unique_G, list_of_lema_text_unique_SUS)
print("stemming")
print("-----------------------------")
deteccion_plagio(list_of_stemming_text_G, list_of_stemming_text_SUS)
print("stemming unique")
print("-----------------------------")
deteccion_plagio(list_of_stemming_text_unique_G, list_of_stemming_text_unique_SUS)