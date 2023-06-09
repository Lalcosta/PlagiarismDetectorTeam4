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
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import data_reuse_detection



#Creamos las variables para el texto de genuino
list_clean_text_SUS, list_of_lists_text_SUS, list_of_stemming_text_SUS, list_of_stemming_text_unique_SUS, list_of_lema_text_SUS, list_of_lema_text_unique_SUS,list_of_unigrams_by_text_SUS, list_of_bigrams_by_text_SUS, list_of_trigrams_by_text_SUS = data_reuse_detection.reuse_Text_Detection.training_Texts_SUS()

#Creamos las variables para el texto sospechoso
list_clean_text_G, list_of_lists_text_G, list_of_stemming_text_G, list_of_stemming_text_unique_G, list_of_lema_text_G, list_of_lema_text_unique_G,list_of_unigrams_by_text_G, list_of_bigrams_by_text_G, list_of_trigrams_by_text_G = data_reuse_detection.reuse_Text_Detection.training_Texts_G()


def deteccion_plagio(genuin, suspicious):
    # Cargar el modelo BERT
    model = SentenceTransformer('LaBSE')

    
    predictions = []  # Predicciones del modelo (similitud de coseno)

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

        
        labels=[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]  # Etiquetas verdaderas (0: genuino, 1: plagio)

        predictions.append(max_similarity)
        result=""
        if max_similarity>=0.81:
            result="PLAGIO DETECTADO"
        else:
            result="DOCUMENTO GENUINO"
        print(f"Texto sospechoso {i+1} Similitud de: {max_similarity} con el texto genuino {max_similarity_index+1}----{result}")

    labels = np.array(labels)
    predictions = np.array(predictions)

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()
    print("Labels:")
    print(labels)
    print("--------")
    print("Predictions:")
    print(predictions)
    print("-----")
    print(f"AUC: {roc_auc}")

    
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