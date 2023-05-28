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
# Importamos nuestros procesos a esta nuevo documento
import data_reuse_detection

#----------------------------------------------------------------#
#                           CODIGO                               #
#----------------------------------------------------------------#

#Autor: Roberto Valdez Jasso
# Probamos todo el proceso de reutilización de texto y datos
# con la siguiente función
def main():
    # Resultado de documento genuinos
    list_clean_text_G, list_of_lists_text_G, list_of_stemming_text_G, list_of_stemming_text_unique_G, list_of_lema_text_G, list_of_lema_text_unique_G,list_of_unigrams_by_text_G, list_of_bigrams_by_text_G, list_of_trigrams_by_text_G = data_reuse_detection.reuse_Text_Detection.training_Texts_G()
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_clean_text_G)
    print("#-------------------------------------------------------------------------------------------------#")
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_of_lists_text_G)
    print("#-------------------------------------------------------------------------------------------------#")
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_of_stemming_text_G)
    print("#-------------------------------------------------------------------------------------------------#")
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_of_stemming_text_unique_G)
    print("#-------------------------------------------------------------------------------------------------#")
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_of_lema_text_G)
    print("#-------------------------------------------------------------------------------------------------#")
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_of_lema_text_unique_G)
    print("#-------------------------------------------------------------------------------------------------#")
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_of_unigrams_by_text_G)
    print("#-------------------------------------------------------------------------------------------------#")
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_of_bigrams_by_text_G)
    print("#-------------------------------------------------------------------------------------------------#")
    print("#-------------------------------------------------------------------------------------------------#")
    print(list_of_trigrams_by_text_G)



if __name__ == "__main__":
    main()