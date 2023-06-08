from nltk import FreqDist
from nltk.metrics.distance import cosine_distance

def find_similarity(list1, list2):
    results = []  
    
    for suspicious_text in list2:
        similarities = []  
        
        for genuine_text in list1:
            
            dist = cosine_distance(FreqDist(suspicious_text), FreqDist(genuine_text))
            similarities.append(dist)
        
     
        sorted_similarities = sorted(enumerate(similarities), key=lambda x: x[1])[:5]
        
        
        similar_documents = [list1[i[0]] for i in sorted_similarities]
        results.append(similar_documents)
    
    return results