from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

model_name = "bert-base-multilingual-cased"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

word_list = ["heshapllo", "world", "vectorize", "input","plagiarism "]  # Lista de palabras en inglés
encoded_input = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)
embeddings_EN = model_output.last_hidden_state

print(embeddings_EN.shape)
print(embeddings_EN)

texto_ejemplo = "hola","mundo","vectorize", "input","plagiarism "  # Texto en español
encoded_input = tokenizer(texto_ejemplo, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)
embeddings_ES = model_output.last_hidden_state

print(embeddings_ES.shape)
print(embeddings_ES)
similarity = cosine_similarity(embeddings_ES[0], embeddings_EN[0])

print("Similarity:\n")
print(similarity)