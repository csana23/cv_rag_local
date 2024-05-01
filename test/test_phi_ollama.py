from langchain_community.llms import Ollama
import ollama
import os
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

ollama_client = ollama.Client(host="http://localhost:11434")

embedding_function = OllamaEmbeddingFunction(url="http://host.docker.internal:11434", model_name=os.getenv("AGENT_MODEL"))

print(embedding_function)

def get_models():
  models = [model['name'] for model in ollama_client.list()['models']]

  print(models)

  if 'phi3:latest' in models:
      print("model exists")
  else:
      print("model does not exist, pulling")


# response = ollama_client.embeddings(model="phi3", prompt="text to embed")
# embedding = response["embedding"]
# print(embedding)

'''
response = ollama_client.chat(model='phi3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response)
'''
