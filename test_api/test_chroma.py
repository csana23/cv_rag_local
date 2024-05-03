import chromadb
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

AGENT_MODEL = os.getenv("AGENT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

client = chromadb.HttpClient(host="127.0.0.1", port=8000)
print("listing chromadb collections from chain:", client.list_collections())

""" ollama_client = ollama.Client(host="http://127.0.0.1:11434")

models = [model['name'].replace(":latest", "") for model in ollama_client.list()['models']]

print(models) """



embedding_function = OllamaEmbeddings(base_url="http://127.0.0.1:11434", model=EMBEDDING_MODEL)

# Create a ChromaDB instance
db = Chroma(
    client=client,
    collection_name="resume_collection",
    embedding_function=embedding_function,
    collection_metadata={"hnsw:space": "cosine"}
)

query_text = "Did Richard Csanaki complete his MSc studies?"

results = db.similarity_search_with_relevance_scores(
    query=query_text, 
    k= 3,
    score_threshold=0.3
)

for result in results:
    print(result)
    print("\n\n")




