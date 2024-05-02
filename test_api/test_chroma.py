import chromadb
import os

client = chromadb.HttpClient(host="127.0.0.1", port=8000)

print("listing chromadb collections from chain:", client.list_collections())

from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

url = "127.0.0.1" if os.getenv("LOCAL") == "yes" else "host.docker.internal"

# Load your documents (replace 'docs' with your actual document data)
embedding_function = OllamaEmbeddings(base_url="http://" + url + ":11434", model="phi3")
docs = ["Document 1", "Document 2", ...]  # Your actual documents

# Create a ChromaDB instance
db = Chroma(
    client=client,
    collection_name="resume_collection",
    embedding_function=embedding_function
)

# Now, to get all documents from the Chroma database:
all_documents = db.get()

# Print the IDs, embeddings, and documents
print("all_documents:", all_documents)
