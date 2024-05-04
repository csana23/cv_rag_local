# cv_rag_local
a repo for a rag project with a local model served by ollama
- backend: FastAPI, pydantic
- LLM stuff: llama3 in ollama, nomic-embed-text embedding model in other ollama instance, ColBERT reranking for RAG precision
- chromadb vector database
- streamlit frontend

demo link: https://www.youtube.com/watch?v=M--grymJ7ic

## How to run

Create .env file in projects root folder:

```
AGENT_MODEL=llama3
EMBEDDING_MODEL=nomic-embed-text
CHATBOT_URL=http://host.docker.internal:8005/cv-rag-agent
```

Add your pdfs to cv_rag/chromadb_etl/data folder

Build with docker:

```console
docker-compose up --build
```