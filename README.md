# cv_rag_local
a repo for a rag project with a local model served by ollama
- llama3 as LLM and nomic-embed-text as embedding model, both running in separate ollama instances
- chromadb vector database for storing user input-, and PDF file embeddings, running in server mode
- RAG component with cosine similarity search followed by ColBERT reranking and metadata inclusion in retrieved document chunks
- FastAPI backend to process user prompts and initiate LangChain process
- Streamlit frontend
- deployed as a Docker container, able to utilize Nvidia GPUs
- also available with OpenAI's gpt-3.5 model with added LangChain agent functions

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
