import ollama
from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from pydantic import BaseModel
from async_utils import async_retry

app = FastAPI(
    title="Test Chatbot",
    description="basic chatbot",
)

class QueryInput(BaseModel):
    text: str

llm = Ollama(base_url="http://localhost:11434", model="phi3")

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    response = llm.invoke(query)
    response_text = {"text": response}

    return response_text

@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/query-model")
async def query_model(query: str):
    query_response = await invoke_agent_with_retry(query)
    print("query_response", query_response)

    return query_response