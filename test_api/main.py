import ollama
from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from pydantic import BaseModel
from async_utils import async_retry
from cv_question_chain import question_vector_chain, client
from langchain.schema import Document

app = FastAPI(
    title="Test Chatbot",
    description="basic chatbot",
)

class QueryInput(BaseModel):
    text: str

class QueryOutput(BaseModel):
    input: str
    context: list[Document]
    answer: str


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):

    return question_vector_chain.invoke({"input": query})

@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/query-model")
async def query_model(query: QueryInput) -> QueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    print("query_response", query_response)
    print("listing chromadb collections from chain:", client.list_collections())

    return dict(query_response)