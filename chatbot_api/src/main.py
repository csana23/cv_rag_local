# from agents.cv_rag_agent import cv_rag_agent_executor
from chains.cv_question_chain import question_vector_chain
from fastapi import FastAPI
from models.cv_rag_query import CVQueryInput, CVQueryOutput
from utils.async_utils import async_retry

app = FastAPI(
    title="CV Chatbot",
    description="Endpoints for a CV RAG chatbot",
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return question_vector_chain.invoke({"input": query}) 


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/cv-rag-agent")
async def query_cv_agent(query: CVQueryInput) -> CVQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    print("query_response", query_response)

    return query_response
