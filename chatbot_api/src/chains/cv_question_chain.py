import os

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

AGENT_MODEL = os.getenv("AGENT_MODEL")

# docker mode: host.docker.internal
# local mode: 127.0.0.1

client = chromadb.HttpClient(host="host.docker.internal", port=8000)

print("listing chromadb collections from chain:", client.list_collections())

# persist_directory = "chroma"

print("this is from the chain")
print(os.getcwd())

embedding_function = OllamaEmbeddings(base_url="http://host.docker.internal:11434", model=AGENT_MODEL)

vector_db = Chroma(
    client=client,
    collection_name="resume_collection",
    embedding_function=embedding_function,
    collection_metadata={"hnsw:space": "cosine"}
)

llm = Ollama(base_url="http://host.docker.internal:11434", model=AGENT_MODEL, keep_alive="-1", temperature=0.0)

question_template = """Your job is to answer questions about CVs and resumes based on the below context.
If the question is not related to the content of a resume, past job experiences or skills, you can say you don't know.
If the prompt is not a question related to a resume, let the user know. 
Use the entire prompt to generate your answer.
Keep your answers concise and to the point.

{context}
"""

question_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=question_template
    )
)

question_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["input"], template="{input}"
    )
)

messages=[question_system_prompt, question_human_prompt]

question_prompt = ChatPromptTemplate(
    input_variables=["context", "input"], messages=messages
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

document_chain = create_stuff_documents_chain(llm=llm, prompt=question_prompt)

question_vector_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)



