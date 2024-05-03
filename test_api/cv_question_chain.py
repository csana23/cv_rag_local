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

CHROMA_PATH = "../chromadb_etl/src/chroma"
client = chromadb.PersistentClient(path=CHROMA_PATH)

print("listing chromadb collections from chain:", client.list_collections())

# persist_directory = "chroma"

print("this is from the chain")
print(os.getcwd())

embedding_function = OllamaEmbeddings(base_url="http://localhost:11434", model=AGENT_MODEL)

vector_db = Chroma(
    client=client,
    collection_name="langchain",
    embedding_function=embedding_function
)

llm = Ollama(base_url="http://localhost:11434", model=AGENT_MODEL, keep_alive="-1")

question_template = """Your job is to answer questions about CVs and resumes based on the below context.
If the question or input is not related to CVs or resumes please let the user know.

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

retriever = vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})

document_chain = create_stuff_documents_chain(llm=llm, prompt=question_prompt)

question_vector_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
