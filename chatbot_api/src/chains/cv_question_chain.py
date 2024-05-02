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

url = "0.0.0.0" if os.getenv("LOCAL") == "yes" else "host.docker.internal"

client = chromadb.HttpClient(host=url, port=8000)

print("listing chromadb collections from chain:", client.list_collections())

# persist_directory = "chroma"

print("this is from the chain")
print(os.getcwd())

# embedding_function = OllamaEmbeddingFunction(url="http://host.docker.internal:11434", model_name=os.getenv("AGENT_MODEL"))
embedding_function = OllamaEmbeddings(base_url="http://" + url + ":11434", model="phi3")

vector_db = Chroma(
    client=client,
    collection_name="resume_collection",
    embedding_function=embedding_function
)

llm = Ollama(base_url="http://" + url + ":11434", model="phi3", keep_alive="-1")

question_template = """Your sole job is to answer questions about resumes.
Use only the following context to answer the questions.
Be as detailed as possible in your responses but don't make up any information that is not
from the context. If you don't know the answer, say you don't know. You are not permitted to make up information.
For instance, the user can ask questions like "What is the candidate's education level?" or
"How long they have been working in a given field?". Take into account all their job experiences,
education, and skills when answering questions.
Use the entire prompt as input to the tool. For instance, if the prompt is
"Did he complete his MSc studies?", the input should be
"Did he complete his MSc studies?".
{context}
"""

question_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=question_template
    )
)

question_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["input"], template="{question}"
    )
)

messages=[question_system_prompt, question_human_prompt]

question_prompt = ChatPromptTemplate(
    input_variables=["context", "input"], messages=messages
)

retriever = vector_db.as_retriever(k=3)

document_chain = create_stuff_documents_chain(llm=llm, prompt=question_prompt)

question_vector_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)


