import chromadb
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import ollama
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    StructuredQueryOutputParser,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain_community.document_loaders import DirectoryLoader

DATA_PATH = "../chromadb_etl/data"

AGENT_MODEL = os.getenv("AGENT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

client = chromadb.HttpClient(host="127.0.0.1", port=8000)
print("listing chromadb collections from chain:", client.list_collections())

ollama_client = ollama.Client(host="http://127.0.0.1:11434")

embedding_function = OllamaEmbeddings(base_url="http://127.0.0.1:11434", model=EMBEDDING_MODEL)
llm = Ollama(base_url="http://127.0.0.1:11435", model=AGENT_MODEL, keep_alive="-1m", temperature=0.0)
response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt="test prompt", keep_alive="-1m")

# Create a ChromaDB instance
vector_db = Chroma(
    client=client,
    collection_name="resume_collection",
    embedding_function=embedding_function,
    collection_metadata={"hnsw:space": "cosine"}
)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents

documents = load_documents(DATA_PATH)

question_template = """
[INST] Your sole job is to answer questions about CVs and resumes based on the below context.
If the question is not related to the content of a resume, past job experiences or skills, you can say you don't know.
If the input is not a question related to a resume, let the user know. 
Do not provide answers that are not related to the input.
Keep your answers concise and to the point.
Do not provide more information than what is asked for.
If the context is not relevant to the question, you can say you don't know.
If the context contains metadata about the source file, the name of the file might indicate the 
candidate's name. Take the context's metadata (the file source) into account. Each file contains
information about one candidate. Do not mix-and-match information from multiple files. [/INST]
Context: {context}
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

retriever = vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.3, "k": 10})

document_chain = create_stuff_documents_chain(llm=llm, prompt=question_prompt)

question_vector_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)


"""
query_text = "Did Richard Csanaki complete his MSc studies?"

results = db.similarity_search_with_relevance_scores(
    query=query_text, 
    k= 3,
    score_threshold=0.3
)

print(results)
"""





