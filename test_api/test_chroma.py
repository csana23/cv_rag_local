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
from ragatouille import RAGPretrainedModel

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

question_template = """
[INST] Your sole job is to answer questions about CVs and resumes based on the below context.
If the question is not related to the content of a resume, past job experiences or skills, you can say you don't know.
If the input is not a question related to a resume, let the user know. 
Do not provide answers that are not related to the input.
Keep your answers concise and to the point.
Do not provide more information than what is asked for.
If the context is not relevant to the question, you can say you don't know.
If the context contains metadata about the source file, the name of the file might indicate the 
candidate's name. You find this in 'source tag' of the context. Take the context's metadata (the file source) into account when 
answering questions about multiple candidate's at the same time.
Each file contains information about strictly one candidate. Do not mix-and-match information from multiple files.
If something is not explicitly mentioned in the candidate's CV/resume, do not infer further information about
it on your own! Do NOT make assumptions!  [/INST]
Context: 
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

query = "How would you rate Imre Szucs' skills in data analytics?"

relevant_docs = vector_db.similarity_search(query=query, k=20)
relevant_docs = [str((doc.page_content, doc.metadata)) for doc in relevant_docs]

reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

relevant_docs = reranker.rerank(query, relevant_docs, k=5)
relevant_docs = [doc["content"] for doc in relevant_docs]

relevant_docs = relevant_docs[:5]

context = "\nExtracted documents:\n"
context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

print("context:\n", context)

# retriever = vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.35, "k": 5})

# document_chain = create_stuff_documents_chain(llm=llm, prompt=question_prompt)

question_prompt = question_prompt.format(context=context, input=query)

answer = llm.invoke(question_prompt)

print(answer)

# question_vector_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

# result = question_vector_chain.invoke({"input": "Did Richard Csanaki complete his MSc studies?"})

"""
query_text = "Did Richard Csanaki complete his MSc studies?"

results = db.similarity_search_with_relevance_scores(
    query=query_text, 
    k= 3,
    score_threshold=0.3
)

print(results)
"""





