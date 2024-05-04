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
from ragatouille import RAGPretrainedModel

class ChainBuilder:
    AGENT_MODEL = os.getenv("AGENT_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

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

    def __init__(self, host: str = "127.0.0.1"):
        self.client = chromadb.HttpClient(host=host, port=8000)
        self.embedding_function = OllamaEmbeddings(base_url=f"http://{host}:11434", model=ChainBuilder.EMBEDDING_MODEL)
        self.vector_db = Chroma(
            client=self.client,
            collection_name="resume_collection",
            embedding_function=self.embedding_function,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.llm = Ollama(base_url=f"http://{host}:11435", model=ChainBuilder.AGENT_MODEL, keep_alive="-1m", temperature=0.0)
        self.reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0", n_gpu=1)

    def process_question(self, query: str):
        question_system_prompt = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context"], template=ChainBuilder.question_template
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

        # retrieve relevant documents
        relevant_docs = self.vector_db.similarity_search(query=query, k=20)
        relevant_docs = [str((doc.page_content, doc.metadata)) for doc in relevant_docs]

        relevant_docs = self.reranker.rerank(query, relevant_docs, k=5)
        relevant_docs = [doc["content"] for doc in relevant_docs]

        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

        question_prompt = question_prompt.format(context=context, input=query)

        answer = str(self.llm.invoke(question_prompt))

        return {"input": query, "context": context, "answer": answer}
