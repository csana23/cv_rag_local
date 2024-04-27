import os

from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

AGENT_MODEL = os.getenv("AGENT_MODEL")

persist_directory = "../../../chroma/"

vector_db = Chroma(
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings()
)

question_template = """Your job is to use resumes 
provided in the data folder to answer questions about their contents.
Use the following context to answer the questions.
Be as detailed as possible in your responses but don't make up any information that is not
from the context. If you don't know the answer, say you don't know.
{context}
"""

question_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=question_template
    )
)

question_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"], template="{question}"
    )
)

question_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=[question_system_prompt, question_human_prompt]
)

question_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=AGENT_MODEL, temperature=0),
    chain_type="stuff",
    retriever=vector_db.as_retriever(k=3)
)

question_vector_chain.combine_documents_chain.llm_chain.prompt = question_prompt