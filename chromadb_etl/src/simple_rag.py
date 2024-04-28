import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import dotenv
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "chroma"

dotenv.load_dotenv("../../.env")

review_template_str = """Your sole job is to answer questions about resumes.
Use only the following context to answer the questions.
Be as detailed as possible in your responses but don't make up any information that is not
from the context. 
If you don't know the answer, say you don't know. You are not permitted to make up information.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(),
)

retriever = vector_db.as_retriever(k=5)

review_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    |  review_prompt_template
    | chat_model
    | StrOutputParser()
)

question = "Did Richard Csanaki complete his MSc studies?"

print(review_chain.invoke(question))

