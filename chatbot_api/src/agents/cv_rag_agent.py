import os

from chains.cv_question_chain import question_vector_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_openai_functions_agent
from langchain_openai import ChatOpenAI

AGENT_MODEL = os.getenv("AGENT_MODEL")

cv_rag_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

'''
cv_rag_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer any use questions based solely on the context below: {context}"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
'''



tools = [
    Tool(
        name="AskCVQuestion",
        func=question_vector_chain.invoke,
        description="""Useful for answering questions about resumes. 
        For instance, you can ask questions like "What is the candidate's education level?" or
        "How long they have been working in a given field?". Take into account all their job experiences,
        education, and skills when answering questions.
        Use the entire prompt as input to the tool. For instance, if the prompt is
        "Did he complete his MSc studies?", the input should be
        "Did he complete his MSc studies?".
        """,
    )
]

chat_model = ChatOpenAI(
    model=AGENT_MODEL,
    temperature=0,
)

cv_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=cv_rag_agent_prompt,
    tools=tools,
)

cv_rag_agent_executor = AgentExecutor(
    agent=cv_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True
)