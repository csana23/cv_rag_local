from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain import hub
import ollama
import os
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent

ollama_client = ollama.Client(host="http://localhost:11434")

embedding_function = OllamaEmbeddingFunction(url="http://host.docker.internal:11434", model_name=os.getenv("AGENT_MODEL"))

print(embedding_function)

tools = [
    Tool(
        name="AskCVQuestion",
        func=lambda x: str(x),
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



chat_model = ChatOllama(
    base_url="http://host.docker.internal:11434",
    model=os.getenv("AGENT_MODEL"),
    temperature=0,
)

print("chat_model:", chat_model)

cv_rag_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

cv_rag_agent = create_tool_calling_agent(
    llm=chat_model,
    prompt=cv_rag_agent_prompt,
    tools=tools,
)

def get_models():
  models = [model['name'] for model in ollama_client.list()['models']]

  print(models)

  if 'phi3:latest' in models:
      print("model exists")
  else:
      print("model does not exist, pulling")


# response = ollama_client.embeddings(model="phi3", prompt="text to embed")
# embedding = response["embedding"]
# print(embedding)

'''
response = ollama_client.chat(model='phi3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response)
'''
