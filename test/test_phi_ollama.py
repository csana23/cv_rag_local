from langchain_community.llms import Ollama

llm = Ollama(model="phi3")
print(llm.invoke("Hello!"))