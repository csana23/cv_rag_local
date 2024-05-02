from pydantic import BaseModel
from langchain.schema import Document

class CVQueryInput(BaseModel):
    text: str

class CVQueryOutput(BaseModel):
    input: str
    context: list[Document]
    answer: str