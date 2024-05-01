from pydantic import BaseModel

class CVQueryInput(BaseModel):
    text: str


class CVQueryOutput(BaseModel):
    text: str