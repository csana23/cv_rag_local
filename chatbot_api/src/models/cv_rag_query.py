from pydantic import BaseModel

class CVQueryInput(BaseModel):
    text: str


class CVQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]