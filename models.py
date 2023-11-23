from pydantic import BaseModel

class Prediction_Input(BaseModel):
    id: int
    text_input: str

class Prediction_Output(BaseModel):
    id: int
    text_input: str
    pred: float