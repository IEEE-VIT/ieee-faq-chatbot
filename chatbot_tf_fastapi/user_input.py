from pydantic import BaseModel
class user_input(BaseModel):
    query: str
