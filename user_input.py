from pydantic import BaseModel
class user_input(BaseModel):  
    query: str                      #take user query as string input
