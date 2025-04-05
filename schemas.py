from pydantic import BaseModel

class APIKeyRequest(BaseModel):
    api_key: str

class QueryRequest(BaseModel):
    question: str
