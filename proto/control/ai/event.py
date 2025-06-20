from pydantic import BaseModel

class Event(BaseModel):
    # Represents a meaningul event that happened in the environment, worth the attention of system 2
    description: str